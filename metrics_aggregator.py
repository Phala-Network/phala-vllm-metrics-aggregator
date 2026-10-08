"""Authenticated, complete-only Prometheus federation of four PIG diagnostics."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from decimal import Decimal
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import threading
import time
import urllib.request

from prometheus_client.parser import text_string_to_metric_families

NAME = r'[a-zA-Z_:][a-zA-Z0-9_:]*'
DECLARATION = re.compile(r'^# (HELP|TYPE) (' + NAME + r')(?: (.*))?$')
SAMPLE_NAME = re.compile('^(' + NAME + ')')
MAX_BYTES = 4 * 1024 * 1024
TIMEOUT = 3.0
MARKER = '# --- Backend Metrics ---'


class InvalidMetrics(ValueError):
    pass


def quote(value):
    return '"' + value.replace('\\', '\\\\').replace('\n', '\\n').replace('"', '\\"') + '"'


def sample_tail(line, name):
    """Preserve the original numeric value/timestamp, avoiding float rounding."""
    pos = len(name)
    if pos < len(line) and line[pos] == '{':
        quoted = escaped = False
        for pos in range(pos + 1, len(line)):
            char = line[pos]
            if escaped:
                escaped = False
            elif char == '\\' and quoted:
                escaped = True
            elif char == '"':
                quoted = not quoted
            elif char == '}' and not quoted:
                pos += 1
                break
        else:
            raise InvalidMetrics('labels_not_closed')
    tail = line[pos:]
    if not tail or not tail[0].isspace():
        raise InvalidMetrics('invalid_sample_separator')
    return tail.strip()


def parse_document(text, replica):
    if text.count(MARKER) != 1:
        raise InvalidMetrics('missing_pig_backend_section')
    backend = text.split(MARKER, 1)[1]
    if '# failed to fetch backend metrics:' in backend or '# backend metrics status ' in backend:
        raise InvalidMetrics('pig_backend_fetch_failed')
    if not any(line.startswith('vllm:') for line in backend.splitlines()):
        raise InvalidMetrics('missing_vllm_backend_samples')
    # This deployment pins vLLM's Rust OpenMetrics endpoint. PIG can swallow an
    # upstream copy error, so a syntactically valid prefix is not a complete scrape.
    openmetrics = True
    if sum(line.strip() == '# EOF' for line in backend.splitlines()) != 1 or not backend.rstrip().endswith('# EOF'):
        raise InvalidMetrics('invalid_openmetrics_termination')
    metadata, samples, seen = {}, [], set()
    in_backend = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('#'):
            if line == MARKER:
                in_backend = True
            declaration = DECLARATION.fullmatch(line)
            if declaration:
                kind, name, value = declaration.groups()
                value = value or ''
                entry = metadata.setdefault(name, {})
                if kind in entry and entry[kind] != value:
                    raise InvalidMetrics('conflicting_metadata')
                if kind == 'TYPE' and value not in ('counter', 'gauge', 'histogram', 'summary', 'untyped'):
                    raise InvalidMetrics('unsupported_metric_type')
                entry[kind] = value
                if len(metadata) > 10000:
                    raise InvalidMetrics('too_many_metric_families')
            elif line.startswith(('# HELP ', '# TYPE ')):
                raise InvalidMetrics('invalid_metadata')
            # EOF/UNIT and free comments are not part of Prometheus 0.0.4 output.
            continue
        match = SAMPLE_NAME.match(line)
        if not match:
            raise InvalidMetrics('invalid_metric_name')
        name = match.group(1)
        try:
            parsed = [sample for family in text_string_to_metric_families(line) for sample in family.samples]
        except (ValueError, TypeError):
            raise InvalidMetrics('invalid_metric_sample') from None
        if len(parsed) != 1 or parsed[0].name != name:
            raise InvalidMetrics('invalid_sample_count')
        labels = dict(parsed[0].labels)
        if 'replica' in labels or '__name__' in labels:
            raise InvalidMetrics('reserved_label_collision')
        if not all(re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*', label) for label in labels):
            raise InvalidMetrics('unsupported_label_name')
        identity = (name, tuple(sorted(labels.items())))
        if identity in seen:
            raise InvalidMetrics('duplicate_series')
        seen.add(identity)
        labels['replica'] = str(replica)
        labels_text = ','.join(key + '=' + quote(value) for key, value in sorted(labels.items()))
        tail = sample_tail(line, name)
        fields = tail.split()
        if len(fields) == 2 and openmetrics and in_backend:
            # OpenMetrics timestamps are seconds; text 0.0.4 uses integer milliseconds.
            tail = fields[0] + ' ' + str(int(Decimal(fields[1]) * 1000))
        samples.append((name, name + '{' + labels_text + '} ' + tail))
        if len(samples) > 20000:
            raise InvalidMetrics('too_many_samples')
    names = {name for name, _ in samples}
    # Rust emits OpenMetrics counter TYPE/HELP on the base name. Prometheus
    # 0.0.4 declarations name the _total sample; values and names stay untouched.
    for name in list(metadata):
        if metadata[name].get('TYPE') == 'counter' and name not in names and name + '_total' in names:
            normalized = name + '_total'
            if normalized in metadata and metadata[normalized] != metadata[name]:
                raise InvalidMetrics('counter_metadata_collision')
            metadata[normalized] = metadata.pop(name)
    return metadata, samples


def merge_documents(documents):
    if set(documents) != {0, 1, 2, 3}:
        raise InvalidMetrics('not_four_replicas')
    metadata, samples = {}, []
    for replica in range(4):
        declarations, parsed = parse_document(documents[replica], replica)
        for name, entry in declarations.items():
            combined = metadata.setdefault(name, {})
            for kind, value in entry.items():
                if kind in combined and combined[kind] != value:
                    raise InvalidMetrics('cross_replica_metadata_conflict')
                combined[kind] = value
        samples.extend(parsed)
    groups = {}
    for name, line in samples:
        candidates = []
        if name in metadata:
            candidates.append(name)
        for suffix in ('_bucket', '_count', '_sum'):
            if name.endswith(suffix):
                parent = name[:-len(suffix)]
                kind = metadata.get(parent, {}).get('TYPE')
                if kind == 'histogram' or (kind == 'summary' and suffix != '_bucket'):
                    candidates.append(parent)
        if len(set(candidates)) > 1:
            raise InvalidMetrics('ambiguous_metric_family')
        family = candidates[0] if candidates else name
        groups.setdefault(family, []).append(line)
    output = []
    for name in sorted(set(metadata) | set(groups)):
        for kind in ('HELP', 'TYPE'):
            if kind in metadata.get(name, {}):
                output.append('# ' + kind + ' ' + name + ' ' + metadata[name][kind])
        output.extend(groups.get(name, []))
    return ('\n'.join(output) + '\n').encode()


class Aggregator:
    def __init__(self, token, timeout=TIMEOUT):
        self.token = token
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='metrics')
        self.scrape_slot = threading.BoundedSemaphore(1)
        # Never send the credential through environment-configured proxy servers.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def fetch(self, replica):
        deadline = time.monotonic() + self.timeout
        request = urllib.request.Request(f'http://127.0.0.1:{8200 + replica}/v1/metrics', headers={'Authorization': 'Bearer ' + self.token, 'Accept': 'text/plain; version=0.0.4'})
        with self.opener.open(request, timeout=self.timeout) as response:
            if response.status != 200:
                raise InvalidMetrics('upstream_status')
            content_length = response.headers.get('Content-Length')
            chunks, size = [], 0
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError('upstream_deadline')
                chunk = response.read1(min(65536, MAX_BYTES + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_BYTES:
                    raise InvalidMetrics('upstream_body_too_large')
            if content_length is not None and size != int(content_length):
                raise InvalidMetrics('truncated_http_body')
        text = b''.join(chunks).decode('utf-8', errors='strict')
        parse_document(text, replica)
        return text

    def collect(self):
        if not self.scrape_slot.acquire(blocking=False):
            return 503, b'{"error":"scrape_busy"}\n'
        try:
            futures = {self.executor.submit(self.fetch, n): n for n in range(4)}
            done, pending = wait(futures, timeout=self.timeout + 0.25)
            failed = [futures[future] for future in pending]
            documents = {}
            for future in done:
                try:
                    documents[futures[future]] = future.result()
                except Exception:
                    failed.append(futures[future])
            for future in pending:
                future.cancel()
            if failed:
                return 503, (json.dumps({'error': 'incomplete_metrics', 'failed_replicas': sorted(failed), 'expected_replicas': 4}) + '\n').encode()
            try:
                return 200, merge_documents(documents)
            except InvalidMetrics:
                return 503, b'{"error":"incompatible_metrics","expected_replicas":4}\n'
        finally:
            self.scrape_slot.release()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_body(self, status, body, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            if self.command != 'HEAD':
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def do_GET(self):
        if self.path == '/healthz':
            self.send_body(200, b'ok\n', 'text/plain')
            return
        if self.path != '/v1/metrics':
            self.send_body(404, b'not found\n', 'text/plain')
            return
        auth = self.headers.get_all('Authorization', [])
        if len(auth) != 1 or not hmac.compare_digest(auth[0].encode(), ('Bearer ' + self.server.aggregator.token).encode()):
            self.send_body(401, b'unauthorized\n', 'text/plain')
            return
        status, body = self.server.aggregator.collect()
        self.send_body(status, body, 'text/plain; version=0.0.4; charset=utf-8' if status == 200 else 'application/json')

    do_HEAD = do_GET


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, aggregator, address=('127.0.0.1', 30081)):
        self.aggregator = aggregator
        self.connection_slots = threading.BoundedSemaphore(8)
        super().__init__(address, Handler)

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(5)
        return request, address

    def process_request(self, request, address):
        if not self.connection_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.connection_slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.connection_slots.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--healthcheck', action='store_true')
    args = parser.parse_args()
    if args.healthcheck:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open('http://127.0.0.1:30081/healthz', timeout=3) as response:
            return 0 if response.status == 200 else 1
    token = os.environ.get('TOKEN')
    if not token:
        raise SystemExit('TOKEN is required')
    Server(Aggregator(token)).serve_forever()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
