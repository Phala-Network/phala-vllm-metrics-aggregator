import http.client
import io
import threading
import time
import unittest
from unittest.mock import patch

from metrics_aggregator import Aggregator, InvalidMetrics, Server, merge_documents


FIXTURE = '''pig_info{version="test"} 1
pig_rejected_total 9007199254740993
# --- Backend Metrics ---
# HELP vllm:tokens Tokens generated.
# TYPE vllm:tokens counter
vllm:tokens_total{model_name="model"} 9007199254740993
# HELP vllm:latency Request latency.
# TYPE vllm:latency histogram
vllm:latency_bucket{le="1",model_name="model"} 2
vllm:latency_bucket{le="+Inf",model_name="model"} 3
vllm:latency_count{model_name="model"} 3
vllm:latency_sum{model_name="model"} 4.25
# TYPE vllm:summary summary
vllm:summary{quantile="0.9",model_name="model"} 7
vllm:summary_count{model_name="model"} 3
vllm:summary_sum{model_name="model"} 8
# EOF
'''


def four(text=FIXTURE):
    return {n: text for n in range(4)}


class MergeTests(unittest.TestCase):
    def test_four_replicas_without_synthetic_totals_or_precision_loss(self):
        text = merge_documents(four()).decode()
        self.assertEqual(text.count('pig_rejected_total{'), 4)
        self.assertEqual(text.count('9007199254740993'), 8)
        for n in range(4):
            self.assertIn('replica="' + str(n) + '"', text)
        self.assertEqual(text.count('# TYPE vllm:tokens_total counter'), 1)
        self.assertNotIn('# TYPE vllm:tokens counter', text)
        self.assertEqual(text.count('quantile="0.9"'), 4)
        self.assertNotIn('# EOF', text)

    def test_histogram_and_summary_family_are_contiguous(self):
        text = merge_documents(four()).decode()
        histogram = text.split('# TYPE vllm:latency histogram\n')[1].split('# TYPE vllm:summary summary')[0]
        self.assertEqual(len(histogram.strip().splitlines()), 16)
        self.assertTrue(all(line.startswith('vllm:latency_') for line in histogram.strip().splitlines()))

    def test_labels_escaping_and_existing_instance_preserved(self):
        source = FIXTURE.replace('pig_rejected_total 9007199254740993', 'pig_rejected_total{instance="node",job="serving",detail="brace}comma,quote\\\"slash\\\\"} 9')
        text = merge_documents(four(source)).decode()
        self.assertIn('detail="brace}comma,quote\\\"slash\\\\",instance="node",job="serving",replica="0"', text)

    def test_metadata_conflict_rejected(self):
        for altered in [FIXTURE.replace('Tokens generated.', 'Conflicting help.'), FIXTURE.replace('vllm:latency histogram', 'vllm:latency summary')]:
            documents = four()
            documents[2] = altered
            with self.assertRaises(InvalidMetrics):
                merge_documents(documents)

    def test_reserved_labels_duplicate_series_and_duplicate_labels_rejected(self):
        for sample in ['pig_info{replica="old"} 1', 'pig_info{a="x",a="y"} 1', 'pig_info 1\npig_info 2']:
            source = FIXTURE.replace('pig_info{version="test"} 1', sample)
            with self.assertRaises(InvalidMetrics):
                merge_documents(four(source))

    def test_missing_backend_and_hidden_pig_failure_rejected(self):
        for source in [FIXTURE.replace('# --- Backend Metrics ---', '# absent'), 'pig_info 1\n# --- Backend Metrics ---\n# failed to fetch backend metrics: test\n', 'pig_info 1\n# --- Backend Metrics ---\n# backend metrics status 500\n']:
            with self.assertRaises(InvalidMetrics):
                merge_documents(four(source))

    def test_openmetrics_timestamp_seconds_to_milliseconds(self):
        text = merge_documents(four(FIXTURE.replace(' 4.25\n', ' 4.25 1700000000.125\n'))).decode()
        self.assertIn('4.25 1700000000125\n', text)

    def test_truncated_valid_line_prefix_without_eof_rejected(self):
        with self.assertRaises(InvalidMetrics):
            merge_documents(four(FIXTURE.replace('# EOF\n', '')))


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.aggregator = Aggregator('test-token', timeout=0.02)

    def tearDown(self):
        self.aggregator.executor.shutdown(wait=True, cancel_futures=True)

    def test_failed_replica_is_explicit_503_without_partial_samples(self):
        def fetch(n):
            if n == 2:
                raise RuntimeError('upstream failure')
            return FIXTURE
        with patch.object(self.aggregator, 'fetch', side_effect=fetch):
            status, body = self.aggregator.collect()
        self.assertEqual(status, 503)
        self.assertIn(b'"failed_replicas": [2]', body)
        self.assertNotIn(b'pig_info', body)

    def test_parallel_fetch_and_bounded_timeout(self):
        def fetch(n):
            time.sleep(0.5)
            return FIXTURE
        started = time.monotonic()
        with patch.object(self.aggregator, 'fetch', side_effect=fetch):
            status, body = self.aggregator.collect()
        self.assertLess(time.monotonic() - started, 0.4)
        self.assertEqual(status, 503)
        self.assertIn(b'"failed_replicas": [0, 1, 2, 3]', body)

    def test_busy_scrape_does_not_queue_unbounded_work(self):
        self.aggregator.scrape_slot.acquire()
        try:
            self.assertEqual(self.aggregator.collect()[0], 503)
        finally:
            self.aggregator.scrape_slot.release()

    def test_http_content_length_mismatch_rejected(self):
        class Response(io.BytesIO):
            status = 200
            headers = {'Content-Length': str(len(FIXTURE.encode()) + 10)}
        with patch.object(self.aggregator.opener, 'open', return_value=Response(FIXTURE.encode())):
            with self.assertRaisesRegex(InvalidMetrics, 'truncated_http_body'):
                self.aggregator.fetch(0)

    def test_http_auth_and_head(self):
        with patch.object(self.aggregator, 'collect', return_value=(200, b'metric{replica="0"} 1\n')):
            server = Server(self.aggregator, ('127.0.0.1', 0))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for method, headers, expected in [('GET', {}, 401), ('GET', {'Authorization': 'Bearer wrong'}, 401), ('GET', {'Authorization': 'Bearer test-token'}, 200), ('HEAD', {'Authorization': 'Bearer test-token'}, 200)]:
                    connection = http.client.HTTPConnection(*server.server_address, timeout=2)
                    connection.request(method, '/v1/metrics', headers=headers)
                    response = connection.getresponse()
                    self.assertEqual(response.status, expected)
                    payload = response.read()
                    if method == 'HEAD':
                        self.assertEqual(payload, b'')
                        self.assertGreater(int(response.getheader('Content-Length')), 0)
                    connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()
