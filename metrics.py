from prometheus_client import start_http_server, Counter, Gauge, Histogram
from prometheus_client import multiprocess
from prometheus_client import CollectorRegistry
import os
import psutil


def get_registry():
    if not  "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/"

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return registry

def start_metrics(port):
    start_http_server(port, registry=get_registry())


class Metrics:

    my_basic_counter = Counter(
        'my_basic_counter',
        'A basic counter.',
        registry=get_registry()
    )


    h = Histogram('request_latency_seconds',
               'Description of histogram',
                buckets=[0.0001, 0.002, 0.03, 0.4, 1],
                registry=get_registry()
    )   
