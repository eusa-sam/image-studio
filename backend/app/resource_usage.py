"""Per-request sampled worker memory; never confuse it with a lifetime process peak."""
import ctypes
from functools import wraps
import json
import os
from pathlib import Path
import threading
import time


def process_memory():
    if os.name != 'nt':
        raise OSError('Process memory sampling is currently Windows-only')
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ('peak_working_set', 'working_set',
            'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged', 'commit', 'peak_commit', 'private_commit')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    query = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    query.restype = ctypes.c_int
    if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'working_set_bytes': counters.working_set, 'private_commit_bytes': counters.private_commit}


class MemorySamples:
    def __init__(self, interval=.1):
        self.interval = interval
        self.samples = 0
        self.peak = {}
        self.first = self.last = None
        self.error = None
        self.stopping = threading.Event()

    def observe(self):
        try:
            row = process_memory()
        except OSError as error:
            self.error = str(error)
            return
        self.samples += 1
        if self.first is None:
            self.first = row
        self.last = row
        for key, value in row.items():
            self.peak[key] = max(self.peak.get(key, 0), value)

    def loop(self):
        while not self.stopping.wait(self.interval):
            self.observe()

    def result(self):
        return dict(scope='worker_process', method='sampled', interval_seconds=self.interval,
                    samples=self.samples, start=self.first, peak=self.peak, end=self.last,
                    error=self.error)


def record_resources(function):
    @wraps(function)
    def measured(request_path, *args, **kwargs):
        memory = MemorySamples()
        started = time.monotonic()
        memory.observe()
        sampler = threading.Thread(target=memory.loop, daemon=True)
        sampler.start()
        try:
            return function(request_path, *args, **kwargs)
        finally:
            memory.stopping.set()
            sampler.join()
            memory.observe()
            resources = dict(worker_wall_seconds=time.monotonic()-started, ram=memory.result())
            folder = Path(request_path).parent
            try:
                (folder/'resources.json').write_text(json.dumps(resources, indent=2), encoding='utf-8')
                report_path = folder/'report.json'
                if report_path.exists():
                    report = json.loads(report_path.read_text(encoding='utf-8'))
                    report['resources'] = resources
                    temporary = folder/'report.resources.tmp'
                    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
                    temporary.replace(report_path)
            except (OSError, ValueError, TypeError) as error:
                print('Resource report unavailable: '+str(error), flush=True)
    return measured
