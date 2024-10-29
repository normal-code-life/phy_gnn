import statistics
import time

import psutil


def monitor_cpu_usage(duration=5):
    for _ in range(duration):
        cpu_percentages = psutil.cpu_percent(interval=1, percpu=True)
        print(f"CPU Usage avg Core: {statistics.median(cpu_percentages)}")
        print(f"CPU Usage per Core: {cpu_percentages}%")
        time.sleep(1)
