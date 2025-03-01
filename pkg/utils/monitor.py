"""CPU Monitoring Module

This module provides functionality for monitoring CPU usage including:
- Per-core CPU utilization percentages
- Average CPU usage across cores
- Configurable monitoring duration

The module is designed for:
- System performance monitoring
- Resource utilization tracking
- CPU load analysis
"""

import statistics
import time

import psutil


def monitor_cpu_usage(duration=5):
    """Monitor CPU usage statistics over a specified duration.

    Samples CPU utilization metrics at 1 second intervals and prints:
    - Median CPU usage across all cores
    - Per-core CPU usage percentages

    Args:
        duration: Number of seconds to monitor for (default: 5)

    Returns:
        None - Results are printed to stdout
    """
    for _ in range(duration):
        cpu_percentages = psutil.cpu_percent(interval=1, percpu=True)
        print(f"CPU Usage avg Core: {statistics.median(cpu_percentages)}")
        print(f"CPU Usage per Core: {cpu_percentages}%")
        time.sleep(1)
