"""CPU Load Controller — 控制CPU使用率维持在指定水平。

使用方式：
    python cpu_load_controller.py start [--target 40] [--cores 4]
    python cpu_load_controller.py stop

参数说明：
    start  -- 启动CPU负载任务
    stop   -- 停止所有CPU负载进程
    --target  目标CPU使用率百分比 (默认 40)
    --cores   使用核心数 (默认 自动检测所有可用核心)
"""

import argparse
import math
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path

PID_FILE = Path("cpu_load_pids.txt")


def cpu_worker(work_ratio: float, worker_id: int) -> None:
    """单个CPU worker，通过 work/sleep 比例控制负载。

    Args:
        work_ratio: 工作时间占比 (0.0 ~ 1.0)，如 0.4 表示 40% CPU
        worker_id: worker 编号
    """
    cycle_duration = 0.1  # 每个周期 100ms
    work_time = cycle_duration * work_ratio
    sleep_time = cycle_duration * (1.0 - work_ratio)

    print(f"[worker {worker_id}] start, work_ratio={work_ratio:.2f}")

    try:
        while True:
            # 工作：执行简单数学运算
            if work_time > 0:
                start = time.perf_counter()
                while time.perf_counter() - start < work_time:
                    # 简单计算任务
                    _ = math.sqrt(12345.6789) * math.sin(1.234) + math.cos(2.345)

            # 休眠：释放CPU
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print(f"[worker {worker_id}] stopped")


def monitor_cpu_usage(target: float, num_workers: int) -> None:
    """监控进程，持续打印当前CPU使用率（仅用于调试）。"""
    import psutil

    print(f"[monitor] target={target}%, workers={num_workers}")

    while True:
        cpu_percent = psutil.cpu_percent(interval=1.0)
        print(f"[monitor] CPU usage: {cpu_percent:.1f}%")
        time.sleep(2.0)


def start_load(target_percent: float, num_cores: int) -> None:
    """启动CPU负载进程。

    Args:
        target_percent: 目标CPU使用率百分比
        num_cores: 使用核心数
    """
    if target_percent <= 0 or target_percent > 100:
        print(f"[error] target_percent must be in [1, 100], got {target_percent}")
        sys.exit(1)

    # 每个核心的工作比例
    # 总CPU使用率 ≈ num_cores * work_ratio / total_cores
    total_cores = multiprocessing.cpu_count()
    if num_cores > total_cores:
        num_cores = total_cores

    # work_ratio: 每个worker的CPU占用比例
    # 目标：target_percent ≈ num_cores * work_ratio * 100 / total_cores
    # 所以 work_ratio ≈ target_percent * total_cores / (num_cores * 100)
    work_ratio = min(1.0, target_percent * total_cores / (num_cores * 100))

    print(f"[config] total_cores={total_cores}, num_cores={num_cores}")
    print(f"[config] work_ratio per worker={work_ratio:.3f}")
    print(f"[config] expected total CPU ≈ {num_cores * work_ratio * 100 / total_cores:.1f}%")

    # 启动worker进程
    workers = []
    pids = []

    for i in range(num_cores):
        p = multiprocessing.Process(target=cpu_worker, args=(work_ratio, i))
        p.start()
        workers.append(p)
        pids.append(p.pid)

    # 可选：启动监控进程（需要 psutil）
    try:
        import psutil
        monitor = multiprocessing.Process(target=monitor_cpu_usage, args=(target_percent, num_cores))
        monitor.start()
        workers.append(monitor)
        pids.append(monitor.pid)
    except ImportError:
        print("[info] psutil not installed, skipping CPU monitor")

    # 保存PID到文件
    with open(PID_FILE, "w") as f:
        for pid in pids:
            f.write(f"{pid}\n")

    print(f"[start] {len(workers)} processes started, PIDs saved to {PID_FILE}")

    # 等待所有进程（主进程阻塞）
    try:
        for p in workers:
            p.join()
    except KeyboardInterrupt:
        print("\n[interrupt] stopping all workers...")
        for p in workers:
            p.terminate()
        PID_FILE.unlink(missing_ok=True)


def stop_load() -> None:
    """停止所有CPU负载进程。"""
    if not PID_FILE.exists():
        print("[stop] no PID file found, nothing to stop")
        return

    pids = []
    with open(PID_FILE, "r") as f:
        for line in f:
            pid = int(line.strip())
            pids.append(pid)

    print(f"[stop] killing {len(pids)} processes: {pids}")

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"[stop] PID {pid} terminated")
        except ProcessLookupError:
            print(f"[stop] PID {pid} already dead")
        except PermissionError:
            print(f"[stop] PID {pid} permission denied")

    PID_FILE.unlink(missing_ok=True)
    print("[stop] done")


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU Load Controller")
    parser.add_argument("action", choices=["start", "stop"], help="start or stop CPU load")
    parser.add_argument("--target", type=float, default=40.0, help="target CPU usage percent (default: 40)")
    parser.add_argument("--cores", type=int, default=0, help="number of cores to use (default: auto)")

    args = parser.parse_args()

    if args.action == "start":
        num_cores = args.cores if args.cores > 0 else multiprocessing.cpu_count()
        # 限制最大核心数，避免过载
        if num_cores > multiprocessing.cpu_count():
            num_cores = multiprocessing.cpu_count()
        start_load(args.target, num_cores)
    elif args.action == "stop":
        stop_load()


if __name__ == "__main__":
    main()