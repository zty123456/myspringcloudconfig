#!/bin/bash
# CPU Load Controller — Shell 版本
# 使用方式：
#   ./cpu_load_controller.sh start [target_percent] [num_cores]
#   ./cpu_load_controller.sh stop
#
# 示例：
#   ./cpu_load_controller.sh start 40 4   # 目标40%，使用4核心
#   ./cpu_load_controller.sh start 40     # 目标40%，自动检测核心数
#   ./cpu_load_controller.sh stop         # 停止所有负载

PID_FILE="./cpu_load_pids.txt"
LOCK_FILE="./cpu_load.lock"

# 获取系统核心数
get_total_cores() {
    if command -v nproc &> /dev/null; then
        nproc
    elif [ -f /proc/cpuinfo ]; then
        grep -c ^processor /proc/cpuinfo
    else
        echo 4  # 默认值
    fi
}

# 单个 CPU worker 函数
cpu_worker() {
    local work_ratio=$1
    local worker_id=$2
    local cycle=0.1  # 100ms 周期
    local work_time=$(echo "$cycle * $work_ratio" | bc -l)
    local sleep_time=$(echo "$cycle * (1 - $work_ratio)" | bc -l)

    echo "[worker $worker_id] start, work_ratio=$work_ratio"

    while true; do
        # 工作阶段：执行计算任务
        if [ "$work_time" != "0" ]; then
            local start_time=$(date +%s.%N)
            local elapsed=0
            while [ $(echo "$elapsed < $work_time" | bc -l) -eq 1 ]; do
                # 简单数学运算
                awk 'BEGIN { for(i=0; i<1000; i++) x=sqrt(12345)*sin(1.23)+cos(2.34) }'
                elapsed=$(echo "$(date +%s.%N) - $start_time" | bc -l)
            done
        fi

        # 休眠阶段
        if [ "$sleep_time" != "0" ] && [ "$sleep_time" != "0.0" ]; then
            sleep $sleep_time
        fi
    done
}

# 启动 CPU 负载
start_load() {
    local target=$1
    local cores=$2

    # 参数校验
    if [ -z "$target" ]; then
        target=40
    fi

    if [ -z "$cores" ] || [ "$cores" -le 0 ]; then
        cores=$(get_total_cores)
    fi

    local total_cores=$(get_total_cores)
    if [ "$cores" -gt "$total_cores" ]; then
        cores=$total_cores
    fi

    # 计算 work_ratio
    # 总CPU% ≈ cores * work_ratio * 100 / total_cores
    local work_ratio=$(echo "scale=3; $target * $total_cores / ($cores * 100)" | bc -l)
    
    # 限制 work_ratio 在 [0, 1] 范围
    if [ $(echo "$work_ratio > 1" | bc -l) -eq 1 ]; then
        work_ratio=1
    fi

    echo "[config] total_cores=$total_cores, num_cores=$cores"
    echo "[config] work_ratio=$work_ratio"
    echo "[config] expected CPU ≈ $(echo "$cores * $work_ratio * 100 / $total_cores" | bc -l)%"

    # 清理旧的 PID 文件
    rm -f "$PID_FILE"

    # 启动 workers
    local pids=""
    for i in $(seq 1 $cores); do
        # 在后台运行 worker
        (
            work_ratio=$work_ratio
            worker_id=$i
            cycle=0.1
            work_time=$(echo "$cycle * $work_ratio" | bc -l)
            sleep_time=$(echo "$cycle * (1 - $work_ratio" | bc -l)

            while true; do
                if [ $(echo "$work_time > 0" | bc -l) -eq 1 ]; then
                    awk 'BEGIN { for(i=0; i<5000; i++) x=sqrt(12345)*sin(1.23)+cos(2.34)+log(100)+exp(0.5) }' >/dev/null 2>&1
                fi
                if [ $(echo "$sleep_time > 0" | bc -l) -eq 1 ]; then
                    sleep $sleep_time
                fi
            done
        ) &
        local pid=$!
        pids="$pids $pid"
        echo "[worker $i] PID=$pid"
    done

    # 保存 PID
    for pid in $pids; do
        echo $pid >> "$PID_FILE"
    done

    echo "[start] $cores workers started, PIDs saved to $PID_FILE"
    echo "[info] 使用 'top' 或 'htop' 查看 CPU 使用率"
    echo "[info] 使用 '$0 stop' 停止所有进程"
}

# 停止 CPU 负载
stop_load() {
    if [ ! -f "$PID_FILE" ]; then
        echo "[stop] PID 文件不存在，无进程需要停止"
        return 0
    fi

    local count=0
    while read pid; do
        if [ -n "$pid" ]; then
            if kill -0 $pid 2>/dev/null; then
                kill $pid 2>/dev/null
                echo "[stop] PID $pid 已终止"
                count=$((count + 1))
            else
                echo "[stop] PID $pid 已不存在"
            fi
        fi
    done < "$PID_FILE"

    rm -f "$PID_FILE"
    echo "[stop] 已停止 $count 个进程"
}

# 监控 CPU 使用率（可选）
monitor_cpu() {
    if command -v top &> /dev/null; then
        echo "[monitor] 按 Ctrl+C 退出监控"
        top -d 2
    elif command -v htop &> /dev/null; then
        htop
    else
        echo "[monitor] 请安装 top 或 htop 查看CPU使用率"
        echo "[monitor] Windows: 使用任务管理器"
        echo "[monitor] Linux: 使用 top/htop 命令"
    fi
}

# 主入口
main() {
    local action=$1

    case "$action" in
        start)
            local target=$2
            local cores=$3
            start_load "$target" "$cores"
            ;;
        stop)
            stop_load
            ;;
        monitor)
            monitor_cpu
            ;;
        status)
            if [ -f "$PID_FILE" ]; then
                echo "[status] 运行中的进程 PID:"
                cat "$PID_FILE"
                echo "[status] 使用 'top' 查看CPU使用率"
            else
                echo "[status] 未在运行"
            fi
            ;;
        *)
            echo "用法:"
            echo "  $0 start [target_percent] [num_cores]  — 启动CPU负载"
            echo "  $0 stop                                — 停止CPU负载"
            echo "  $0 monitor                             — 监控CPU使用率"
            echo "  $0 status                              — 查看状态"
            echo ""
            echo "示例:"
            echo "  $0 start 40      — 目标40%CPU，自动检测核心数"
            echo "  $0 start 40 4    — 目标40%CPU，使用4核心"
            echo "  $0 stop          — 停止所有负载"
            ;;
    esac
}

main "$@"