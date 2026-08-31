#!/usr/bin/env bash
set -Eeuo pipefail

readonly PORT=8777

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BACKEND_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
RUNTIME_DIR="$BACKEND_DIR/runtime"
PID_FILE="$RUNTIME_DIR/backend.pid"
LOCK_FILE="$RUNTIME_DIR/backend-restart.lock"

fail() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

port_pids() {
  lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | sort -un || true
}

is_this_backend() {
  local pid="$1"
  local process_cwd process_command

  [[ -r "/proc/$pid/cmdline" ]] || return 1
  process_cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  process_command="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  [[ "$process_cwd" == "$BACKEND_DIR" && "$process_command" == *"uvicorn app.main:app"* ]]
}

describe_process() {
  local pid="$1"
  ps -p "$pid" -o pid=,lstart=,cmd= 2>/dev/null || printf 'PID %s（进程信息不可读）\n' "$pid"
}

command -v lsof >/dev/null 2>&1 || fail "缺少 lsof，无法可靠确认 $PORT 端口占用"
command -v pgrep >/dev/null 2>&1 || fail "缺少 pgrep，无法查找后端进程"
command -v flock >/dev/null 2>&1 || fail "缺少 flock，无法防止并发启停"

mkdir -p "$RUNTIME_DIR"
exec 9>"$LOCK_FILE"
flock -n 9 || fail "另一个后端启动、停止或重启操作正在执行"

declare -A stop_pids=()
recorded_pid=""
if [[ -f "$PID_FILE" ]]; then
  IFS= read -r recorded_pid < "$PID_FILE" || true
  if [[ "$recorded_pid" =~ ^[0-9]+$ ]] && kill -0 "$recorded_pid" 2>/dev/null; then
    if is_this_backend "$recorded_pid"; then
      stop_pids["$recorded_pid"]=1
    else
      printf '忽略失真的 PID 文件：PID %s 不是当前目录的后端进程。\n' "$recorded_pid" >&2
    fi
  fi
fi

# 捕获当前项目中尚未开始监听或已经关闭监听但仍未退出的 Uvicorn 进程。
while IFS= read -r pid; do
  if [[ "$pid" =~ ^[0-9]+$ ]] && is_this_backend "$pid"; then
    stop_pids["$pid"]=1
  fi
done < <(pgrep -f 'uvicorn app\.main:app' 2>/dev/null || true)

# 8777 是本服务的专用端口；无论 PID 文件是否正确，都清理它的实际监听者。
while IFS= read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] && stop_pids["$pid"]=1
done < <(port_pids)

if (("${#stop_pids[@]}" == 0)); then
  rm -f "$PID_FILE"
  printf '后端已经停止：%s 端口没有监听进程。\n' "$PORT"
  exit 0
fi

printf '正在完全停止后端进程：\n'
for pid in "${!stop_pids[@]}"; do
  describe_process "$pid"
  kill -TERM "$pid" 2>/dev/null || fail "无法停止 PID $pid，请检查进程权限"
done

for _ in {1..30}; do
  any_alive=false
  for pid in "${!stop_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      any_alive=true
      break
    fi
  done
  [[ "$any_alive" == false ]] && break
  sleep 0.5
done

for pid in "${!stop_pids[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    printf 'PID %s 未在 15 秒内退出，执行强制停止。\n' "$pid" >&2
    kill -KILL "$pid" 2>/dev/null || fail "无法强制停止 PID $pid"
  fi
done

for _ in {1..20}; do
  mapfile -t remaining_pids < <(port_pids)
  (("${#remaining_pids[@]}" == 0)) && break
  sleep 0.25
done
mapfile -t remaining_pids < <(port_pids)
(("${#remaining_pids[@]}" == 0)) \
  || fail "$PORT 端口仍被 PID ${remaining_pids[*]} 占用"

rm -f "$PID_FILE"
printf '后端已完全停止：%s 端口已释放，PID 文件已移除。\n' "$PORT"
printf '任务目录和数据库未被修改：%s\n' "$RUNTIME_DIR"
