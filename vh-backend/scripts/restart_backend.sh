#!/usr/bin/env bash
set -Eeuo pipefail

readonly PORT=8777
readonly HOST=0.0.0.0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BACKEND_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
RUNTIME_DIR="$BACKEND_DIR/runtime"
PYTHON="$BACKEND_DIR/.venv/bin/python"
PID_FILE="$RUNTIME_DIR/backend.pid"
PID_TEMP="$PID_FILE.starting.$$"
LOG_FILE="$RUNTIME_DIR/backend.log"
HEALTH_URL="http://127.0.0.1:$PORT/health"
new_pid=""

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

cleanup_failed_start() {
  local status=$?
  if [[ -n "$new_pid" ]] && kill -0 "$new_pid" 2>/dev/null; then
    kill -TERM "$new_pid" 2>/dev/null || true
  fi
  rm -f "$PID_TEMP"
  return "$status"
}
trap cleanup_failed_start EXIT

command -v lsof >/dev/null 2>&1 || fail "缺少 lsof，无法可靠确认 $PORT 端口占用"
command -v curl >/dev/null 2>&1 || fail "缺少 curl，无法执行健康检查"
command -v flock >/dev/null 2>&1 || fail "缺少 flock，无法防止并发重启"
[[ -x "$PYTHON" ]] || fail "未找到 $PYTHON；只需首次执行 uv sync，无需每次重装环境"

mkdir -p "$RUNTIME_DIR"
exec 9>"$RUNTIME_DIR/backend-restart.lock"
flock -n 9 || fail "另一个后端重启操作正在执行"

cd "$BACKEND_DIR"

# 在停止旧服务前验证现有虚拟环境和新代码可导入，减少无谓停机。
"$PYTHON" -c 'import uvicorn; import app.main' \
  || fail "现有 .venv 或后端代码不可用，请先修复后再重启"

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

while IFS= read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] && stop_pids["$pid"]=1
done < <(port_pids)

if (("${#stop_pids[@]}" > 0)); then
  printf '正在停止占用 %s 端口的进程：\n' "$PORT"
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
fi

for _ in {1..20}; do
  mapfile -t remaining_pids < <(port_pids)
  (("${#remaining_pids[@]}" == 0)) && break
  sleep 0.25
done
mapfile -t remaining_pids < <(port_pids)
(("${#remaining_pids[@]}" == 0)) \
  || fail "$PORT 端口仍被 PID ${remaining_pids[*]} 占用，拒绝启动以避免误判"

rm -f "$PID_FILE"
: > "$LOG_FILE"

if command -v setsid >/dev/null 2>&1; then
  nohup setsid "$PYTHON" -m uvicorn app.main:app \
    --host "$HOST" --port "$PORT" \
    </dev/null >"$LOG_FILE" 2>&1 9>&- &
else
  nohup "$PYTHON" -m uvicorn app.main:app \
    --host "$HOST" --port "$PORT" \
    </dev/null >"$LOG_FILE" 2>&1 9>&- &
fi
new_pid=$!
printf '%s\n' "$new_pid" > "$PID_TEMP"

started=false
for _ in {1..60}; do
  if ! kill -0 "$new_pid" 2>/dev/null; then
    tail -n 80 "$LOG_FILE" >&2 || true
    fail "新后端进程已提前退出"
  fi

  mapfile -t listener_pids < <(port_pids)
  owns_port=false
  foreign_listener=false
  for pid in "${listener_pids[@]}"; do
    if [[ "$pid" == "$new_pid" ]]; then
      owns_port=true
    else
      foreign_listener=true
    fi
  done

  if [[ "$foreign_listener" == true ]]; then
    fail "$PORT 端口被其他 PID ${listener_pids[*]} 抢占"
  fi
  if [[ "$owns_port" == true ]]; then
    health_response="$(curl -fsS --connect-timeout 1 --max-time 2 "$HEALTH_URL" 2>/dev/null || true)"
    if [[ "$health_response" == *'"status":"ok"'* ]]; then
      started=true
      break
    fi
  fi
  sleep 0.5
done

if [[ "$started" != true ]]; then
  tail -n 80 "$LOG_FILE" >&2 || true
  fail "后端未在 30 秒内通过健康检查"
fi

mv -f "$PID_TEMP" "$PID_FILE"
trap - EXIT
new_pid=""

printf '后端启动成功：PID=%s，监听=%s:%s\n' "$(cat "$PID_FILE")" "$HOST" "$PORT"
printf '健康检查：%s\n' "$(curl -fsS --max-time 2 "$HEALTH_URL")"
printf '日志文件：%s\n' "$LOG_FILE"
