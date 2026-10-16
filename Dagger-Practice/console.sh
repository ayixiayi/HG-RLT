#!/usr/bin/env bash
# The web console in the background, so closing SSH does not stop a collection.
#
#   ./hgrlt console start [--host H] [--port P] [--new-token]
#   ./hgrlt console status | url | log | attach | stop [--force]
#
# Runs in a dedicated tmux server (tmux -L hgrlt), else setsid. The console's
# own log is logs/console.log; its terminal output (with the access URL) goes
# to logs/console.out; both are owner-only. `stop` interrupts the console,
# which ends a collection with its own keys and stops the model server it
# started before exiting.
set -euo pipefail
umask 077
H=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
logs="${HGRLT_CONSOLE_LOGS:-$H/logs}"
out="$logs/console.out"
state="$logs/console.state" # host port pid
pidfile="$logs/console.pid"
token_file="$HOME/.config/hgrlt/console-token"
mkdir -p "$logs"
touch "$out" && chmod 600 "$out"
tmux_cmd=(tmux -L hgrlt)
# One tmux session per installation on the user's shared hgrlt tmux server.
session="console-$(printf %s "$H" | sha256sum | cut -c1-8)"
have_tmux() { [[ -z "${HGRLT_CONSOLE_NO_TMUX:-}" ]] && command -v tmux >/dev/null; }

running_pid() {
  [[ -f "$state" ]] || return 1
  local pid
  pid="$(awk '{print $3}' "$state")"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null || return 1
  # The pid may belong to another process since this console exited.
  ps -o args= -p "$pid" 2>/dev/null | grep -q -- "dagger_offline.webui --pidfile $pidfile" || return 1
  echo "$pid"
}

url() {
  local host port token
  read -r host port _ < "$state"
  token="${HGRLT_UI_TOKEN:-}"
  [[ -n "$token" ]] || token="$(tr -d '\n' < "$token_file")"
  echo "http://$host:$port/?token=$token"
  if [[ "$host" == 127.0.0.1 ]]; then
    echo "其他电脑：ssh -N -L $port:127.0.0.1:$port -p 2222 ziyu@<实验室地址>，再打开上面的网址"
  fi
}

case "${1:-}" in
start)
  shift
  exec 9> "$logs/console.lock"
  flock -n 9 || { echo "另一个控制台正在启动或停止" >&2; exit 1; }
  if pid="$(running_pid)"; then
    echo "控制台已在运行（PID $pid）："; url; exit 0
  fi
  host=127.0.0.1 port=8810
  args=("$@")
  for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
      --host) host="${args[i + 1]:-}" ;;
      --host=*) host="${args[i]#--host=}" ;;
      --port) port="${args[i + 1]:-}" ;;
      --port=*) port="${args[i]#--port=}" ;;
    esac
  done
  if ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN; then
    echo "端口 $port 已被占用；用 --port 换一个" >&2; exit 1
  fi
  cmd=("$H/hgrlt" ui --pidfile "$pidfile" "${args[@]}")
  printf -v quoted '%q ' "${cmd[@]}"
  echo "=== $(date '+%F %T') start" >> "$out"
  from=$(( $(wc -l < "$out") + 1 ))
  rm -f "$pidfile"
  if have_tmux; then
    "${tmux_cmd[@]}" kill-session -t "$session" 2>/dev/null || true
    "${tmux_cmd[@]}" new-session -d -s "$session" -c "$H" \
      "exec $quoted >> $(printf %q "$out") 2>&1" 9>&-
  else
    (cd "$H" && exec setsid "${cmd[@]}" >> "$out" 2>&1 < /dev/null 9>&- &)
  fi
  ready=""
  for _ in $(seq 1 $(( ${HGRLT_CONSOLE_START_SECONDS:-20} * 5 ))); do
    if [[ -s "$pidfile" ]] && tail -n "+$from" "$out" | grep -q "HG-RLT 控制台："; then ready=1; break; fi
    sleep 0.2
  done
  if [[ -z "$ready" ]]; then
    echo "控制台没有按时启动（见 $out）：" >&2; tail -n 20 "$out" >&2
    if [[ -s "$pidfile" ]]; then kill -TERM "$(tr -d '\n' < "$pidfile")" 2>/dev/null || true; fi
    if have_tmux; then "${tmux_cmd[@]}" kill-session -t "$session" 2>/dev/null || true; fi
    exit 1
  fi
  pid="$(tr -d '\n' < "$pidfile")"
  echo "$host $port $pid" > "$state"
  echo "控制台在后台运行（PID $pid；日志 $logs/console.log）"
  url
  ;;
status)
  if pid="$(running_pid)"; then echo "运行中（PID $pid）"; url; else echo "未运行"; exit 1; fi
  ;;
url)
  running_pid >/dev/null || { echo "未运行" >&2; exit 1; }
  url
  ;;
log)
  exec tail -n "${2:-50}" -f "$logs/console.log"
  ;;
attach)
  have_tmux || { echo "没有 tmux；用 ./hgrlt console log" >&2; exit 1; }
  exec "${tmux_cmd[@]}" attach -t "$session"
  ;;
stop)
  exec 9> "$logs/console.lock"
  flock -n 9 || { echo "另一个控制台正在启动或停止" >&2; exit 1; }
  pid="$(running_pid)" || { echo "未运行"; rm -f "$state" "$pidfile"; exit 0; }
  # Graceful: the console ends a collection with its own keys (s / q), stops
  # the model server it started, then exits.
  kill -INT "$pid"
  echo -n "正在关闭控制台（PID $pid）"
  wait_s="${HGRLT_CONSOLE_STOP_SECONDS:-200}"
  for i in $(seq 1 $(( wait_s * 5 ))); do
    kill -0 "$pid" 2>/dev/null || break
    (( i % 25 == 0 )) && echo -n "."
    sleep 0.2
  done
  echo
  if kill -0 "$pid" 2>/dev/null; then
    if [[ "${2:-}" != "--force" ]]; then
      echo "$wait_s 秒后仍在关闭（见 $logs/console.log）。继续等待，或：./hgrlt console stop --force" >&2
      exit 1
    fi
    # Last resort: the collection loses its terminal and exits without
    # saving the open episode; the daemon holds once its lease lapses.
    echo "强制结束控制台进程" >&2
    kill -KILL "$pid" 2>/dev/null || true
    sleep 0.5
  fi
  rm -f "$state" "$pidfile"
  if have_tmux; then "${tmux_cmd[@]}" kill-session -t "$session" 2>/dev/null || true; fi
  echo "已关闭"
  ;;
*)
  sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
  ;;
esac
