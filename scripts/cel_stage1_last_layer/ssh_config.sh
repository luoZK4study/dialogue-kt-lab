#!/usr/bin/env bash

# Shared SSH configuration for strict formal experiment scripts.
#
# Defaults match the repository's current server assumptions. Override these
# from the environment when a local SSH config alias is preferred, e.g.
#   TASK_CONDITIONED_SSH_TARGET=3090
#   TASK_CONDITIONED_SSH_OPTS=
#   TASK_CONDITIONED_SSH_AUTOMATION_OPTS="-o BatchMode=yes -o ConnectTimeout=15"
#   TASK_CONDITIONED_REMOTE_ROOT=/home/user4/dialogue-kt

TASK_CONDITIONED_DEFAULT_SSH_TARGET="user4@119.29.183.125"
TASK_CONDITIONED_DEFAULT_SSH_OPTS="-i $HOME/.ssh/id_rsa_u4 -p 22049"
TASK_CONDITIONED_DEFAULT_SSH_AUTOMATION_OPTS="-o BatchMode=yes -o ConnectTimeout=15"
TASK_CONDITIONED_DEFAULT_REMOTE_ROOT="/home/user4/dialogue-kt"

SSH_TARGET="${TASK_CONDITIONED_SSH_TARGET:-$TASK_CONDITIONED_DEFAULT_SSH_TARGET}"
SSH_OPTS="${TASK_CONDITIONED_SSH_OPTS-$TASK_CONDITIONED_DEFAULT_SSH_OPTS}"
SSH_AUTOMATION_OPTS="${TASK_CONDITIONED_SSH_AUTOMATION_OPTS:-$TASK_CONDITIONED_DEFAULT_SSH_AUTOMATION_OPTS}"
REMOTE_ROOT="${TASK_CONDITIONED_REMOTE_ROOT:-$TASK_CONDITIONED_DEFAULT_REMOTE_ROOT}"

SSH_OPTS_ARR=()
if [[ -n "$SSH_OPTS" ]]; then
  read -r -a SSH_OPTS_ARR <<<"$SSH_OPTS"
fi

SSH_AUTOMATION_OPTS_ARR=()
if [[ -n "$SSH_AUTOMATION_OPTS" ]]; then
  read -r -a SSH_AUTOMATION_OPTS_ARR <<<"$SSH_AUTOMATION_OPTS"
fi

ssh_effective_config() {
  ssh "${SSH_OPTS_ARR[@]}" "${SSH_AUTOMATION_OPTS_ARR[@]}" -G "$SSH_TARGET"
}

ssh_run() {
  ssh "${SSH_OPTS_ARR[@]}" "${SSH_AUTOMATION_OPTS_ARR[@]}" "$@"
}

ssh_target_command() {
  local parts=("ssh")
  if [[ ${#SSH_OPTS_ARR[@]} -gt 0 ]]; then
    parts+=("${SSH_OPTS_ARR[@]}")
  fi
  parts+=("$SSH_TARGET")
  printf '%q ' "${parts[@]}"
}

rsync_ssh_command() {
  local parts=("ssh")
  if [[ ${#SSH_OPTS_ARR[@]} -gt 0 ]]; then
    parts+=("${SSH_OPTS_ARR[@]}")
  fi
  if [[ ${#SSH_AUTOMATION_OPTS_ARR[@]} -gt 0 ]]; then
    parts+=("${SSH_AUTOMATION_OPTS_ARR[@]}")
  fi
  printf '%q ' "${parts[@]}"
}
