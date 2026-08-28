#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/cel_stage1_last_layer/ssh_config.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cel_stage1_last_layer/check_ssh_formal_config.sh

Purpose:
  Validate the local SSH configuration that strict formal experiment scripts
  will use, without opening a network connection.

Environment:
  TASK_CONDITIONED_SSH_TARGET=<target>  Override SSH target, e.g. 3090
  TASK_CONDITIONED_SSH_OPTS=<opts>      Override SSH opts; set empty when using SSH config alias
  TASK_CONDITIONED_SSH_AUTOMATION_OPTS=<opts>  Override automation-only SSH opts
  TASK_CONDITIONED_REMOTE_ROOT=<path>   Override remote repo root

Examples:
  bash scripts/cel_stage1_last_layer/check_ssh_formal_config.sh
  TASK_CONDITIONED_SSH_TARGET=3090 TASK_CONDITIONED_SSH_OPTS= \
    bash scripts/cel_stage1_last_layer/check_ssh_formal_config.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v ssh >/dev/null 2>&1; then
  echo "ssh config check failed: ssh command not found" >&2
  exit 1
fi

effective_cfg="$(ssh_effective_config 2>/dev/null || true)"
if [[ -z "$effective_cfg" ]]; then
  echo "ssh config check failed: unable to resolve effective SSH config for target '$SSH_TARGET'" >&2
  exit 1
fi

effective_user="$(awk '/^user / {print $2; exit}' <<<"$effective_cfg")"
effective_host="$(awk '/^hostname / {print $2; exit}' <<<"$effective_cfg")"
effective_port="$(awk '/^port / {print $2; exit}' <<<"$effective_cfg")"
effective_identityfile="$(awk '/^identityfile / {print $2; exit}' <<<"$effective_cfg")"
effective_preferredauth="$(awk '/^preferredauthentications / {print $2; exit}' <<<"$effective_cfg")"
effective_identitiesonly="$(awk '/^identitiesonly / {print $2; exit}' <<<"$effective_cfg")"
effective_batchmode="$(awk '/^batchmode / {print $2; exit}' <<<"$effective_cfg")"
effective_connecttimeout="$(awk '/^connecttimeout / {print $2; exit}' <<<"$effective_cfg")"

identity_status="unknown"
identity_warning=""
if [[ -n "$effective_identityfile" ]]; then
  if [[ "$effective_identityfile" =~ ^[A-Za-z]:\\ ]]; then
    identity_status="windows_path"
    identity_warning="effective identityfile still resolves to a Windows path; WSL ssh may not be able to use it reliably"
  elif [[ -e "$effective_identityfile" ]]; then
    identity_status="present"
  else
    identity_status="missing"
    identity_warning="effective identityfile does not exist on the local filesystem"
  fi
fi

local_ssh_dir="$HOME/.ssh"
local_ssh_config="$local_ssh_dir/config"
ssh_dir_mode="$(stat -c '%a' "$local_ssh_dir" 2>/dev/null || echo 'missing')"
config_mode="$(stat -c '%a' "$local_ssh_config" 2>/dev/null || echo 'missing')"
identity_mode="missing"
if [[ -n "$effective_identityfile" && -e "$effective_identityfile" ]]; then
  identity_mode="$(stat -c '%a' "$effective_identityfile" 2>/dev/null || echo 'unknown')"
fi

alias_mode="no"
if [[ -z "$SSH_OPTS" ]]; then
  alias_mode="yes"
fi

config_source="explicit_ssh_opts"
if [[ "$alias_mode" == "yes" ]]; then
  config_source="ssh_config_alias"
fi

template_script="$ROOT_DIR/scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh"
strict_fail=0

echo "== Strict Formal SSH Config Check =="
echo "ssh_target=$SSH_TARGET"
echo "ssh_opts=${SSH_OPTS:-}"
echo "ssh_automation_opts=${SSH_AUTOMATION_OPTS:-}"
echo "remote_root=$REMOTE_ROOT"
echo "effective_user=${effective_user:-unknown}"
echo "effective_host=${effective_host:-unknown}"
echo "effective_port=${effective_port:-unknown}"
echo "effective_identityfile=${effective_identityfile:-none}"
echo "effective_preferredauthentications=${effective_preferredauth:-unknown}"
echo "effective_identitiesonly=${effective_identitiesonly:-unknown}"
echo "effective_batchmode=${effective_batchmode:-unknown}"
echo "effective_connecttimeout=${effective_connecttimeout:-unknown}"
echo "identity_status=$identity_status"
echo "alias_mode=$alias_mode"
echo "config_source=$config_source"
echo "ssh_dir_mode=$ssh_dir_mode"
echo "ssh_config_mode=$config_mode"
echo "identityfile_mode=$identity_mode"

if [[ "$REMOTE_ROOT" != /home/user4/dialogue-kt ]]; then
  echo "warning: remote_root differs from the repository default: $REMOTE_ROOT" >&2
fi

if [[ "$ssh_dir_mode" != "700" ]]; then
  echo "warning: $local_ssh_dir permissions are $ssh_dir_mode; 700 is the safer default for SSH directories" >&2
fi

if [[ "$config_mode" != "600" ]]; then
  echo "warning: $local_ssh_config permissions are $config_mode; 600 is the safer default for SSH config" >&2
fi

if [[ "$identity_mode" != "missing" && "$identity_mode" != "600" ]]; then
  echo "warning: identity file permissions are $identity_mode; 600 is the safer default for SSH private keys" >&2
fi

if [[ -n "$identity_warning" ]]; then
  echo "warning: $identity_warning" >&2
fi

if [[ "$effective_batchmode" != "yes" ]]; then
  echo "warning: effective batchmode is '$effective_batchmode'; automated sync/launch commands should run with BatchMode=yes to avoid hanging on password prompts" >&2
  strict_fail=1
fi

if [[ -n "$effective_preferredauth" && "$effective_preferredauth" != *publickey* ]]; then
  echo "warning: preferredauthentications does not include publickey; the formal SSH loop expects public-key auth" >&2
fi

if [[ "$alias_mode" == "yes" && "$effective_identitiesonly" != "yes" ]]; then
  echo "warning: alias-based SSH mode should set IdentitiesOnly yes so the formal loop uses the intended key deterministically" >&2
  echo "hint: see $template_script for the recommended ~/.ssh/config block" >&2
  strict_fail=1
fi

if [[ "$identity_status" == "missing" || "$identity_status" == "windows_path" ]]; then
  exit 1
fi

if [[ "$strict_fail" != "0" ]]; then
  exit 1
fi

echo "ssh config check passed"
