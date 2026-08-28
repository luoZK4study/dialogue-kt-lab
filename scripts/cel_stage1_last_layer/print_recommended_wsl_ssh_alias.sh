#!/usr/bin/env bash
set -euo pipefail

SSH_ALIAS="${1:-3090}"
IDENTITY_PATH="${2:-$HOME/.ssh/id_rsa_u4}"

cat <<EOF
== Recommended WSL SSH Alias ==
Add the following block to ~/.ssh/config:

Host $SSH_ALIAS
    HostName 119.29.183.125
    Port 22049
    User user4
    PreferredAuthentications publickey
    IdentityFile $IDENTITY_PATH
    IdentitiesOnly yes

Recommended permissions:
  chmod 700 ~/.ssh
  chmod 600 ~/.ssh/config
  chmod 600 $IDENTITY_PATH

Recommended local launch:
  bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh $SSH_ALIAS v21
EOF
