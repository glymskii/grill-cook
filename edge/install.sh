#!/bin/bash
# Grill Cook edge worker — macOS provisioning (MacBook / Mac mini).
# Installs a launchd service that starts the worker at boot, restarts it on
# crash and keeps the machine awake while it runs.
#
# Usage: bash edge/install.sh --hub wss://<hub>/ws/agent --token <AGENT_TOKEN> [--station main]
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
HUB=""; TOKEN=""; STATION="main"
while [ $# -gt 0 ]; do case "$1" in
  --hub) HUB="$2"; shift 2;;
  --token) TOKEN="$2"; shift 2;;
  --station) STATION="$2"; shift 2;;
  *) echo "unknown arg $1"; exit 1;;
esac; done
[ -n "$HUB" ] && [ -n "$TOKEN" ] || { echo "need --hub and --token"; exit 1; }
[ -x .venv/bin/python ] || { echo "run from a repo with .venv prepared"; exit 1; }

PLIST=~/Library/LaunchAgents/com.grillcook.worker.plist
mkdir -p ~/Library/LaunchAgents "$REPO/app/state"
cat > "$PLIST" << XML
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.grillcook.worker</string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/caffeinate</string><string>-is</string>
    <string>$REPO/.venv/bin/python</string><string>-u</string>
    <string>$REPO/app/edge_worker.py</string>
    <string>--hub</string><string>$HUB</string>
    <string>--token</string><string>$TOKEN</string>
    <string>--station</string><string>$STATION</string>
  </array>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$REPO/app/state/worker.log</string>
  <key>StandardErrorPath</key><string>$REPO/app/state/worker.log</string>
</dict></plist>
XML
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "✅ installed: worker runs at boot, restarts on crash, keeps the Mac awake"
echo "   log: $REPO/app/state/worker.log · stop: launchctl unload $PLIST"
