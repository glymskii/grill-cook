#!/bin/bash
# Sync the shared UI into the cloud bundle and deploy the hub.
set -e
cd "$(dirname "$0")"
cp ../app/static/admin.html ../app/static/hud.html static/
cp ../app/recommend.py recommend.py
railway up --service hub --detach
