# Grill Cook edge

Одна коробка на кухню: принимает RTSP, гоняет детектор локально, шлёт в облако
только события/снапшоты/клипы. Наружу ничего не открывает — исходящий WSS.

## macOS (MacBook / Mac mini) — боевой способ

```bash
bash edge/install.sh --hub wss://<hub>/ws/agent --token <AGENT_TOKEN> --station main
```

launchd: автозапуск при загрузке, рестарт при падении, caffeinate против сна.
Лог: `app/state/worker.log`. Данные (события, клипы, кэш): `app/state/`.

## Ручной запуск (отладка)

```bash
.venv/bin/python -u app/edge_worker.py --hub wss://<hub>/ws/agent --token <TOKEN>
```

## Jetson / Linux — план

Тот же worker + systemd unit; детектор экспортируется в TensorRT
(`yolo export format=engine`). Понадобится при выходе за пределы пилота.
