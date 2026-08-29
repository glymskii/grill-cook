# Пилот: веб-камера у плиты (США) → облако (Railway) → MacBook → HUD

## Топология

```
[веб-камера] --USB--> [ноутбук партнёра] --ffmpeg push RTSP/TCP--> [Railway relay]
                                                                        |
[браузер партнёра] <--https/wss-- [Railway hub] <--outbound ws-- [MacBook: детекция]
```

Партнёру не нужны VPN и установки кроме ffmpeg: публичная точка — Railway.
MacBook подключается к облаку исходящим соединением, наружу ничего не открыто.

## Партнёру (2 шага)

**1. Публикация камеры** (один раз установить ffmpeg: `brew install ffmpeg`,
список камер: `ffmpeg -f avfoundation -list_devices true -i ""`):

macOS:
```bash
ffmpeg -f avfoundation -framerate 30 -video_size 1280x720 -i "0:none" \
  -pix_fmt yuv420p -c:v libx264 -preset veryfast -tune zerolatency \
  -g 30 -b:v 3M -an -f rtsp -rtsp_transport tcp \
  rtsp://altaria.proxy.rlwy.net:43387/cam
```

Windows (имя камеры: `ffmpeg -list_devices true -f dshow -i dummy`):
```bash
ffmpeg -f dshow -framerate 30 -video_size 1280x720 -i video="Integrated Camera" ^
  -pix_fmt yuv420p -c:v libx264 -preset veryfast -tune zerolatency ^
  -g 30 -b:v 3M -an -f rtsp -rtsp_transport tcp ^
  rtsp://altaria.proxy.rlwy.net:43387/cam
```

Камера сверху-вбок на сковороду, штатив зафиксирован, вся жарочная поверхность
в кадре. Команда после запуска «висит» молча — это нормально, поток идёт.

**2. HUD** — открыть в браузере на весь экран и включить 🔊 sound:

```
https://hub-production-3136.up.railway.app/hud?v=7e37b12aae2bf12e
```

Это read-only ссылка: с планшета жарщика админку открыть нельзя — на нём
безопасно оставлять её навсегда.

Кольцо-прогресс на каждой котлете; за 3 с — крупный отсчёт и бипы, `FLIP` —
переворачивать; вовремя → 🔥 тост и стрик; рано → оранжевый (недостающее время
переносится на другую сторону); поздно → красная рамка.

## Админка (у вас)

```
https://hub-production-3136.up.railway.app/admin?k=08cdd123264d2251
```

1. Patty & griddle: единицы inches/°F, «Suggest target times» → apply → Save targets.
2. Shifts: имя/код партнёра, дни недели, часы (таймзона хаба: America/New_York —
   поменять можно переменной TZ в Railway).
3. RTSP source уже прописан: `rtsp://altaria.proxy.rlwy.net:43387/cam`.
4. ▶ Start. Чипы: edge online, stream ok, camera/processing fps.
5. Если модель мажет на домашней сковороде — Model → YOLO-World fallback, Restart.

## MacBook (edge, у вас)

Worker уже запущен. Перезапуск при необходимости:

```bash
.venv/bin/python app/edge_worker.py --hub wss://hub-production-3136.up.railway.app/ws/agent --token <AGENT_TOKEN из Railway vars>
```

- При обрыве связи переподключается сам; после редеплоя хаба восстанавливает
  настройки и смены из локального кэша (`app/state/hub_cache.json`).
- События копятся локально в `app/state/events.jsonl` — сырьё для разбора.

## Заметки

- Ключи раздельные: `?k=…` — админка (только у вас), `?v=…` — планшет жарщика (read-only). Вводятся один раз — дальше кука.
- Латентность US→Railway→KZ не влияет на точность таймеров: дедлайны абсолютные,
  HUD ведёт отсчёт по своим часам с поправкой.
- Безопасность: система контролирует консистентность, не готовность. Фарш —
  160°F / 71°C внутри, проверяется термометром.
