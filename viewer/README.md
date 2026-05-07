# 3D Episode Replay Viewer

Cosmetic 3D replay of pursuit-evasion episodes (same 2D sim logic, pretty arena for clips).

## Quick start

```bash
# 1. Export episode from Python logs
cd ..
PYTHONPATH=. python scripts/export_episode_for_viewer.py \
  --log-dir logs_self_steer \
  --episode-id self_steer_scattered_nv2_seed0

# 2. Install & run viewer
cd viewer
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Controls

- **Play / Pause** — timeline playback
- **Scrubber** — jump to any step
- **Speed** — 0.5×–3×
- **Cinematic cam** — follows hero (disable for free orbit with mouse)
- **Episode dropdown** — any `public/episodes/*.json`

## Export from other runs

```bash
PYTHONPATH=. python scripts/export_episode_for_viewer.py --log-dir logs_groq --episode-id <id>
```

## Stack

- Next.js 14 + React Three Fiber + drei
- Reads compact JSON (no prompts, no API keys in browser)

## Twitter clip tip

Record with OBS or browser tab capture at 1.5× speed, cinematic cam on, 15–20s around closest villain approach or capture.
