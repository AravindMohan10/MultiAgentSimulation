# Multi-agent pursuit–evasion: what I built and measured

This matches my [MultiAgentSimulation](https://github.com/) repo — swap in your public URL when it’s up. It’s a 2D sim: a hero and two villains each turn; policies are LLMs (Groq) or rule-based. The engine does maps, visibility, physics, optional messaging. I log everything to JSON/JSONL so I can score runs afterward.

**Where it started:** I’ve spent a lot of time in **Conway’s Game of Life** — same rules every tick, gliders and chaos from a grid. I wanted that **iterated, watchable** feel in a **multi-agent** setup with **real maps**, not a CA. This repo is that (LLM agents, not XOR-of-neighbors).

## What’s in the box

- Procedural maps: scattered, hub-and-spokes, asymmetric labyrinth, gradient.
- Regimes (R1/R2/R3): sight radii, noise, message delay/budgets.
- Prompt versions: baseline, communication-heavy, guided (`src/agents/prompts.py`).
- Pygame for debugging and capture; scripts turn logs into CSV and markdown reports.

## Larger pilot (20 episodes)

I ran one structured sweep: four maps, five seeds each, 150 step cap, V2 guided, R1. **16 escapes**, **4 captures**:

| Map | Escaped | Captured |
|-----|---------|----------|
| scattered | 4 | 1 |
| hub_and_spokes | 3 | 2 |
| asymmetric_labyrinth | 4 | 1 |
| gradient | 5 | 0 |

So about **20% capture rate**; **gradient** was easiest for the hero; **hub** was hardest in that batch. I keep the counts in `results/aggregates/pilot_20ep_snapshot.json` so I can regenerate figures.

## Smaller on-disk runs (`results/pilot_01/`)

Nine episodes (three maps × three seeds) that mostly **hit 80 steps** with escapes. Different horizon/settings than the 20-ep pilot — don’t mix the two in a chart without saying so.

## Screen recordings

Three clips (~35–55s): different maps/prompts, **messaging on** vs **`--disable-messages`**.

1. **Hub + V2 guided, seed 0:** Escape over the run; oscillation escape partway through.
2. **Hub + V1 communication, seed 3:** Capture mid-run; villain 2 started very close.
3. **Scattered + V2 guided, seed 0:** Full-horizon escape.

Pygame doesn’t draw chat text; comms still hit the logs.

**What I see on video (hub-and-spokes):** with **comms on**, both villains **align** and **move on the hero together**; the hero **uses the spokes** to hide. With **`--disable-messages`**, it **splits**: one **pursues**, one **explores**. Same code path, different channel.

**How I built it:** multi-agent orchestration (parallel turns, one world state), JSON actions with validation and fallbacks, timeouts/retries/rate limits, and per-step logs + manifests + metrics so a bad run is inspectable end-to-end — same kind of pressure as shipping agentic systems.

I upload **native video** to LinkedIn for reach; for GitHub I’d use one short GIF or a still + link.

## Figures

```bash
PYTHONPATH=. python scripts/plot_research_figures.py
```

Writes `results/figures/pilot_20ep_outcomes_by_map.png`, `repo_summaries_steps_by_map.png`, `repo_summaries_table.csv`.

## Limits I’m upfront about

- LLM runs **vary** even with the same seed.
- Hero motion mixes model output with **boundary blending** and **oscillation escape** where the engine kicks in.
- Phase metrics like “beacon” / “superadditivity” are **helpers**, not automatic truth — I treat weak signals as “need more data,” not claims.

## What I might do next

- Same map×seed grid across V0/V1/V2.
- More paired runs: `--disable-messages` vs default; pull capture time and message stats from logs.
- More rule-based baselines per map.
- More seeds when budget allows; track $ per episode on Groq.

---

### README GIF

LinkedIn: MP4 is fine. For GitHub, one short GIF or screenshot + link — full clips are too heavy as GIFs.

```bash
ffmpeg -ss 5 -t 12 -i clip1.mov -vf "fps=8,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 assets/demo.gif
```
