# LinkedIn (draft — paste and edit)

I’ve been building a 2D **multi-agent pursuit–evasion** sim in Python (LLM villains vs a hero on procedural maps, Groq, pygame). The original itch was **Conway’s Game of Life** — discrete time, simple rules, patterns worth watching — but I wanted **multiple agents** and **real geometry**, not a grid CA.

I recorded runs on the **same hub-and-spokes style map** twice: **messaging on** vs **`--disable-messages`**.

**Comms on:** the two villains **line up** and **push toward the hero together**; the hero **dives into the spokes** to break line-of-sight / hide. Coordinated pursuit is visible on screen.

**Comms off:** behavior **splits** — **one villain keeps chasing**, the **other drifts into exploration** instead of staying on the hero. Same stack, different channel.

**Broader sweep:** 20 episodes (4 maps × 5 seeds) landed around **~20% capture rate**; hub was the toughest map in that batch. Chart’s in the repo.

**How it’s built:** **multi-agent orchestration** (parallel turns, single world state), **contracted LLM I/O** (JSON actions, validation, fallbacks), **operational hardening** (timeouts, retries, rate limits), **full-run observability** (per-step JSONL, manifests, metrics) so failures are debuggable like a trace. Same design pressures as **shipping agent systems**, not a one-off notebook demo.

Repo: *[add your GitHub URL]*

Upload the **video file** on LinkedIn (native). Hook viewers in the first few seconds (on-screen labels help).
