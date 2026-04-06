# Recording a demo

Screen record the pygame window while a run is in progress. macOS: QuickTime → New Screen Recording, or the screenshot toolbar recorder.

Optional GIF for the repo (keep file small):

```bash
ffmpeg -i demo.mov -vf "fps=10,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 assets/demo.gif
```

Then add `![demo](assets/demo.gif)` to `README.md` if you want it inline.

Pygame does not render chat text; messages are still in `*_steps.jsonl`.
