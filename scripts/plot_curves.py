"""Draw the training curves for the README as a dependency-free SVG.

    uv run python scripts/plot_curves.py   -> assets/training_curves.svg

Left: pretraining validation loss (each point is a sampled subset of the validation windows; the
last point uses all of them). Right: fine-tuning validation loss on the assistant tokens, for the
run that kept the chat-token rows and the run that reinitialized them.
"""

import json
from pathlib import Path

W, H = 920, 360
PANEL_W, TOP, BOTTOM = 400, 60, 300
FG, MUTED, GRID = "#24292f", "#57606a", "#d0d7de"
RED, BLUE = "#c8374f", "#0969da"


def text(x, y, s, size=13, fill=MUTED, anchor="start", weight=400):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def panel(x0, title, series, x_max, y_lo, y_hi, x_label, y_ticks):
    px = lambda x: x0 + 50 + (PANEL_W - 70) * x / x_max
    py = lambda y: BOTTOM - (BOTTOM - TOP) * (y - y_lo) / (y_hi - y_lo)
    out = [text(x0 + 50, TOP - 24, title, 15, FG, weight=600)]
    for t in y_ticks:
        out.append(f'<line x1="{x0 + 50}" x2="{x0 + PANEL_W - 20}" y1="{py(t):.1f}" y2="{py(t):.1f}" stroke="{GRID}"/>')
        out.append(text(x0 + 42, py(t) + 4, f"{t:g}", 12, anchor="end"))
    for t in (0, x_max // 2, x_max):
        out.append(text(px(t), BOTTOM + 18, f"{t:,}", 12, anchor="middle"))
    out.append(text(x0 + 50 + (PANEL_W - 70) / 2, BOTTOM + 38, x_label, 12, anchor="middle"))
    for i, (label, color, pts) in enumerate(series):
        d = " ".join(f"{'M' if j == 0 else 'L'}{px(s):.1f},{py(v):.1f}" for j, (s, v) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        out += [f'<circle cx="{px(s):.1f}" cy="{py(v):.1f}" r="2.6" fill="{color}"/>' for s, v in pts]
        out.append(f'<rect x="{x0 + PANEL_W - 190}" y="{TOP + 4 + i * 18}" width="12" height="3" fill="{color}"/>')
        out.append(text(x0 + PANEL_W - 172, TOP + 9 + i * 18, label, 12))
    return "".join(out)


def main() -> None:
    pre = json.loads(Path("checkpoints/pretrain.log.json").read_text())
    keep = json.loads(Path("checkpoints/sft_keep.log.json").read_text())
    reinit = json.loads(Path("checkpoints/sft_reinit.log.json").read_text())
    pts = lambda log: [(e["step"], e["val_loss"]) for e in log]
    body = panel(0, "Pretraining · validation loss", [("val loss", RED, pts(pre))], 15000, 3.0, 4.8,
                 "training step", (3.0, 3.5, 4.0, 4.5)) + \
        panel(PANEL_W + 100, "Chat fine-tuning · assistant-token val loss",
              [("rows reinitialized", RED, pts(reinit)), ("rows kept", BLUE, pts(keep))], 1632, 2.5, 3.15,
              "fine-tuning step", (2.6, 2.8, 3.0))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">'
           f'<rect width="{W}" height="{H}" rx="10" fill="#ffffff"/>{body}</svg>')
    Path("assets").mkdir(exist_ok=True)
    Path("assets/training_curves.svg").write_text(svg, encoding="utf-8")
    print("assets/training_curves.svg")


if __name__ == "__main__":
    main()
