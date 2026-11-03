"""Regenerate MultiLingua ASR PaperLoom-style logo banner GIF."""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
W, H, N, DUR = 920, 211, 32, 90
PAPER, INK, MUTED, SOFT, MARK = (251, 250, 246), (23, 26, 34), (113, 118, 111), (69, 74, 68), (23, 26, 34)
ACCENT, ACCENT2 = (16, 185, 129), (6, 182, 212)
TITLE, UI = r"C:\Windows\Fonts\georgiai.ttf", r"C:\Windows\Fonts\segoeui.ttf"


def fnt(p: str, s: int):
    return ImageFont.truetype(p, s)


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def underline(frame: int) -> int:
    t = frame / (N - 1)
    if t < 0.45:
        return int(420 * ease(t / 0.45))
    if t < 0.75:
        return 420
    return int(420 * (1 - ease((t - 0.75) / 0.25)))


def pulse(frame: int) -> float:
    return 0.5 + 0.5 * math.sin(2 * math.pi * (frame * DUR / 1000) / 2 + 0.4)


def frame(i: int) -> Image.Image:
    img = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(img)
    for y in range(H):
        t = y / (H - 1)
        d.line([(0, y), (W, y)], fill=(int(251 - 9 * t), int(250 - 11 * t), int(246 - 15 * t)))
    ox, oy = 72, 36
    d.rounded_rectangle([ox + 8, oy + 8, ox + 140, oy + 140], radius=20, fill=MARK)
    ix, iy = ox + 8, oy + 8
    for x in (28, 48, 68, 88, 108):
        d.line([(ix + x, iy + 20), (ix + x, iy + 112)], fill=(242, 239, 231), width=2)
    for y, color, delay in ((40, ACCENT, 0.0), (62, ACCENT2, 0.18), (84, (242, 239, 231), 0.36)):
        cycle = (i / N + delay) % 1.0
        local = ease(cycle / 0.5) if cycle <= 0.5 else 1 - ease((cycle - 0.5) / 0.5)
        x1 = ix + 16 + int(100 * local)
        if x1 > ix + 18:
            d.line([(ix + 16, iy + y), (x1, iy + y)], fill=color, width=4)
    d.text((ix + 66, iy + 78), "M", fill=(251, 250, 246), font=fnt(TITLE, 52), anchor="mm")
    r = 7 + 2 * pulse(i)
    node = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(node).ellipse([ix + 108 - r, iy + 20 - r, ix + 108 + r, iy + 20 + r], fill=(*ACCENT, int(140 + 115 * pulse(i))))
    img = Image.alpha_composite(img.convert("RGBA"), node).convert("RGB")
    d = ImageDraw.Draw(img)
    d.text((260, 62), "MultiLingua ASR", fill=INK, font=fnt(TITLE, 46))
    d.text((262, 118), "CODE-SWITCH SPEECH INTO ONE TRANSCRIPT", fill=MUTED, font=fnt(UI, 13))
    uw = underline(i)
    if uw:
        d.rounded_rectangle([260, 148, 260 + uw, 151], radius=1, fill=ACCENT)
    d.text((260, 168), "RLVR ASR across eight language pairs.", fill=SOFT, font=fnt(UI, 15))
    return img


def main() -> None:
    frames = [frame(i) for i in range(N)]
    q = [f.convert("P", palette=Image.Palette.ADAPTIVE, colors=128) for f in frames]
    out = ROOT / "docs" / "images" / "logo-banner.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(out, save_all=True, append_images=q[1:], duration=DUR, loop=0, optimize=True, disposal=2)
    q[0].save(ROOT / "MultiLingua_ASR_Logo.gif", save_all=True, append_images=q[1:], duration=DUR, loop=0, optimize=True, disposal=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
