#!/usr/bin/env python3
"""Produces ONE still header per run, rotating through three readings of the frame.

usage: still.py <run_number>

The run number picks both the network seed and which reading gets published:
    run % 3 == 0  ->  the frame the random decoder produced
    run % 3 == 1  ->  that frame resynthesized as 320 sine partials, re-analyzed
    run % 3 == 2  ->  that frame resynthesized as noise-excited ISTFT, re-analyzed

So the header is never animated, but it is never the same picture twice either:
a new network every run, and the reading cycles 0,1,2,0,... in order.

Prints the caption line for the README on stdout.
"""
import pathlib, sys
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"; ASSETS.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))
import dream
from dream import lut

W, H, COLORS = 640, 320, 96

READINGS = [
    ("as generated", "random transposed CNN · no training, no gradient"),
    ("as 320 sine partials", "read as a spectrogram, resynthesized additively, re-analyzed"),
    ("as noise-excited ISTFT", "read as a spectrogram, one random phase per bin, re-analyzed"),
]


def colorize(field: np.ndarray) -> Image.Image:
    return Image.fromarray(np.flipud(lut()[(field * 255).astype(np.uint8)])).resize((W, H), Image.LANCZOS)


def main() -> int:
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = run % 3

    frame = dream.single_frame(seed=run)          # 640x320 PIL image, one point on the latent circle
    if mode == 0:
        out = frame
    else:
        # the audio path costs a few seconds, so it only runs on the runs that need it
        from frame_to_sound import frame_to_magnitude, synthesize, noise_excited, log_spectrogram
        tmp = ASSETS / "_frame.png"; frame.save(tmp)
        grid = frame_to_magnitude(tmp); tmp.unlink()
        y = synthesize(grid) if mode == 1 else noise_excited(grid)
        out = colorize(log_spectrogram(y))

    out.quantize(colors=COLORS, method=Image.MEDIANCUT, dither=Image.Dither.NONE) \
       .save(ASSETS / "header.png", optimize=True)
    key, sub = READINGS[mode]
    print("run %d · %s · %s" % (run, key, sub))
    return 0


if __name__ == "__main__":
    sys.exit(main())
