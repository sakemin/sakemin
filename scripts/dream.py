#!/usr/bin/env python3
"""A randomly initialized transposed-CNN generator, driven by Gaussian noise.

No training, no torch: the weights are drawn once from N(0, He) and never
updated. What you see is the prior of the architecture itself — a random
decoder is already a multi-scale texture generator, which is the same fact
Deep Image Prior leans on.

Noise enters at two points, and they are different things:
  1. the latent z, a 4x4x C0 field that is walked over time;
  2. per-layer noise injection, a fresh field added after every upsample,
     which is what puts fine grain on top of the coarse structure.

The walk is a circle in latent space (z0*cos t + z1*sin t), and the injected
fields are blended on the same circle, so frame N wraps onto frame 0 exactly:
the GIF loops without a seam.
"""
import numpy as np, pathlib, sys
from scipy.signal import convolve2d
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"; ASSETS.mkdir(exist_ok=True)

CHANNELS = [28, 24, 20, 16, 12]     # latent 4x8 -> 8x16 -> 16x32 -> 32x64 -> 64x128 -> 128x256
FRAMES = 48
LATENT_H, LATENT_W = 4, 8            # 1:2, so every feature map and the output stay 1:2
OUT_W, OUT_H = 640, 320
NOISE_GAIN = [0.8, 0.55, 0.4, 0.32]   # coarse scales get more noise; white noise at 64x64 is just static


def he(rng, shape):
    fan_in = shape[1] * shape[2] * shape[3]
    return rng.normal(0, np.sqrt(2.0 / fan_in), shape)


BINOM = np.array([1., 2., 1.]) / 4.


def blur(x, passes=1):
    """Separable binomial lowpass. Transposed convolution with stride 2 leaves a
    checkerboard; this is the standard way to take it back out."""
    for _ in range(passes):
        x = np.array([convolve2d(convolve2d(c, BINOM[None, :], "same", "symm"),
                                 BINOM[:, None], "same", "symm") for c in x])
    return x


def conv_transpose(x, w):
    """stride 2, kernel 4, pad 1 — zero-stuff then correlate, done channel-pair-wise."""
    cin, h, wd = x.shape
    cout = w.shape[1]
    up = np.zeros((cin, h * 2 + 2, wd * 2 + 2))
    up[:, 1:-1:2, 1:-1:2] = x                      # insert stride-1 zeros, keep a 1px margin
    out = np.zeros((cout, h * 2, wd * 2))
    for i in range(cin):
        for o in range(cout):
            full = convolve2d(up[i], w[i, o], mode="same")
            out[o] += full[1:-1, 1:-1]
    return out


def build(seed):
    rng = np.random.default_rng(seed)
    ws = []
    dims = CHANNELS + [3]
    for i in range(len(dims) - 1):
        ws.append(he(rng, (dims[i], dims[i + 1], 4, 4)))
    return ws, rng


NEON = [(0.00, (7, 10, 16)), (0.22, (13, 52, 66)), (0.45, (24, 148, 136)),
        (0.63, (61, 220, 151)), (0.78, (126, 158, 236)), (0.90, (168, 85, 247)), (1.00, (248, 236, 255))]


def lut():
    stops = np.array([s for s, _ in NEON]); cols = np.array([c for _, c in NEON], float)
    x = np.linspace(0, 1, 256)
    return np.stack([np.interp(x, stops, cols[:, k]) for k in range(3)], 1).astype(np.uint8)


def forward(z, ws, noise_a, noise_b, phase):
    x = z
    for li, w in enumerate(ws):
        x = conv_transpose(x, w)
        x = blur(x)                                               # de-checkerboard
        x = x / (x.std() + 1e-6)                                  # keep activations in range without training
        if li < len(ws) - 1:
            n = noise_a[li] * np.cos(phase) + noise_b[li] * np.sin(phase)
            x = np.tanh(x + NOISE_GAIN[li] * n)
    return x


def single_frame(seed: int, phase: float | None = None) -> Image.Image:
    """One frame only — used by still.py, which publishes a picture, not a loop."""
    ws, rng = build(seed)
    c0 = CHANNELS[0]
    z0 = rng.normal(size=(c0, LATENT_H, LATENT_W))
    z1 = rng.normal(size=(c0, LATENT_H, LATENT_W))
    sizes = [(LATENT_H * 2 ** k, LATENT_W * 2 ** k) for k in range(1, 5)]
    na = [blur(rng.normal(size=(CHANNELS[i + 1],) + hw), 2 if i < 2 else 1) for i, hw in enumerate(sizes)]
    nb = [blur(rng.normal(size=(CHANNELS[i + 1],) + hw), 2 if i < 2 else 1) for i, hw in enumerate(sizes)]
    na = [n / (n.std() + 1e-6) for n in na]; nb = [n / (n.std() + 1e-6) for n in nb]
    t = float(rng.uniform(0, 2 * np.pi)) if phase is None else phase
    v = forward(z0 * np.cos(t) + z1 * np.sin(t), ws, na, nb, t).mean(0)
    v = (v - v.mean()) / (v.std() + 1e-6)
    v = np.clip(v * 0.5 + 0.5, 0, 1)
    return Image.fromarray(lut()[(v * 255).astype(np.uint8)]).resize((OUT_W, OUT_H), Image.LANCZOS)


def main():
    # seed comes from the workflow (the run number), so every scheduled run draws
    # a different network and a different pair of latent poles
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ws, rng = build(seed)
    c0 = CHANNELS[0]
    z0 = rng.normal(size=(c0, LATENT_H, LATENT_W))
    z1 = rng.normal(size=(c0, LATENT_H, LATENT_W))
    sizes = [(LATENT_H * 2 ** k, LATENT_W * 2 ** k) for k in range(1, 5)]
    # the injected fields are blurred: correlated noise reads as texture, white noise reads as TV static
    na = [blur(rng.normal(size=(CHANNELS[i + 1],) + hw), 2 if i < 2 else 1) for i, hw in enumerate(sizes)]
    nb = [blur(rng.normal(size=(CHANNELS[i + 1],) + hw), 2 if i < 2 else 1) for i, hw in enumerate(sizes)]
    na = [n / (n.std() + 1e-6) for n in na]; nb = [n / (n.std() + 1e-6) for n in nb]

    table, frames = lut(), []
    for k in range(FRAMES):
        t = 2 * np.pi * k / FRAMES
        z = z0 * np.cos(t) + z1 * np.sin(t)
        out = forward(z, ws, na, nb, t)
        v = out.mean(0)                                   # 3 channels -> one field -> palette
        v = (v - v.mean()) / (v.std() + 1e-6)
        v = np.clip(v * 0.5 + 0.5, 0, 1)
        img = Image.fromarray(table[(v * 255).astype(np.uint8)]).resize((OUT_W, OUT_H), Image.LANCZOS)
        frames.append(img)
        if k % 12 == 0:
            print("frame", k)

    # one fixed palette straight off the LUT + no dithering: smaller file, and no
    # dither speckle crawling across a smooth gradient between frames
    ramp = table[np.linspace(0, 255, 48).astype(int)]
    palimg = Image.new("P", (1, 1))
    palimg.putpalette(list(ramp.reshape(-1)) + [0] * (768 - ramp.size))
    q = [f.quantize(palette=palimg, dither=Image.Dither.NONE) for f in frames]
    q[0].save(ASSETS / "dream.gif", save_all=True, append_images=q[1:],
              duration=85, loop=0, optimize=True)
    frames[0].save(ASSETS / "dream_still.png")
    print("seed", seed, "->", ASSETS / "dream.gif", (ASSETS / "dream.gif").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
