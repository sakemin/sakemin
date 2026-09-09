#!/usr/bin/env python3
"""Read the current header image AS a spectrogram and play it back.

The frame is a palette image, so the color -> magnitude step is exact: every
pixel is matched back to its index in the LUT that produced it. That grid is
then read as |STFT|, one sine per row, and resynthesized with random phase
(the image carries magnitude only — phase is gone, which is why this is a
synthesis, not an inversion).

Frequency axis is logarithmic (A2..A8), so a log/mel-axis spectrogram viewer
shows the picture back undistorted; a linear-axis viewer squashes it upward.
"""
import numpy as np, pathlib, subprocess, sys
from PIL import Image, ImageDraw
from scipy.io import wavfile
from scipy.signal import stft, istft

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS, BUILD = ROOT / "assets", ROOT / "build"; BUILD.mkdir(exist_ok=True)
SR, DUR = 22050, 8.0
F_LO, F_HI = 110.0, 7040.0
ROWS, COLS = 320, 512
sys.path.insert(0, str(ROOT / "scripts"))
from dream import lut                      # the exact palette the frame was written with


def frame_to_magnitude(path: pathlib.Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    table = lut().astype(np.int16)                                  # (256, 3)
    d = ((rgb[:, :, None, :] - table[None, None, :, :]) ** 2).sum(-1)
    mag = d.argmin(-1).astype(np.float64) / 255.0                   # nearest palette entry = original level
    im = Image.fromarray((mag * 255).astype(np.uint8)).resize((COLS, ROWS), Image.LANCZOS)
    g = np.flipud(np.asarray(im, dtype=np.float64) / 255.0)         # image row 0 is the top = highest partial
    return g ** 2.0                                                 # gamma: the field is broad, this sharpens ridges


def synthesize(grid: np.ndarray) -> np.ndarray:
    n = int(SR * DUR); t = np.arange(n) / SR
    freqs = np.geomspace(F_LO, F_HI, grid.shape[0])
    frame_t = np.linspace(0, DUR, grid.shape[1])
    rng = np.random.default_rng(5)
    # equal-loudness-ish tilt: without it the top of the picture dominates,
    # because a log axis packs far more partials per octave up there
    tilt = (freqs / F_LO) ** -0.5
    y = np.zeros(n)
    for i, f in enumerate(freqs):
        row = grid[i]
        if row.max() < 0.02:
            continue
        y += tilt[i] * np.interp(t, frame_t, row) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    y /= np.abs(y).max() + 1e-9
    fade = int(0.05 * SR)
    y[:fade] *= np.linspace(0, 1, fade); y[-fade:] *= np.linspace(1, 0, fade)
    return y * 0.9


def noise_excited(grid: np.ndarray) -> np.ndarray:
    """The other reading of the same picture.

    Instead of one sine per row, put the image straight onto the STFT bin grid
    as a magnitude field, give every bin a random phase, and invert. The result
    is broadband — filtered noise shaped like the picture — so the analysis
    matches the source far more closely than a sine bank can, at the cost of
    sounding like wind rather than like an organ.
    """
    nper, hop = 2048, 256
    n_frames = int(SR * DUR) // hop + 1
    bins = np.fft.rfftfreq(nper, 1 / SR)
    rows = np.geomspace(F_LO, F_HI, grid.shape[0])
    src = np.array(Image.fromarray((grid * 255).astype(np.uint8)).resize((n_frames, grid.shape[0]),
                   Image.LANCZOS), float) / 255.0
    M = np.stack([np.interp(bins, rows, src[:, k], left=0.0, right=0.0) for k in range(n_frames)], axis=1)
    rng = np.random.default_rng(9)
    Z = M * np.exp(2j * np.pi * rng.random(M.shape))
    _, y = istft(Z, fs=SR, nperseg=nper, noverlap=nper - hop, window="hann")
    y = y[: int(SR * DUR)]
    y /= np.abs(y).max() + 1e-9
    fade = int(0.05 * SR)
    y[:fade] *= np.linspace(0, 1, fade); y[-fade:] *= np.linspace(1, 0, fade)
    return y * 0.9


def log_spectrogram(y: np.ndarray) -> np.ndarray:
    f, _, Z = stft(y, fs=SR, nperseg=2048, noverlap=2048 - 256, window="hann")
    mag = np.abs(Z)
    bins = np.geomspace(F_LO, F_HI, ROWS)
    out = np.stack([np.interp(bins, f, mag[:, k]) for k in range(mag.shape[1])], axis=1)
    db = 20 * np.log10(out + 1e-9)
    return np.clip((db - (db.max() - 58)) / 58, 0, 1)


def score(rec, grid):
    a = rec - rec.mean()
    b = np.array(Image.fromarray((grid * 255).astype(np.uint8)).resize((rec.shape[1], rec.shape[0]),
                 Image.BILINEAR), float) / 255.0
    b = b - b.mean()
    return float((a * b).sum()) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    src = ASSETS / "header.png"      # whatever still.py painted last
    grid = frame_to_magnitude(src)
    table = lut()

    for name, y in (("sine", synthesize(grid)), ("noise", noise_excited(grid))):
        wavfile.write(BUILD / ("frame_%s.wav" % name), SR, (y * 32767).astype(np.int16))
        rec = log_spectrogram(y)
        Image.fromarray(np.flipud(table[(rec * 255).astype(np.uint8)])).resize((640, 320), Image.LANCZOS) \
            .save(BUILD / ("reconstructed_%s.png" % name))
        print("%-5s : frame -> sound -> spectrogram correlation %.4f" % (name, score(rec, grid)))

    y = np.concatenate([np.zeros(0)])
    wav = BUILD / "frame_sine.wav"

    # a video so it can simply be played: the frame, with the playhead the ear is at
    base = Image.open(src).convert("RGB")
    vdir = BUILD / "frames"; vdir.mkdir(exist_ok=True)
    FPS = 25; total = int(DUR * FPS)
    for k in range(total):
        im = base.copy(); d = ImageDraw.Draw(im)
        x = int(640 * k / total)
        d.line([(x, 0), (x, 320)], fill=(255, 255, 255), width=2)
        im.save(vdir / ("%04d.png" % k))
    for name in ("sine", "noise"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", str(vdir / "%04d.png"), "-i", str(BUILD / ("frame_%s.wav" % name)),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                        str(BUILD / ("frame_%s.mp4" % name))], check=True)
    print("wrote frame_sine.mp4 / frame_noise.mp4")


if __name__ == "__main__":
    main()
