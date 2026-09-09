<!--HEADER-->
<p align="center">
  <img src="assets/header.png" width="640" height="320" alt="run 0 · as generated">
  <br><sub>run 0 · as generated · random transposed CNN · no training, no gradient</sub>
</p>
<!--/HEADER-->

---

The picture above is repainted every hour, and it is never animated — one still,
replaced. Each run draws a fresh randomly initialized transposed CNN (five
stride-2 layers, `4×8` latent, Gaussian noise injected after every upsample) and
publishes **one of three readings** of the frame it produces, cycling with the
run number:

| `run % 3` | what gets published |
| --- | --- |
| `0` | the frame the network drew |
| `1` | that frame read as a spectrogram, resynthesized as 320 sine partials, then re-analyzed |
| `2` | the same frame resynthesized as noise-excited ISTFT — one random phase per bin — then re-analyzed |

Rows 1 and 2 are the same picture pushed through sound and back. They come out
looking different because the two syntheses fail differently: the sine bank
leaves a lattice where its partials outrun the analysis resolution, while the
noise excitation sits on the STFT grid exactly and smears in time instead.

Nothing here is trained. The weights are drawn from `N(0, 2/fan_in)` once per
run and thrown away — the structure you see is the prior of the architecture,
which is the same fact Deep Image Prior leans on.

```bash
python3 scripts/still.py 7      # paint run 7 into assets/header.png
```
