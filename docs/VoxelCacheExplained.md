# The Voxel Cache, From Scratch

This document explains how the 3D voxel-cache localization mechanism in
`nerfstudio/dc/mask_voxel_cache.py` works. It describes the cache **as it runs
today** — every mechanism below is live in the code. Each concept opens with the
problem it solves, gives the equations, then walks through a small numeric
example. The intended reader knows the basics of NeRF and DDS (Delta Denoising
Score) but has never seen the cache before.

(Several mechanisms that were tried and rejected during development — a
subtractive fusion branch, active contested-region suppression, a peak-held
variance latch — were removed from the code and are no longer documented here.
This file describes only what the current configuration does.)

---

## Table of contents

1. [The starting point: why per-view masks aren't enough](#1-the-starting-point)
2. [The basic voxel grid](#2-the-basic-voxel-grid)
3. [Depth backprojection](#3-depth-backprojection)
4. [Reading the grid back: observed-weighted trilinear](#4-reading-the-grid-back)
5. [EMA aggregation across views](#5-ema-aggregation-across-views)
6. [Fusion with the 2D mask (positive-only)](#6-fusion-with-the-2d-mask)
7. [Scale-matching the cache value before fusion](#7-scale-matching)
8. [Cross-view variance per voxel](#8-cross-view-variance)
9. [The confidence gate: count + variance](#9-the-confidence-gate)
10. [Raw-self input source](#10-raw-self-input-source)
11. [Angular-diversity factor](#11-angular-diversity-factor)
12. [Scene-relative normalization](#12-scene-relative-normalization)
13. [Auto-freeze at the peak](#13-auto-freeze-at-the-peak)
14. [Mass gate](#14-mass-gate)
15. [Summary: final fusion equation + knob→symbol map](#15-summary)
16. [What's load-bearing in the current config](#16-whats-load-bearing)

---

## 1. The starting point

### Problem

DDS trains a 3D scene by sampling a single rendered view per iteration, running
it through a 2D diffusion model, and pushing the gradient back into the NeRF.
The diffusion model produces a noise prediction `ε_tgt` for the target edit and
`ε_src` for the source image. Their difference `ε_tgt − ε_src` localizes the
edit: large where the model wants to change pixels, small where it doesn't.

This per-view edit signal has two problems:

1. **No cross-view consistency.** Two cameras observing the same 3D point can
   get very different mask values, because the model's predictions depend on the
   rendered framing, lighting, and 2D context. The same physical helmet pixel
   might be "edit strongly" in view A and "leave alone" in view B.
2. **Single-iteration noise.** The mask from one view is a sample of a
   stochastic process. It varies across iterations even from the same view.

The goal of the voxel cache is to **lift the per-view mask into 3D and average
evidence across views**, producing a mask that is consistent across viewpoints
because it lives in the 3D scene, not in the 2D image plane.

### Why a voxel grid

The cache uses a **non-parametric voxel grid** rather than something learned
(an MLP over 3D points, per-Gaussian attributes) because:

- No additional learning is required — the cache just stores running averages.
- It's fast to update and query (just indexing).
- It can be inspected directly (save the grid to a 3D file and look at it).

The trade-off is grid discretization error, controlled by the resolution.

---

## 2. The basic voxel grid

### What it is

A 3D grid of cells. Each cell stores a single scalar in `[0, 1]`: the
running-average mask value for the 3D region that cell occupies.

Let the scene be bounded by an axis-aligned box
`[x_min, x_max] × [y_min, y_max] × [z_min, z_max]`. We divide each axis into `V`
cells of equal width, producing `V³` voxels. A point `p = (x, y, z)` maps to a
voxel index via

```
idx(p) = floor( (p − bbox_min) / (bbox_max − bbox_min) · V )
```

clamped to `[0, V−1]` on each axis.

### Numerical example

Suppose `bbox_min = (−1, −1, −1)`, `bbox_max = (1, 1, 1)`, and `V = 4` (each axis
split into 4 cells of width `0.5`).

A point `p = (0.3, −0.7, 0.0)`:

- Normalized: `(0.65, 0.15, 0.5)`
- Times `V=4`: `(2.6, 0.6, 2.0)`
- Floored: `(2, 0, 2)`

So this point goes into voxel `(2, 0, 2)`. A point at `(2.0, 0.0, 0.0)` falls
outside the bbox (normalized x = 1.5); it's marked "out of bounds" and skipped.

### What's stored

- `self.grid`: `[V, V, V]` fp32, the running-average mask value. Initialized to
  `fallback_value = 0.5` ("uncertain").
- `self.observed`: `[V, V, V]` bool, whether each voxel has ever been observed.

In production `V = 64`, so the grid has `64³ ≈ 260k` voxels totaling ~1 MB at
fp32. Coarse but cheap.

---

## 3. Depth backprojection

### Problem

Per-view diffusion masks live in image space: one value per pixel. To put them
in the 3D grid, we need to know, for each pixel, **which 3D point it observes**.

### Solution

NeRF gives us this for free. During rendering, each pixel has a ray `(o, d)`
(origin + unit direction) and a learned depth `t` along the ray (where the
surface lies). The 3D point is

```
p = o + t · d
```

We take `t` from the NeRF's expected-depth output (the integral
`∫ T(t) · σ(t) · t dt` over the ray). No separate depth model is needed.

### Numerical example

Pixel `(120, 200)` has ray origin `o = (0.0, 1.5, 2.0)`, direction
`d = (0.1, −0.3, −0.95)`, depth `t = 2.1`. Then

```
p = (0.0, 1.5, 2.0) + 2.1 · (0.1, −0.3, −0.95) = (0.21, 0.87, 0.005)
```

That pixel's mask value is written into the voxel containing `(0.21, 0.87, 0.005)`.

### Practical detail: the bbox source

Backprojection needs a bbox that **contains the actual surface points** the
cache will index. Naive choices fail: a camera-position AABB misses the subject
(which sits *outside* the camera cluster along the viewing direction), and the
dataparser's scene box is only correct if it's in the same frame as the rays.

The fix is an **observed bbox**. For the first `bbox_observe_steps` (e.g. 50)
iterations the cache is dormant: it accumulates backprojected points but doesn't
build the grid. Then it takes the AABB of those points (with a quantile clip to
drop far-depth outliers), inflates by a small percentage, and uses that as the
final bbox. By construction it contains exactly the points the cache indexes.

---

## 4. Reading the grid back

Once a pixel is backprojected to `p = o + t·d`, we read a mask value out of the
grid at `p`. The read is **observed-weighted trilinear interpolation**.

### Why not nearest-voxel

The simplest read is nearest-voxel: drop `p` into the one voxel that contains it
and return its stored mean, `M_3D(p) = grid[idx(p)]`. At `V = 64` the voxels are
coarse. A small feature (an eye, an armor edge) can sit near a voxel boundary,
so two views whose rays land a hair apart read *two different voxels* — even
though they see the same surface. That produces a **blocky, view-dependent**
mask: stair-steps that look like cross-view disagreement but are pure
quantization. Raising the resolution shrinks the steps but splits each voxel's
observations across 8× more cells, so every voxel is seen by fewer views and the
variance / angular statistics get noisier. You'd trade blur for weaker trust
signals.

### Observed-weighted trilinear

Read all 8 voxel centers surrounding `p`, weight each by the standard trilinear
weight `w_c` (how close `p` is to that corner) *and* by whether that corner has
been observed, then normalize:

```
M_3D(p) = ( Σ_c w_c · observed_c · grid_c ) / ( Σ_c w_c · observed_c )
```

with `c` over the 8 corners. If no surrounding voxel is observed, return the
`fallback_value` (0.5). The `observed_c` weighting is the important part: a plain
trilinear read would blend the 0.5 fallback of unobserved neighbours into the
result and pull edge values toward "uncertain"; weighting by observation keeps
unobserved corners from bleeding in.

This smooths the readback **without** changing grid resolution, so observation
density — and the variance/angular statistics — is untouched. **Only the
returned value is interpolated**; every trust signal (`unique_view_count`,
variance, angular factor, `observed`/valid, mass) is still read at the nearest
voxel, so the gates behave identically. Trilinear changes *what magnitude* the
cache feeds back, not *whether* it's trusted.

### Numerical example

Point `p` sits 30% of the way from voxel A (value 0.8, observed) toward neighbour
B along x; the other two axes land on centers. So `w_A = 0.7`, `w_B = 0.3`.

- **B observed, value 0.6:** `M_3D = (0.7·0.8 + 0.3·0.6)/(0.7+0.3) = 0.74`.
  A smooth blend instead of the hard 0.8 a nearest read would give.
- **B unobserved:** `M_3D = (0.7·0.8)/(0.7) = 0.8`. B contributes nothing — no
  fallback bleed; the value stays at A's 0.8.

Empirically (elf), trilinear is what turned the cache's mask-level consistency
into *rendered* multi-view consistency: `MultiView_pairwise_cos_sim` rose from
0.928 (cache-off) to 0.933, where the nearest-voxel variant left it at ~0.924,
at a small editability cost (`CLIP_direction` within noise). That's why it's the
adopted read.

---

## 5. EMA aggregation across views

### Problem

We can't just *overwrite* a voxel's value each time a new view observes it — that
throws away prior evidence. We need to **combine** evidence from many views.

### Solution

An exponential moving average (EMA). When voxel `q` is observed with value
`m_new`, update its stored value:

```
m(q) ← β · m(q) + (1 − β) · m_new
```

where `β ∈ [0, 1)` is the decay factor. Higher `β` means slower change (more
memory of past observations).

### How β is picked

Camera-count-aware:

```
β = 1 − 1 / (c · N_cameras)
```

with `c ≈ 2` (the knob `mask_voxel_cache_ema_beta_camera_factor`). For
`N_cameras = 65`, `β ≈ 0.9923`; for `N_cameras = 365`, `β ≈ 0.9986`.

**Why scale with N_cameras?** A voxel observed by every camera should sit near
the *average* across cameras, so each observation should contribute about
`1/N_cameras`. Setting `1−β = 1/(c·N_cameras)` makes each observation contribute
roughly that amount — the right magnitude for a sample-mean estimator.

### Within-batch averaging

A single view contributes many pixels to the same voxel (voxels are larger than
pixels). We don't apply the EMA once per pixel — that would let dense regions
overwhelm sparse ones. Instead we **average all pixels of this view that hit
voxel `q`** into one per-view sample, then apply a single EMA step.

### Numerical example

Voxel `q` starts at `m(q) = 0.5`, never observed. View 1's twelve pixels hitting
`q` average to `m_new = 0.72`. The first observation takes the value directly (no
EMA against the fallback):

```
m(q) ← 0.72,   observed(q) ← True
```

View 2 (five pixels, avg 0.60), `β = 0.99`:

```
m(q) ← 0.99 · 0.72 + 0.01 · 0.60 = 0.7188
```

View 3 (`m_new = 0.65`):

```
m(q) ← 0.99 · 0.7188 + 0.01 · 0.65 = 0.7181
```

The running average drifts slowly toward the cross-view mean.

---

## 6. Fusion with the 2D mask

### Problem

Per pixel we now have two mask sources:

- `M_2D(pixel)`: the diffusion model's per-view mask for this iteration.
- `M_3D(pixel)`: the value queried from the cache by backprojecting that pixel.

How do we combine them into one mask for the DDS gradient?

### Solution: positive-only fusion

The cache **adds** agreed-upon edit force where the 3D consensus exceeds the 2D
mask, and never subtracts. Define `ΔM = M_3D − M_2D` and update

```
M_final = M_2D + α(t) · C̃(q) · G · max(ΔM, 0)
```

Where:

- `M_3D` is the observed-weighted trilinear read (§4), scale-matched to `[0,1]`
  (§7).
- `α(t) ∈ [0, 1]` is the warmup blend: it ramps from 0 to its ceiling
  **`mask_voxel_cache_max_blend`** across **`warmup_start` → `warmup_end`**. It
  lets the cache phase in gradually as evidence accumulates.
- `C̃(q)` is the per-voxel confidence (§9, §11, §14).
- `G` is the semantic gate, fixed to `max(M_attn, M_self)` — the cross-attention
  mask and the self-mask, whichever is stronger. This is what keeps the cache
  from pushing edits into regions the 2D masks don't recognize.

**Why positive-only?** During development a *subtractive* branch (letting
`ΔM < 0` pull the mask down to clean up per-view false positives) was tried and
removed. Subtracting edit signal turned out to be more visually destructive than
adding it: the cache's averaged-across-views value is moderate while a per-view
2D mask peaks much higher on fine detail (stormtrooper leg armor, helmet
outline), so subtraction eroded exactly the high-frequency edits we wanted to
keep. Clamping to `max(ΔM, 0)` keeps only the safe direction.

### Numerical example

A pixel with `M_2D = 0.4`, `M_3D = 0.7`, so `ΔM = 0.3`. Suppose the blend and
confidence fold to `α·C̃ = 0.4` and `G = 0.8`:

```
M_final = 0.4 + 0.4 · 0.8 · max(0.3, 0) = 0.4 + 0.096 = 0.496
```

The cache nudged the 2D mask up from 0.4 to ~0.5, moderated by the semantic gate.

A background pixel with `M_2D = 0.6`, `M_3D = 0.2`, so `ΔM = −0.4`:

```
M_final = 0.6 + 0.4 · 0.8 · max(−0.4, 0) = 0.6 + 0 = 0.6
```

`ΔM < 0` is clamped away — the cache adds nothing and never subtracts. The 2D
mask stands as-is.

---

## 7. Scale-matching

> **Knob: `mask_voxel_cache_scale_normalize_quantile`** (`q`, default `0.95`).
> Scale-matching itself is always on.

### Problem

The fusion in §6 differences two quantities of *different nature*:

- `M_2D` is a **sharp** per-view mask — a near-indicator peaking close to 1 on
  the edit region.
- `M_3D` is a **multi-view mean** — averaging across views compresses it toward
  mid-range, and under-observed voxels carry the 0.5 fallback.

So even when the cache and the 2D mask *agree on where the edit is*, `M_3D` sits
systematically **below** `M_2D`'s peaks over the whole edit region. That makes
`ΔM < 0` over the edit itself — not from real disagreement, purely because the
two masks live on different value scales. Under positive-only fusion this means
the cache reads `max(ΔM, 0) = 0` right where it should be *supporting* the edit,
so it contributes nothing useful.

### Solution

Before fusion, **contrast-stretch the queried cache mask to `[0,1]`** over its
observed voxels, so its active range matches the sharp 2D mask. Take the
`[1−q, q]` percentiles of `M_3D` over observed pixels and rescale:

```
lo = quantile(M_3D[observed], 1 − q)
hi = quantile(M_3D[observed], q)
M_3D' = clamp( (M_3D − lo) / (hi − lo), 0, 1 )
```

`mask_voxel_cache_scale_normalize_quantile` is `q` (default `0.95`): the upper
percentile mapped to 1 (and `1−q = 0.05` mapped to 0). Larger `q` → gentler
stretch (only the extreme tail saturates); smaller `q` (e.g. 0.85) → harder
stretch, more pixels pinned to 0/1. Clamped to `[0.5, 0.999]`.

The selection uses the *observed* mask (not confidence), so the low end captures
background voxels; invalid/fallback pixels carry zero confidence downstream, so
their post-stretch value never reaches the gradient.

### Numerical example

Over the edit region the cache reads `M_3D ∈ [0.30, 0.55]` (compressed), while
`M_2D` peaks near 0.9. With `q = 0.95`, suppose `lo = 0.32`, `hi = 0.53`.

- Foreground voxel `M_3D = 0.52` → `(0.52 − 0.32)/(0.53 − 0.32) = 0.95`. Now it
  sits *above* a typical `M_2D`, so `ΔM > 0` and the cache can support the edit.
- Background voxel `M_3D = 0.33` → `≈ 0.05`. Stays low, so on real background
  (where `M_2D` is also low) `ΔM ≈ 0` and the cache stays quiet.

Scale-matching reads percentiles over the queried values, so it runs on what the
trilinear read (§4) produced. The two compose: trilinear smooths the value,
scale-matching re-ranges it.

---

## 8. Cross-view variance

### Problem

The EMA gives us the *mean* mask value per voxel, but not how *consistent* the
views were. A voxel could read `m(q) = 0.5` because all views agreed on 0.5
(trustworthy) or because half said 0.0 and half said 1.0 (contested). We need a
measure of cross-view agreement: **variance**.

### Solution: Welford's online variance

For each voxel, maintain a running mean and a running sum of squared deviations
(`M2`). When a new per-view sample `x` arrives at a voxel already observed by
`n − 1` distinct views:

```
n ← n + 1
δ  ← x − μ_{n−1}
μ_n ← μ_{n−1} + δ / n
δ′  ← x − μ_n
M2  ← M2 + δ · δ′
σ² ← M2 / (n − 1)        # sample variance
```

Numerically stable (Welford 1962) and O(1) per sample. It needs
`unique_view_count[q]` (n), `running_mean[q]` (μ), and `running_m2[q]` (M2).

### Critical detail: unique views

If the same camera observes the same voxel at iteration 100 and again at
iteration 2700, we must **not** count that as two samples — it's the same view
agreeing with itself by construction, which would deflate the variance. We track
a `[n_voxels, num_views]` boolean tensor `view_observed[q, v]` recording which
views have ever observed each voxel; the variance update only fires when a *new*
view first observes a voxel.

### Numerical example

Voxel `q`, three consistent views:

- View 1 (0.8): `n=1, μ=0.8, M2=0` (needs n≥2 for variance).
- View 2 (0.7): `δ=−0.1, n=2, μ=0.75, δ′=−0.05, M2=0.005, σ²=0.005`.
- View 3 (0.9): `δ=0.15, n=3, μ=0.80, δ′=0.10, M2=0.020, σ²=0.010`.

Mean 0.80, variance 0.010 (std ≈ 0.10 between views). A contested voxel (view 1
= 0.2, view 2 = 0.9) gives `μ = 0.55, M2 = 0.245, σ² = 0.245` — same observation
count, far higher variance.

### Decayed (exponentially-weighted) variance

> **Knob: `mask_voxel_cache_variance_decay`** = the EW weight `α` below (`0.0` =
> cumulative Welford; the adopted config uses `0.2`).

Cumulative Welford weights every distinct view equally over the *entire* run. But
the edit is **non-stationary**: the scene changes as training proceeds. A voxel
first observed by camera A at edit-step 500 (immature edit) and by camera B at
edit-step 30000 (mature edit) records the A↔B difference as cross-view
"variance" — when it's really cross-*time* drift. The symptom is
`mean_observed_variance` climbing monotonically over a run (observed on elf:
0.0175 → 0.033 across 33k steps), slowly walking the average voxel up to the
`max_variance` ceiling and choking the gate late in training.

The fix is an exponentially-weighted variance (Finch 2009). With new-view weight
`α`, on each new view's sample `x`:

```
δ  ← x − μ
μ  ← μ + α · δ
σ² ← (1 − α) · (σ² + δ · α · δ)
```

Because views first-observe a voxel in training-time order, `α` down-weights the
stale early-edit samples, so `σ²` tracks the *current* cross-view disagreement
(effective memory ≈ `1/α` views; `α = 0.2` ≈ last 5). Caveat: the EW estimator
is biased low by ≈ `(1 − α)`, so re-read `query_variance_mean` and re-tune
`max_variance` after changing it. Note the per-(view, voxel) statistics update
only on each view's **first** observation of a voxel, so for static geometry the
estimate is effectively frozen after roughly one pass over the cameras — the EW
weighting reweights among first-observation samples in arrival order.

---

## 9. The confidence gate

### Problem

Not every voxel deserves equal trust. We want to ignore voxels observed by too
few unique views (insufficient evidence) and down-weight voxels with high
cross-view variance (contested evidence). This trust signal should multiply the
*fusion strength*, not the cache's mean value itself — we still *learn* every
observation, we just don't necessarily *use* every voxel.

### Solution

A per-voxel confidence `C(q) ∈ [0, 1]`. The count and variance factors are:

```
C(q) = 1[n(q) ≥ n_min] · max(0, 1 − σ²(q) / σ²_max)
```

Where:

- `n(q)` is the unique-view count.
- `n_min` is camera-count-aware: `n_min = clamp(ceil(N_cameras · frac), floor, cap)`
  with `frac = observation_fraction` (0.10), `floor = 5`, `cap = 12`.
- `σ²(q)` is the running variance (Welford or EW, §8).
- `σ²_max` = `mask_voxel_cache_max_variance`, above which a voxel is fully
  untrusted (scene-tuned; the examples below use 0.04).

`C(q)` multiplies into the fusion (§6). Two more multiplicative factors — angular
diversity (§11) and mass (§14) — extend this to the full `C̃(q)` shown in §15.

### Numerical example

- Voxel `q1`: `n = 15`, `σ² = 0.01`, `n_min = 12`, `σ²_max = 0.04` →
  `C = 1 · max(0, 1 − 0.01/0.04) = 0.75`.
- Voxel `q2`: `n = 8` → `C = 0` (insufficient views).
- Voxel `q3`: `n = 20`, `σ² = 0.05 > σ²_max` →
  `C = 1 · max(0, 1 − 0.05/0.04) = 0`.

The first passes both gates at 75% strength; the second is silenced for low view
count; the third for high variance.

---

## 10. Raw-self input source

> **Knob: `gradient_mask_raw_norm_quantile`** (lives in `DCConfig`, not the cache
> config; `1.0` = per-frame max, the adopted config uses `0.95` = divide by p95).

### Problem

What gets fed into the cache as "the 2D mask value"? The DDS gradient's own
internal mask goes through several preprocessing steps, including a
**percentile normalization** that maps the *rank* of each pixel's value, not its
absolute magnitude. That's the right choice for the gradient (invariant to
noise-level scale) but wrong for the cache: a high-contrast scene and a
low-contrast scene end up producing visually similar masks after percentile
normalization, so the cache can't tell foreground from background by value.

Symptom: the histogram of per-voxel means was unimodal near zero with a thin
tail — no separable trough between foreground and background.

### Solution: raw_self with a robust per-frame scale

Bypass percentile normalization for the cache input. Use the raw edit magnitude,
scaled per frame:

```
raw_self = ||ε_tgt − ε_src|| / scale_per_frame
```

Per-frame scaling preserves within-frame absolute structure (foreground stays
high, background stays low) while keeping values in `[0, 1]`. After this the
histogram becomes bimodal and foreground/background separate cleanly.

**Why a quantile, not the max.** Dividing by the frame's single brightest pixel
is fragile: one outlier pixel (a specular highlight, a denoising artifact) at
`||ε_tgt − ε_src|| = 1.4` in one frame versus a hottest pixel of `0.9` in another
makes the *same* 3D point with true relevance `0.5` read `0.5/1.4 ≈ 0.36` in the
first frame and `0.5/0.9 ≈ 0.56` in the second. That `Δ ≈ 0.20` is **spurious
cross-view variance manufactured by the divisor** — contributing variance on the
order of `0.003–0.01`, a large fraction of a `max_variance ≈ 0.02` gate budget.
Dividing by a **robust quantile** (e.g. p95) instead is still a monotone linear
rescale — pixels above the quantile saturate to 1.0 — but a single hot pixel can
no longer rescale the whole frame. This is the cheapest reduction of spurious
cross-view variance, and it must be applied *before* tightening the variance
gate, or the gate is tuned against the normalizer rather than the edit.

### Numerical example

A frame's raw `||ε_tgt − ε_src||` ranges from 0.05 to 1.4.

- **Percentile normalization** maps the distribution to ≈ uniform in `[0, 1]`,
  so background (bottom 80% of pixels) lands anywhere in `0.0–0.8` and picks up a
  lot of mass near 0.5–0.7. Foreground and background aren't separable by value.
- **p95 scaling** divides by the 95th-percentile magnitude, so background stays
  low and foreground stays high — a bimodal distribution the cache can threshold.

---

## 11. Angular-diversity factor

### Problem

Even with the count and variance gate, a voxel can be "confidently wrong" if all
its cameras cluster in viewpoint. Cross-view variance is only meaningful when the
views are *geometrically distinct*: if 10 cameras all look at the helmet from
nearly the same direction, they agree by construction, but that agreement
doesn't certify the value.

Diagnosed on elf (65 cameras, clustered viewpoints): variance ran at 0.005 and
`confidence_coverage_0.5 = 0.99` — the gate was effectively off, yet the cache
hurt the edit. The variance signal couldn't detect that the views weren't
actually triangulating.

### Solution

A second factor measuring *angular* diversity, via circular statistics (Fisher
1953). For each voxel, maintain a running sum of unit ray directions:

```
S(q) = Σ_v d̂_v(q)
```

where `d̂_v(q)` is the unit direction from camera `v` to voxel `q`. The
**resultant length** is

```
R(q) = ||S(q)|| / n(q)  ∈ [0, 1]
```

- `R = 1`: all rays parallel (no triangulation evidence).
- `R = 0`: rays spread uniformly on the sphere (maximal triangulation).

The diversity factor `A(q) = 1 − R(q)` multiplies into the confidence. The
denominator `n(q)` is the evidence-gated unique-view count (not a pure geometric
count): coupling "is this voxel being edited" into the factor is deliberate — a
purely geometric count let voxels with no edit activity receive full cache
contribution and produced artifacts (stormtrooper hand, crotch). The mass gate
(§14) is the explicit version of that coupling.

### Numerical example

Voxel `q1`, 4 cameras from nearly the same direction:

```
d̂_1 = (0,0,−1)   d̂_2 = (0,0,−1)
d̂_3 = (0.05, 0.01, −0.998)   d̂_4 = (0, 0.02, −0.9998)
```

`S ≈ (0.05, 0.03, −3.996)`, `||S|| ≈ 3.996`, `R = 3.996/4 = 0.999`,
`A = 0.001`. A tight cone of viewpoints — the factor crushes its confidence.

Voxel `q2`, 4 cameras from four sides:
`d̂ = (1,0,0), (−1,0,0), (0,1,0), (0,−1,0)`. `S = 0`, `R = 0`, `A = 1.0`.
Maximal diversity — fully trusted.

---

## 12. Scene-relative normalization

### Problem

The absolute `A(q) = 1 − R(q)` is bounded by the capture geometry, not by a
scene-comparable `[0, 1]`. On clown (365 cameras orbiting horizontally at waist
level) every voxel's rays have a dominant y-component, so `R ≈ 0.95` and
`A ≈ 0.05` even for well-triangulated voxels. On a forward-facing capture
`R ≈ 0.99`, `A ≈ 0.01`. Applying `A` directly would collapse confidence on
*every* voxel of such scenes — the factor becomes a no-op damping multiplier.

### Solution: divide by the trusted-population mean

Normalize by the scene-wide mean angular factor:

```
Ã(q) = min(1, A(q) / Ā)
```

`Ã = 1` at the scene's typical triangulation, `< 1` below typical (damped),
clamped to 1 above. The gate now damps the *worst* voxels of each scene
regardless of the rig's absolute diversity scale.

**Which population defines `Ā`?** Restrict it to voxels at or above `n_min` —
the *trusted* voxels the confidence gate uses anyway:

```
Ā = E[A(q′) | n(q′) ≥ n_min]
```

The naive alternative (all voxels with `n ≥ 2`) fails: as training progresses,
more edge voxels reach `n = 2` — each seen by only two nearby cameras, hence low
diversity — and they drag `Ā → 0.001`, making every `Ã` clamp to 1.0 (no-op
gate). The trusted mean reflects "typical triangulation quality in well-observed
regions," which is the right reference.

### Numerical example

Clown with `Ā = 0.10`:

- `A = 0.12` → `Ã = 1.0` (typical or better, no damping).
- `A = 0.04` → `Ã = 0.40`; `A = 0.02` → `Ã = 0.20` (below typical, damped).

Elf with `Ā = 0.02` (5× smaller absolute scale, same behavior):

- `A = 0.03` → `Ã = 1.0`; `A = 0.005` → `Ã = 0.25`.

The trusted denominator matters. After 500 iterations on clown, suppose 1000
edge voxels (`n = 2`, `A ≈ 0.01–0.05`) and 200 trusted voxels
(`n ≥ 12`, `A ≈ 0.05–0.15`). Naive mean ≈ 0.025 (dominated by edge voxels);
trusted mean ≈ 0.10. A typical trusted voxel with `A = 0.10` reads
`Ã = 4.0 → 1.0` naively (clamped, no damping — wrong) but `Ã = 1.0` with the
trusted denominator (correct). A poor voxel with `A = 0.02` reads `0.80` naively
(mild) but `0.20` trusted (strong damping — correct).

---

## 13. Auto-freeze at the peak

### Problem

Even with the trusted denominator, `Ā` drifts over training: as edge voxels
graduate into the trusted population, the mean shifts. The peak of the trusted
curve occurs early (only "core" well-observed voxels have qualified), then decays.
If we recompute `Ā` per query, the same voxel gets different damping at different
iterations — inconsistent.

### Solution

**Freeze `Ā` once** at the peak of the trusted curve (the "core" reference
population). The peak location varies by scene, so hardcoding a step fails.
Instead, auto-detect: track the running max of `Ā`, and when it hasn't been
beaten for `patience` consecutive edit-steps, freeze at that max.

```
For each iteration t (after warmup):
    current ← mean_angular_factor_trusted()
    if current > peak_value:
        peak_value ← current;  peak_step ← t
    elif t − peak_step ≥ patience:
        freeze Ā ← peak_value;  stop tracking
```

Knobs: `mask_voxel_cache_angular_freeze_warmup` (when tracking starts) and
`mask_voxel_cache_angular_freeze_patience` (the no-improvement window).

### Numerical example

```
step 50:  warmup ends, tracking starts
step 60:  current 0.08 → peak 0.08 @ 60
step 65:  current 0.10 → peak 0.10 @ 65
step 70:  current 0.12 → peak 0.12 @ 70
step 75:  0.11  (no new peak)
step 80:  0.10  (no new peak)
...
step 170: 0.07  → 100 steps without a new peak → freeze Ā ← 0.12
```

For the rest of training `Ā = 0.12` constant, so a voxel with `A = 0.06` is
*always* damped to 50% — never sometimes 20% and sometimes 100%.

---

## 14. Mass gate

### Problem

The confidence so far combines triangulation quality (angular) and statistical
sample size (count) and cross-view agreement (variance). It does *not* directly
ask the most basic question: **is the model committing edit signal to this voxel
at all?** The angular factor couples this in by accident (via the evidence-gated
count), but that conflates two things we'd rather tune separately.

### Solution

An explicit per-voxel mass gate. The cache stores a mean mask value `m(q)` at
each voxel — itself a measure of "how much edit signal has been committed here."
The gate is

```
C_mass(q) = min(1, m(q) / m_threshold)
```

with `m_threshold` = `mask_voxel_cache_mass_threshold` controlling where damping
starts (`m_threshold = 0` disables it). Low-mass voxels (regions the model
consistently says "don't edit") are damped, so the cache stays quiet on
extremities (stormtrooper hand) and doesn't blur high-frequency detail (elf
eyes). It's monotone in `m(q)` — same edit signal always produces the same gate
behavior — and interpretable: "voxels with cached value below the threshold are
treated as background and the cache doesn't contribute there."

### Numerical example

With `m_threshold = 0.3`:

- Torso, `m = 0.7` → `C_mass = min(1, 0.7/0.3) = 1.0` (full contribution).
- Hand, `m = 0.15` → `C_mass = 0.5` (half-damped).
- Background, `m = 0.05` → `C_mass = 0.167` (mostly silent).

(The live config uses `m_threshold = 0.18`.)

---

## 15. Summary

The final fusion equation, as implemented (positive-only):

```
M_final = M_2D + α(t) · C̃(q) · G · max(ΔM, 0)

where:
    ΔM    = M_3D − M_2D
    M_3D  = observed-weighted trilinear read of the grid at p_pixel   (§4)
            then scale-matched to [0,1]                               (§7)
    p_pixel = ray_origin + depth · ray_direction                      (§3)
    α(t)  = warmup ramp from 0 to max_blend                           (§6)
    G     = semantic gate, max(M_attn, M_self)                        (§6)
    C̃(q) = 1[n(q) ≥ n_min]                                           (§9)
            · max(0, 1 − σ²(q)/σ²_max)      (σ²: Welford OR EW, §8)    (§9)
            · min(1, (1 − R(q)) / Ā_frozen)                           (§11–§13)
            · min(1, m(q) / m_threshold)                              (§14)
    Ā_frozen = peak of the trusted-population mean angular factor,
               auto-frozen when no improvement for `patience` steps   (§13)
```

Every factor of `C̃(q)` is **linear** and multiplicative: a voxel must clear the
count gate and have low variance and be well-triangulated and carry real edit
mass to contribute at full strength.

### Knob → symbol map

Every tunable above, the symbol it controls, and where it's explained. Keep this
open while sweeping.

| Knob | Symbol / role | Effect of increasing it | § |
|---|---|---|---|
| `mask_voxel_cache_resolution` | grid `V` | finer grid (but fewer views/voxel) | 2 |
| `mask_voxel_cache_accumulation_threshold` | render-acc gate for backprojection | ignores low-opacity (sky/empty) rays | 3 |
| `mask_voxel_cache_bbox_observe_steps` / `_bbox_observe_quantile` / `_bbox_inflation` | observed-bbox construction | later build / tighter clip / larger box | 3 |
| `mask_voxel_cache_ema_beta_camera_factor` | `c` in EMA `β = 1 − 1/(c·N_cam)` | slower value updates / longer memory | 5 |
| `mask_voxel_cache_max_blend` | ceiling of `α(t)` | stronger cache pull overall | 6 |
| `mask_voxel_cache_warmup_start` / `_end` | ramp window of `α(t)` | later / slower phase-in | 6 |
| `mask_voxel_cache_scale_normalize_quantile` | `q` in the `M_3D` re-ranging stretch | larger `q` = gentler stretch | 7 |
| `mask_voxel_cache_variance_decay` | `σ²` estimator (`α` EW weight; `0.0` = Welford) | forgets stale edit-drift; flattens variance climb | 8 |
| `mask_voxel_cache_max_variance` | `σ²_max` (variance gate) | looser gate, more voxels pass | 9 |
| `mask_voxel_cache_observation_fraction` / `_min_observations_floor` / `_min_observations_cap` | `n_min` (count gate) | needs more views before a voxel counts | 9 |
| `gradient_mask_raw_norm_quantile` *(in `DCConfig`)* | per-frame scale of `raw_self` (the cache input) | robust divisor; less spurious variance | 10 |
| `mask_voxel_cache_angular_freeze_patience` / `_warmup` | when `Ā_frozen` locks | later freeze captures a later peak | 13 |
| `mask_voxel_cache_mass_threshold` | `m_threshold` (mass gate) | higher = damp more low-edit-signal voxels | 14 |

**Fixed behaviors (not knobs).** These are hardcoded on, so don't look for a flag:
observed-weighted trilinear read (§4), scale-matching (§7), positive-only fusion
(§6), the semantic gate `G = max(M_attn, M_self)` (§6), camera-count-auto β (§5)
and `n_min` (§9), scene-relative angular normalization (§11–§12), and the
observed bbox (§3).

### Per-voxel state

```
grid[q]               # EMA mean (running average) — β auto: 1 − 1/(c·N_cam)
observed[q]           # bool: has this voxel ever been observed
unique_view_count[q]  # n(q): number of distinct views
running_mean[q]       # μ (Welford or EW mean)
running_m2[q]         # Welford: M2 → σ² = M2/(n−1);  EW: holds σ² directly
view_observed[q, v]   # bool: which views have observed each voxel
view_dir_sum[q]       # Σ d̂_v for the angular resultant R(q)
```

`geom_view_count[q]` / `view_observed_geom[q, v]` also exist but are
**diagnostic only** — they count every ray that intersected a voxel (ignoring the
evidence gate) so you can measure per scene how much the value gate restricts the
trusted population. The confidence gate uses the evidence-gated
`unique_view_count`.

---

## 16. What's load-bearing

Every mechanism below is active in the default (Part-2) configuration; this table
is the quick status map.

| Mechanism | § | Notes |
|---|---|---|
| Voxel grid + depth backprojection + observed bbox | 2, 3 | core; required |
| Observed-weighted trilinear read | 4 | core; delivers the rendered MV-consistency gain |
| EMA aggregation (camera-count-auto β) | 5 | core |
| Positive-only fusion with the 2D mask | 6 | core; cache adds edit force, never subtracts |
| Scale-matching the cache value | 7 | core; makes `M_3D` comparable to the sharp 2D mask |
| Cross-view variance (Welford **or** EW/decayed) | 8 | EW on (`variance_decay = 0.2`); recency-weighted |
| Confidence gate (count + variance) | 9 | core; `max_variance` is **scene-tuned** |
| Raw-self input + robust per-frame scale | 10 | core; reduces spurious variance |
| Angular-diversity factor (linear, scene-relative) | 11–12 | core |
| Auto-freeze of the angular denominator | 13 | core; keeps the gate consistent across training |
| Mass gate (`C_mass`, linear) | 14 | core; damps low-edit-force voxels |

> **`max_variance` is scene-tuned, not a fixed default.** Read it off
> `dc_debug/voxel_cache_variance_map` and set it in the gap between the wanted
> edit (low variance) and the over-edit region (high variance), staying above the
> former. With `variance_decay` on, read it in the decayed scale (the map already
> is). Read it mid-run on a healthy run: once an over-edit *consolidates* (all
> views agreeing on the wrong edit), its variance collapses and the map can no
> longer separate it.

### How to read this when re-orienting

To understand "what's actually running when I launch the default config":

1. §2–§5 (grid + depth + trilinear read + EMA) — how the cache stores and reads
   values.
2. §6 + §7 (positive-only fusion + scale-matching) — how the cache value is
   prepped and combined with the 2D mask.
3. §8 + §9 (variance + confidence gate) — per-voxel trust from count and
   agreement.
4. §10 (raw-self input + robust scale) — what's fed in and how it's normalized.
5. §11–§14 (angular factor + scene-relative + auto-freeze + mass gate) — the
   remaining confidence factors.
6. Keep the §15 knob → symbol map open while sweeping.
