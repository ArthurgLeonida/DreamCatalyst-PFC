# The Voxel Cache, From Scratch

This document explains how the 3D voxel-cache localization mechanism in
`nerfstudio/dc/mask_voxel_cache.py` works, in the order each piece was
introduced and motivated empirically. Each concept opens with the problem
it solves, gives the equations, then walks through a small numeric example.
The intended reader is someone who knows the basics of NeRF and DDS
(Delta Denoising Score) but has never seen the cache before.

---

## Table of contents

1. [The starting point: why per-view masks aren't enough](#1-the-starting-point)
2. [The basic voxel grid](#2-the-basic-voxel-grid)
3. [Depth backprojection](#3-depth-backprojection)
3b. [Reading the grid back: nearest vs. trilinear](#3b-reading-the-grid-back) — trilinear, always on (hardcoded 2026-07-05)
4. [EMA aggregation across views](#4-ema-aggregation-across-views)
5. [Bidirectional fusion with the 2D mask](#5-bidirectional-fusion-with-the-2d-mask)
5b. [Variance-gated negative branch](#5b-variance-gated-negative-branch)
5c. [Scale-matching the cache value before fusion](#5c-scale-matching) — always on; knob `mask_voxel_cache_scale_normalize_quantile`
5d. [Contested-region suppression (active distrust)](#5d-contested-region-suppression-active-distrust) — retired 2026-06-15 (negative result)
6. [Welford statistics: cross-view variance per voxel](#6-welford-statistics)
6b. [Decayed (exponentially-weighted) variance](#6b-decayed-exponentially-weighted-variance) — knob `mask_voxel_cache_variance_decay`
6c. [Peak-held variance (anti-collapse latch)](#6c-peak-held-variance-anti-collapse-latch) — retired 2026-06-15 (negative result)
7. [Confidence-gated fusion](#7-confidence-gated-fusion)
8. [Per-voxel-count gating (vs. unique-view-count)](#8-per-voxel-count-gating)
9. [Asymmetric fusion (positive vs. negative correction)](#9-asymmetric-fusion)
10. [Raw-self input source](#10-raw-self-input-source)
10b. [Robust per-frame scale](#10b-robust-per-frame-scale-refinement) — knob `gradient_mask_raw_norm_quantile`
11. [Angular-diversity factor](#11-angular-diversity-factor)
11b. [The geometry-vs-evidence misdiagnosis](#11b-the-geometry-vs-evidence-misdiagnosis)
11c. [Mass gate (C_mass)](#11c-mass-gate-c_mass)
12. [Scene-relative normalization](#12-scene-relative-normalization)
13. [Trusted-population denominator](#13-trusted-population-denominator)
14. [Auto-freeze at the peak](#14-auto-freeze-at-the-peak)
15. [Summary of the final fusion equation + knob→symbol map](#15-summary)
16. [What's load-bearing vs. optional in the current config](#16-whats-load-bearing-vs-optional-in-the-current-config)

---

## 1. The starting point

### Problem

DDS (Delta Denoising Score) trains a 3D scene by sampling a single rendered
view per iteration, running it through a 2D diffusion model, and pushing
the gradient back into the NeRF. The diffusion model produces a noise
prediction `ε_tgt` for the target edit and `ε_src` for the source image.
Their difference `ε_tgt − ε_src` localizes the edit: large where the
diffusion model wants to change pixels, small where it doesn't.

This per-view edit signal has two problems:

1. **No cross-view consistency.** Two different cameras observing the same
   3D point can get very different mask values, because the diffusion
   model's predictions depend on the rendered framing, lighting, and
   2D context. The same physical helmet pixel might be "edit strongly"
   in view A and "leave alone" in view B.
2. **Single-iteration noise.** The mask from one view is a sample of a
   stochastic process (the diffusion model's noise prediction). It varies
   across iterations even from the same view.

The goal of the voxel cache is to **lift the per-view mask into 3D and
average evidence across views**, producing a mask signal that is
consistent across viewpoints because it lives in the 3D scene, not in
the 2D image plane.

### Why a voxel grid

We could imagine more elaborate solutions (an MLP that takes a 3D point
and outputs a mask value; per-Gaussian attributes for 3DGS; etc.). The
cache uses a **non-parametric voxel grid** because:

- No additional learning is required — the cache just stores running
  averages.
- It's fast to update and query (just indexing).
- It can be inspected directly (we can save the grid to a 3D file and
  look at it).

The trade-off is grid discretization error, which we control by picking
the resolution.

---

## 2. The basic voxel grid

### What it is

A 3D grid of cells. Each cell stores a single scalar in `[0, 1]`: the
running-average mask value for the 3D region that cell occupies.

Mathematically, let the scene be bounded by an axis-aligned box
`[x_min, x_max] × [y_min, y_max] × [z_min, z_max]`. We divide each axis
into `V` cells of equal width, producing `V³` voxels total. Each voxel
`q` corresponds to a small cube in world space.

A point `p = (x, y, z)` is mapped to a voxel index via

```
idx(p) = floor( (p − bbox_min) / (bbox_max − bbox_min) · V )
```

clamped to `[0, V−1]` on each axis.

### Numerical example

Suppose `bbox_min = (−1, −1, −1)`, `bbox_max = (1, 1, 1)`, and `V = 4`.
Then each axis is divided into 4 cells of width `0.5`.

A point `p = (0.3, −0.7, 0.0)`:

- Normalized: `(0.3 − (−1)) / 2 = 0.65`, `(−0.7 − (−1)) / 2 = 0.15`,
  `(0.0 − (−1)) / 2 = 0.5`
- Multiplied by `V=4`: `(2.6, 0.6, 2.0)`
- Floored: `(2, 0, 2)`

So this point goes into voxel `(2, 0, 2)` in the 4×4×4 grid.

A point at `(2.0, 0.0, 0.0)` falls outside the bbox (normalized x = 1.5).
We mark it as "out of bounds" and skip it during updates.

### What's stored

- `self.grid`: shape `[V, V, V]`, fp32. The running average mask value.
  Initialized to `fallback_value = 0.5` ("uncertain").
- `self.observed`: shape `[V, V, V]`, bool. Whether this voxel has been
  observed at least once. Initially all `False`.

In production we use `V = 64`, so the grid has `64³ ≈ 260k` voxels
totaling ~1 MB at fp32. Coarse but cheap.

---

## 3. Depth backprojection

### Problem

Per-view diffusion masks live in image space: a 2D array of values, one
per pixel. To put them in the 3D voxel grid, we need to know, for each
pixel, **which 3D point that pixel is observing**.

### Solution

NeRF gives us this for free. During rendering, each pixel has an
associated ray `(o, d)` (origin + unit direction) and a learned depth `t`
along the ray (where the surface lies). The 3D point is

```
p = o + t · d
```

This is the standard volume-rendering surface point.

We get the depth from the NeRF's expected-depth output (the integral
`∫ T(t) · σ(t) · t dt` over the ray). We don't need a separate depth
model.

### Numerical example

Pixel `(120, 200)` of a frame has:

- Ray origin `o = (0.0, 1.5, 2.0)` (camera position, world frame)
- Ray direction `d = (0.1, −0.3, −0.95)` (already unit-normalized)
- Depth `t = 2.1`

Then the 3D point is

```
p = (0.0, 1.5, 2.0) + 2.1 · (0.1, −0.3, −0.95)
  = (0.21, 0.87, 0.005)
```

That pixel's mask value (whatever the diffusion model output for it)
will be written into the voxel containing `(0.21, 0.87, 0.005)`.

### Practical detail: the bbox source

For backprojection to work, we need a bbox that **contains the actual
surface points** the cache will index. Naive choices fail:

- **Camera-position AABB.** For object-centric captures (subject sits in
  front of cameras), the subject is *outside* the camera AABB along the
  viewing direction. Every backprojected point falls outside the bbox.
- **Scene-box from dataparser.** Only correct when the dataparser sets
  scene_box in the same frame as the rays.

The fix used in practice: **observed bbox**. For the first
`bbox_observe_steps` (e.g. 50) iterations, the cache is dormant; it
accumulates the backprojected points but doesn't build the grid yet.
Then it takes the AABB of those points (with a quantile clip to remove
far-depth outliers), inflates by some percentage, and uses that as the
final bbox. By construction, this bbox contains exactly the points
the cache will need to index.

---

## 3b. Reading the grid back

> Observed-weighted trilinear read — **always on** (the `mask_voxel_cache_trilinear` knob was removed 2026-07-05 once trilinear became the adopted design; nearest-voxel was the baseline it replaced, kept below for the comparison).

### Problem

Once a pixel is backprojected to a 3D point `p = o + t·d`, we have to read a
mask value out of the grid at `p`. The simplest read is **nearest-voxel**: drop
`p` into the one voxel that contains it and return that voxel's stored mean.

```
M_3D(p) = grid[ idx(p) ]          # nearest-voxel
```

At `V = 64` the voxels are coarse. A small feature (an eye, an armor edge) can
sit near a voxel boundary, so two views whose rays land a hair apart read *two
different voxels* with different values — even though they're looking at the
same surface. That produces a **blocky, view-dependent** mask: stair-steps that
look like cross-view disagreement but are pure quantization. Raising the
resolution would shrink the steps, but it also splits each voxel's observations
across 8× more cells, so every voxel is seen by fewer views and the variance /
angular statistics get noisier — you trade blur for weaker trust signals.

### Solution

**Observed-weighted trilinear interpolation.** Read all 8 voxel centers
surrounding `p`, weight each by the standard trilinear weight `w_c` (how close
`p` is to that corner) *and* by whether that corner has been observed, then
normalize:

```
M_3D(p) = ( Σ_c w_c · observed_c · grid_c ) / ( Σ_c w_c · observed_c )
```

with `c` ranging over the 8 corners. If no surrounding voxel is observed, return
the `fallback_value` (0.5). The `observed_c` weighting is the important part: a
plain trilinear read would blend the 0.5 fallback of unobserved neighbours into
the result and pull edge values toward "uncertain"; weighting by observation
keeps unobserved corners from bleeding in.

This smooths the readback **without** changing the grid resolution, so
observation density — and therefore the variance and angular statistics — is
untouched. **Only the returned value is interpolated**; every trust signal
(`unique_view_count`, variance, angular factor, `observed`/valid, mass) is still
read at the nearest voxel, so the gates behave identically. Trilinear changes
*what magnitude* the cache feeds back, not *whether* it's trusted.

### Numerical example

Point `p` sits 30% of the way from voxel A (value 0.8, observed) toward its
neighbour B along x; the other two axes land exactly on centers. So `w_A = 0.7`,
`w_B = 0.3`.

- **B observed, value 0.6:** `M_3D = (0.7·0.8 + 0.3·0.6)/(0.7+0.3) = 0.74`.
  A smooth blend instead of the hard 0.8 a nearest read would give.
- **B unobserved:** `M_3D = (0.7·0.8 + 0)/(0.7) = 0.8`. B contributes nothing —
  no fallback bleed; the value stays at A's 0.8.

### When it earns its place

Empirically (elf), trilinear was the variant that turned the cache's mask-level
consistency into *rendered* multi-view consistency: `MultiView_pairwise_cos_sim`
rose from 0.928 (cache-off) to 0.933, while the nearest-voxel package left it at
~0.924. The cost was a small editability dip (`CLIP_direction` ≈ 0.127 → 0.127,
within noise). That result is why trilinear became the adopted design and the
nearest-voxel path was removed from the code.

---

## 4. EMA aggregation across views

### Problem

We can't just *overwrite* a voxel's mask value each time a new view
observes it — that throws away all prior evidence. We need to **combine**
evidence from many views.

### Solution

An exponential moving average (EMA). When voxel `q` is observed with
value `m_new`, we update its stored value `m(q)` as

```
m(q) ← β · m(q) + (1 − β) · m_new
```

where `β ∈ [0, 1)` is the decay factor. Higher `β` means slower change
(more memory of past observations).

### How we pick β

Camera-count-aware:

```
β = 1 − 1 / (c · N_cameras)
```

with `c ≈ 2`. For `N_cameras = 65`, `β ≈ 1 − 1/130 ≈ 0.9923`. For
`N_cameras = 365`, `β ≈ 0.9986`.

**Why scale with N_cameras?** A voxel observed by every camera should
have a value close to the *average* across cameras, which means each
single observation contributes about `1/N_cameras`. The choice
`1−β = 1/(c·N_cameras)` makes each observation contribute roughly that
amount, which is the right magnitude for a sample-mean estimator.

### Within-batch averaging

A single view contributes many pixels to the same voxel (the voxel is
larger than a pixel). We don't want to apply the EMA `m_pixel` times for
each pixel — that would let dense regions overwhelm sparse ones. Instead,
we **average all pixels of this view that hit voxel `q`** to get a single
per-view sample, then apply one EMA step.

### Numerical example

Voxel `q` starts at `m(q) = 0.5` (fallback) and has never been observed.
View 1 arrives. Twelve pixels of view 1 map into voxel `q` with mask
values `[0.7, 0.8, 0.75, 0.65, ..., 0.7]`, averaging to `m_new = 0.72`.

Since `q` has never been observed, the first observation takes the value
directly (no EMA blending against the fallback):

```
m(q) ← 0.72
observed(q) ← True
```

View 2 arrives. Five pixels map to `q` with average `m_new = 0.60`. With
`β = 0.99`:

```
m(q) ← 0.99 · 0.72 + 0.01 · 0.60 = 0.7188
```

View 3 arrives, `m_new = 0.65`:

```
m(q) ← 0.99 · 0.7188 + 0.01 · 0.65 = 0.7181
```

The voxel's running average drifts slowly toward whatever the cross-view
mean value is.

---

## 5. Bidirectional fusion with the 2D mask

### Problem

Now we have two sources of mask information per pixel:

- `M_2D(pixel)`: the diffusion model's per-view mask for this iteration.
- `M_3D(pixel)`: the value queried from the voxel cache by
  backprojecting that pixel.

How do we combine them into a single mask that the DDS gradient uses?

### Solution

**Bidirectional fusion**. Define the difference

```
ΔM = M_3D − M_2D
```

and update the 2D mask by

```
M_final = M_2D + α · [ G · max(ΔM, 0) + λ_↓ · min(ΔM, 0) ]
```

Where:

- `ΔM = M_3D − M_2D`, and **`M_3D`** is the cache value read at this pixel by
  observed-weighted trilinear interpolation (hardcoded; §3b).
- `α ∈ [0, 1]` is the cache's pull strength (the "blend factor"). Its ceiling is
  **`mask_voxel_cache_max_blend`** and it ramps from 0 to that ceiling across
  **`mask_voxel_cache_warmup_start` → `_warmup_end`** (§14-adjacent).
- `G` is the semantic gate on the *positive* branch, hardcoded to
  `max(M_attn, M_self)` (the CA mask and self-mask, whichever is stronger).
- `λ_↓` is the suppression ratio on the *negative* branch. **The negative
  branch has been removed from the code** (λ_↓ was 0.0 in every adopted
  config — "positive-only" fusion); it is kept in this document as design
  history (§5b explains why subtraction was rejected).

**Why bidirectional?** Two kinds of disagreement matter:

1. **Cache > 2D (`ΔM > 0`)**: the 3D evidence says "yes this region is
   in the edit area" but the 2D mask is weak here. Positive correction.
   Useful when the per-view diffusion mask missed an edit area that
   other views agreed on.
2. **Cache < 2D (`ΔM < 0`)**: the 3D evidence says "no, this region is
   not in the edit area" but the 2D mask is firing. Negative correction.
   Useful for cleaning up per-view false positives (background speckles
   that one view spuriously flagged as edit-worthy).

### Numerical example

A pixel has `M_2D = 0.4`, `M_3D = 0.7`. So `ΔM = 0.3`. Suppose
`α = 0.4`, `G = 0.8`, `λ_↓ = 0.3`:

```
M_final = 0.4 + 0.4 · [ 0.8 · max(0.3, 0) + 0.3 · min(0.3, 0) ]
        = 0.4 + 0.4 · [ 0.8 · 0.3 + 0.3 · 0 ]
        = 0.4 + 0.4 · 0.24
        = 0.4 + 0.096
        = 0.496
```

The cache nudged the 2D mask up from `0.4` to `0.5`, but only by
~10% because the semantic gate was moderate.

Another pixel: `M_2D = 0.6`, `M_3D = 0.2`, `ΔM = −0.4`.

```
M_final = 0.6 + 0.4 · [ 0.8 · 0 + 0.3 · (−0.4) ]
        = 0.6 + 0.4 · (−0.12)
        = 0.552
```

The cache pulled the 2D mask down from `0.6` to `0.55`. Modest cleanup.

---

## 5b. Variance-gated negative branch

> **Removed from code 2026-07-05.** The negative branch (`λ_↓`) and its
> variance-gated variant (`p_neg`) were `0.0`/off in every adopted
> configuration, so both were deleted when the fusion was hardcoded to
> positive-only. This section is kept as the design analysis that led to
> that decision.

### Problem

The bidirectional fusion treats the positive and negative branches
symmetrically with respect to confidence: both get scaled by the same
`α · C̃(q)` blend factor. Empirically this is wrong. Subtracting edit
signal is more visually destructive than adding it:

- **Adding**: in the worst case, a small extra mask value gets pushed
  through the gradient at a pixel that was already being edited
  moderately. The diffusion model handles this gracefully.
- **Subtracting**: in the worst case, the gradient stops editing a
  pixel that was being edited correctly. Fine high-frequency detail
  (stormtrooper leg armor, helmet outline) gets eroded because the
  cache's averaged-across-views value is moderate while the per-view
  2D mask peaks much higher.

Compared empirically on stormtrooper: with the subtractive branch active
the armor segments degrade visibly. With positive-only fusion (`λ_↓ = 0`,
no negative branch), the armor is preserved but multi-view inconsistency
on the clown returns (the negative branch was cleaning up per-view mask
noise that produces inconsistency).

So the negative branch is doing two things: cleaning up consistent
background noise (good) and eroding fine detail (bad). The two cases
differ in **cross-view variance**:

- Consistent background noise: cache is consistently low, views agree
  → **low variance**, low cache value.
- Fine detail: cache is moderate, views disagree about peak location
  → **high variance**, moderate cache value.

### Solution

Add an extra confidence exponent on the negative branch only:

```
up   = α · C̃(q) · G · max(ΔM, 0)
down = α · C̃(q) · C̃(q)^p_neg · λ_↓ · min(ΔM, 0)
```

The positive branch keeps its single-confidence weighting. The negative
branch is raised to `(1 + p_neg)` power on confidence, where confidence
already includes the variance gate `max(0, 1 − σ²/σ²_max)`. Higher
`p_neg` makes the negative branch progressively more conservative —
it requires very-high agreement (very low variance) to fire.

- `p_neg = 0`: symmetric (default; legacy).
- `p_neg = 1`: confidence squared on negative branch.
- `p_neg = 2`: cubed; only the cleanest-agreement voxels can subtract.

### Numerical example

Two voxels, both with cache mean below 2D mask (so both have `ΔM < 0`):

**Voxel A** — clown background pixel that one view spuriously flagged:
`M_2D = 0.6, M_3D = 0.1, ΔM = −0.5`, `σ²(q) = 0.005` (very consistent),
`C̃(q) = 0.90`, `α = 0.4`, `λ_↓ = 0.3`, `p_neg = 1`.

```
down (legacy)  = 0.4 · 0.90 · 0.3 · (−0.5) = −0.054
down (p_neg=1) = 0.4 · 0.90 · 0.90 · 0.3 · (−0.5) = −0.049
```

Nearly the same. Background cleanup still fires at 90% strength.

**Voxel B** — stormtrooper leg armor edge that views disagree on:
`M_2D = 0.7, M_3D = 0.4, ΔM = −0.3`, `σ²(q) = 0.025` (contested),
`C̃(q) = 0.40`, same other knobs.

```
down (legacy)  = 0.4 · 0.40 · 0.3 · (−0.3) = −0.0144
down (p_neg=1) = 0.4 · 0.40 · 0.40 · 0.3 · (−0.3) = −0.00576
```

Negative correction is **2.5× weaker** on the contested voxel with
`p_neg = 1`. The fine-detail edit signal is mostly preserved. With
`p_neg = 2`, the same voxel reads `−0.00230` — negative branch is
nearly silent on contested voxels, while the background cleanup
(voxel A) drops to only `−0.044` (still ~80% of legacy strength).

This asymmetric gate reconciles the two observed failure modes:
clown's multi-view inconsistency (needs negative cleanup) and
stormtrooper's armor erosion (needs negative branch silenced on fine
detail).

---

## 5c. Scale-matching

> **Knob: `mask_voxel_cache_scale_normalize_quantile`** (`q`, default `0.95`). The scale-matching itself is always on (the on/off flag was removed once every adopted config had it enabled).

### Problem

The fusion in §5 differences two quantities of *different nature*:

- `M_2D` is a **sharp** per-view mask — a near-indicator that peaks close to 1
  on the edit region.
- `M_3D` is a **multi-view mean** — averaging across views compresses it toward
  mid-range, and under-observed voxels carry the 0.5 fallback.

So even when the cache and the 2D mask *agree on where the edit is*, `M_3D` sits
systematically **below** `M_2D`'s peaks across the whole edit region. That makes
`ΔM = M_3D − M_2D < 0` over the edit itself, so the negative ("down") branch
fires on the genuine edit — not because of real 3D disagreement, but purely
because the two masks live on different value scales. The cache ends up
*eroding* the edit it was supposed to support.

### Solution

Before fusion, **contrast-stretch the queried cache mask to [0,1]** over its
observed voxels, so its active range matches the sharp 2D mask. Take the
`[1−q, q]` percentiles of `M_3D` over the observed pixels and rescale:

```
lo = quantile(M_3D[observed], 1 − q)
hi = quantile(M_3D[observed], q)
M_3D' = clamp( (M_3D − lo) / (hi − lo), 0, 1 )
```

- **`mask_voxel_cache_scale_normalize_quantile`** is `q` (default `0.95`): the
  upper percentile mapped to 1 (and `1−q = 0.05` mapped to 0). Larger `q` → less
  aggressive stretch (only the extreme tail saturates); smaller `q` (e.g. 0.85)
  → harder stretch, more pixels pinned to 0/1. It's clamped to `[0.5, 0.999]`.

The selection uses the *observed* mask (not confidence), so the low end of the
range captures background voxels; invalid/fallback pixels carry zero confidence
downstream, so their post-stretch value never reaches the gradient.

### Numerical example

Over the edit region the cache reads `M_3D ∈ [0.30, 0.55]` (compressed), while
`M_2D` peaks near 0.9. With `q = 0.95`, suppose `lo = 0.32`, `hi = 0.53`.

- A foreground voxel `M_3D = 0.52` → `(0.52 − 0.32)/(0.53 − 0.32) = 0.95`. Now it
  sits *above* a typical `M_2D`, so `ΔM > 0` and the **positive** branch can
  support the edit instead of the negative branch eroding it.
- A background voxel `M_3D = 0.33` → `≈ 0.05`. Stays low, so on real background
  (where `M_2D` is also low) `ΔM ≈ 0` and neither branch does much.

In short: scale-matching is what stops the §5 down-term from firing on the edit
purely as a units mismatch, leaving it to clean up only true background
false-positives.

> **Interaction with trilinear (§3b):** scale-matching reads percentiles over the
> queried values, so it runs on what the trilinear read produced.
> The two compose; trilinear smooths the value, scale-matching re-ranges it.

### 5d. Contested-region suppression (active distrust)

> **RETIRED — removed from code on 2026-06-15** (it never beat the 2D base
> rate; see "Empirical status" below). This section is kept as a development
> record of an idea that was tried and rejected; the `external_mask_contested_suppression_ratio`
> knob no longer exists in `dc.py`.

Every fusion branch so far multiplies by the per-voxel confidence — so where
the variance gate *distrusts* a voxel, the cache simply **abstains** (adds
nothing, subtracts nothing). That is the right default, but it means the cache
can never *fight* a 2D over-edit: on the clown arms, confidence ≈ 0 zeroes the
up branch **and** the down branch, and the 2D process is left alone to paint
at its base rate.

This branch flips the sign of distrust. Where the cache has **enough
cross-view evidence** (`n ≥ n_min`) and that evidence **disagrees**
(`σ² ≥ σ²_max`), high variance is itself information — "the views do not
agree this should be edited" — and the 2D mask is damped there:

```
contested(q) = valid · 1[n(q) ≥ n_min] · min(1, σ²(q)/σ²_max)
M_final      = M_2D + up + down − α(t) · ratio · contested · M_2D
```

Two design points:

- It uses the **scalar warmup blend** `α(t)`, *not* the confidence-carrying
  `blend_map` — using blend_map would re-introduce the abstention exactly
  where the term must act. Max damping at `ratio = 1.0` is `max_blend = 0.2`,
  i.e. ≤ 20% of M per step on fully-contested pixels.
- Suppressing `M` acts twice: it damps the edit force (source-blend collapses
  toward `eps_src` as M → 0) **and** strengthens the `(1 − M)`-scaled
  outside-mask anchor — the restoring force that recovers the original skin.
  The common worry "suppressing M also suppresses recovery" is backwards in
  this architecture: recovery is preservation-driven, not edit-force-driven.

**Diagnostics / kill criteria:** `dc_debug/voxel_cache_contested_map` (always
logged, even at `ratio = 0`) must light up the over-edit region and stay dark
on the wanted edit. If it doesn't, the mechanism is falsified by the map alone
— stop without burning seeds. The count gate `1[n ≥ n_min]` is load-bearing:
without it, early-training voxels with immature variance estimates would get
suppressed scene-wide.

**Empirical status (2026-06-11): retired at `ratio = 0`.** The A/B told a
two-part story. `ratio = 2.0` failed 3/3 (P ≈ 0.037 under the 1/3 base rate →
causally harmful) via self-extinction: strong suppression holds the arms
uniform across views, the contested signal fades, confidence reopens, and the
positive branch rebounds — the suppressor manufactures the failure it was
built to prevent. `ratio = 1.0` initially went 3/3 clean, but extended seeds
landed at 2 failures / 5 total — statistically indistinguishable from the base
rate (the 3/3 was a ~30%-likely draw). Net: the gain's stability ceiling sits
*below* any gain that could beat the 2D base rate. Both the mechanism and the
ceiling finding are documented in the thesis (ch. 4/7). (A theoretical caveat
worth keeping in mind if ever re-enabling: **emergent geometry** — e.g. the
stormtrooper helmet — naturally accumulates the "enough views + high variance"
signature, so suppression would anchor new structure toward the source
background. Untested; the observed stormtrooper belly artifact occurs with the
ratio at 0.0 and is a separate, cache-package-level issue.)

---

## 6. Welford statistics

### Problem

The EMA gives us the *mean* mask value at each voxel. But it doesn't tell
us how *consistent* the views were. A voxel could read `m(q) = 0.5`
because:

- All views agreed on `0.5` (consistent → trustworthy)
- Half the views said `0.0` and half said `1.0` (contested → unreliable)

We need a measure of cross-view agreement: **variance**.

### Solution

Welford's online algorithm. For each voxel, maintain a running mean and
a running sum of squared deviations from the mean (`M2`). When a new
per-view sample `x` arrives at a voxel that has already been observed
by `n − 1` distinct views:

```
n ← n + 1
δ  ← x − μ_{n−1}
μ_n ← μ_{n−1} + δ / n
δ′  ← x − μ_n
M2  ← M2 + δ · δ′
σ² ← M2 / (n − 1)        # sample variance
```

This is numerically stable (Welford 1962, popularized by Knuth TAOCP
vol. 2) and updates in O(1) per sample. We need:

- `unique_view_count[q]`: how many distinct views have observed `q`.
- `running_mean[q]`: μ.
- `running_m2[q]`: M2.

### Critical detail: unique views

If the same camera observes the same voxel at iteration 100 and iteration
2700, we should NOT count that as two samples — it's the same view
contributing twice, and counting it twice would deflate the variance
estimate (the view agrees with itself by construction).

To enforce this, we track a `[n_voxels, num_views]` boolean tensor
`view_observed[q, v]` that records which views have ever observed each
voxel. The Welford update only fires when a *new* view first observes
a voxel.

### Numerical example

Voxel `q` is freshly created. View 1 contributes a per-voxel mean of 0.8:

- `n = 1`, `μ = 0.8`, `M2 = 0` (Welford needs n ≥ 2 for variance).

View 2 contributes 0.7:

- `δ = 0.7 − 0.8 = −0.1`
- `n = 2`, `μ = 0.8 + (−0.1)/2 = 0.75`
- `δ′ = 0.7 − 0.75 = −0.05`
- `M2 = 0 + (−0.1) · (−0.05) = 0.005`
- `σ² = M2 / (n−1) = 0.005`

View 3 contributes 0.9:

- `δ = 0.9 − 0.75 = 0.15`
- `n = 3`, `μ = 0.75 + 0.15/3 = 0.80`
- `δ′ = 0.9 − 0.80 = 0.10`
- `M2 = 0.005 + 0.15 · 0.10 = 0.020`
- `σ² = 0.020 / 2 = 0.010`

After three views, the voxel has mean 0.80 and variance 0.010 — that
is, a standard deviation of ~0.10 between views.

Contrast with a contested voxel:

- View 1 contributes 0.2.
- View 2 contributes 0.9.
- After two views, μ = 0.55, M2 = 0.245, σ² = 0.245.

Same number of observations, much higher variance → contested.

### 6b. Decayed (exponentially-weighted) variance

> **Knob: `mask_voxel_cache_variance_decay`** = the EW weight `α` below (`0.0` = cumulative Welford, default; `(0,1)` = exponential).

The cumulative Welford estimate weights every distinct view equally over the
*entire* run. But the edit is **non-stationary**: the scene changes as DDS
training proceeds. A voxel first observed by camera A at edit-step 500 (immature
edit) and by camera B at edit-step 30000 (mature edit) records the A↔B
difference as cross-view "variance" — when it is really cross-*time* drift, not
view disagreement. The symptom is `mean_observed_variance` climbing monotonically
over the run (observed empirically on elf: 0.0175 → 0.033 across 33k steps),
which slowly walks the average voxel up to the `max_variance` ceiling and chokes
the gate late in training.

The fix is an exponentially-weighted variance (Finch 2009). With new-view weight
`α`, on each new view's sample `x`:

```
delta = x − μ
μ    ← μ + α · delta
σ²   ← (1 − α) · (σ² + delta · α · delta)
```

Because views first-observe a voxel in training-time order, `α` down-weights the
stale early-edit samples, so `σ²` tracks the *current* cross-view disagreement
among recent (mature-edit) observations. Effective memory ≈ `1/α` views
(`α = 0.2` ≈ last 5). Controlled by `mask_voxel_cache_variance_decay`: `0.0` =
cumulative Welford (default), `(0,1)` = EW with that weight. Caveat: the EW
estimator is biased low by ≈ `(1 − α)`, so its absolute scale differs from the
cumulative one — re-read `query_variance_mean` and re-tune `max_variance` after
enabling.

One honesty note on "tracks the current disagreement": the per-(view, voxel)
statistics update only on each view's **first** observation of a voxel (§8), so
for static geometry the statistic is effectively **frozen after roughly one
pass over the cameras**. The EW weighting therefore reweights among the
first-observation samples in arrival order — it does *not* keep tracking
disagreement on later revisits. That is exactly the failure §6c addresses.

### 6c. Peak-held variance (anti-collapse latch)

> **RETIRED — removed from code on 2026-06-15** (the warmup transient dominated
> the peak and the late-warmup config already prevents the collapse it
> targeted; see "Empirical status" below). Kept as a development record; the
> `mask_voxel_cache_variance_peak_decay` knob no longer exists in
> `mask_voxel_cache.py`.

§6b makes the variance estimate *forget* old samples; this section makes the
**gate** *remember* the worst of them. The two compose: the EW estimate is
still what each new sample contributes — the peak is a running max **of** that
estimate.

**The failure it fixes (observed on clown).** The cache amplifies the model's
consistent moderate edit-force on the arms (§9: positive-only fusion can only
add force). Once the arms actually get painted, every later-arriving view
observes the *consolidated* over-edit, agrees with the others, and the
recency-biased EW variance contracts geometrically (×(1−α) per agreeing
sample). The frozen per-voxel variance ends **low** — below any usable
`max_variance` — so the agreement gate re-admits exactly the voxels it was
built to exclude, and the over-edit feeds itself. At that point tightening
`max_variance` cannot help: the bright arms you saw in the early variance map
are simply no longer bright in the statistic the gate reads. (This is the
"a bad run's over-edited region collapses to low variance once painted"
caveat, promoted from a map-reading footnote to the actual mechanism.)

**Mechanism.** Alongside `running_m2`, each voxel keeps `running_var_peak`.
On every new per-view sample (same evidence-gated population as §8):

```
σ²_now   = (EW or Welford estimate after this sample)
peak(q) ← max(peak(q) · ρ, σ²_now)        # ρ = variance_peak_decay
```

and `query` gates on — and returns/visualizes —

```
σ²_gate(q) = max(σ²_now(q), peak(q))
```

so `dc_debug/voxel_cache_variance_map` always shows what the gate sees. With
`ρ = 1.0` a voxel that ever showed cross-view disagreement above the gate
stays distrusted for the whole run ("once contested, always contested");
`ρ < 1` forgives slowly, at one decay step per *new-view* sample.

**What it costs.** The latch is asymmetric by design: it can only make the
gate stricter, never looser. Regions whose variance is genuinely transient
(brief early disagreement that would legitimately settle) stay gated — on
scenes where the cache was already healthy (elf), reproduce the validated
package with `ρ = 0.0`. Read the (now peak-held) variance map the same way as
before: the wanted edit should stay dark, the over-edit region bright; put
`max_variance` in the gap. The floor may sit slightly higher than the
instantaneous map's, since the wanted region's peak ≥ its frozen estimate.

**Empirical status (clown A/B, 2026-06-10): retired at `ρ = 0`.** The "what
it costs" paragraph above turned out to be the whole story: with `ρ = 1.0`
the run-wide peak is dominated by the immature-edit **warmup transient** —
every voxel's worst disagreement happens while the edit is still forming, so
`mean_observed_variance_peak` climbs to ~2× the gate (≈0.03 vs 0.015) and
confidence is zeroed body-wide (the person vanishes from the confidence map;
the cache silently no-ops and runs revert to the 2D base rate). Meanwhile the
collapse this section was built against did **not** occur under the late
warmup (500→1200): the stats accumulate before blending starts, and even the
bad run's arms stay bright/gated in the late variance map. The collapse
observed earlier traced to `warmup_start = 100` + `max_blend = 0.4` —
blending *during* accumulation contaminated the statistic. Conclusion: the
latch addresses a real failure mode, but the late warmup already prevents it;
re-enabling would require starting the peak tracking only after edit
maturity (a `variance_peak_warmup`), which is unimplemented and currently
unmotivated.

---

## 7. Confidence-gated fusion

### Problem

Not every voxel deserves equal trust. We want to:

- Ignore voxels observed by too few unique views (insufficient evidence).
- Down-weight voxels with high cross-view variance (contested evidence).

This trust signal should multiply the fusion strength, not the cache's
mean value itself — we still want to *learn* every observation, just
not necessarily use every voxel.

### Solution

Define a per-voxel confidence `C(q) ∈ [0, 1]`:

```
C(q) = 1[n(q) ≥ n_min] · max(0, 1 − σ²(q)/σ²_max)
```

Where:

- `n(q)` is the unique-view count at voxel `q`.
- `n_min` is a minimum observation threshold (camera-count-aware:
  `n_min = clamp(ceil(N_cameras · 0.10), [5, 12])`).
- `σ²(q)` is the running variance (Welford or EW, §6/§6b); with peak-hold
  enabled (§6c) it is `max(σ²_now, peak)`.
- `σ²_max` is a cap above which a voxel is fully untrusted (scene-tuned;
  0.04 ≈ std dev 0.2 on a [0, 1] scale is used in the examples below —
  the live value is in `method_config.py`).

Then the fusion equation gets `C(q)` multiplying the bracketed term:

```
M_final = M_2D + α · C(q) · [ G · max(ΔM, 0) + λ_↓ · min(ΔM, 0) ]
```

A voxel with no observations or high variance contributes nothing.

### Numerical example

Voxel `q1`: `n = 15`, `σ² = 0.01`, `n_min = 12`, `σ²_max = 0.04`.

```
C(q1) = 1 · max(0, 1 − 0.01/0.04) = 1 · 0.75 = 0.75
```

Voxel `q2`: `n = 8`, `σ² = 0.01`.

```
C(q2) = 0 · ... = 0       (insufficient observations)
```

Voxel `q3`: `n = 20`, `σ² = 0.05` > σ²_max.

```
C(q3) = 1 · max(0, 1 − 0.05/0.04) = 1 · max(0, −0.25) = 0
```

The first voxel passes both gates and contributes 75% of nominal
strength. The second is silenced for low view count. The third is
silenced for high variance.

---

## 8. Per-voxel-count gating

### Problem

The Welford statistics treat per-view *per-voxel* samples. But we
defined the EMA as updating from the within-batch *per-voxel mean* of
pixels in that view. An update value gate (`update_threshold`) also
existed here: if a voxel's per-view mean fell below `T`, the update was
skipped entirely, so early under-edited samples could not poison the EMA
or the variance. **The gate has been removed from the code** — it was
`0.0` (off) in every adopted config, because the warmup ramp (§14)
already keeps immature evidence from mattering. Two design notes from
its lifetime remain useful:

- the gate operated on the per-voxel mean, not raw pixel values —
  filtering pixels first would still let a voxel with mostly-low
  evidence contribute a sample (a few high-value pixels survive);
- the count gate below (`n_min`) is what actually carries the
  "not enough evidence yet" protection today.

### Numerical example

A view contributes 10 pixels to voxel `q`. Their mask values:
`[0.1, 0.05, 0.1, 0.7, 0.65, 0.08, 0.05, 0.1, 0.1, 0.07]`.

With pixel-level gating at threshold 0.5, only 2 pixels survive,
average = 0.675. That voxel would write 0.675 to the cache.

With per-voxel-mean gating, the average of all 10 is 0.21, which is
below 0.2 only if `T = 0.25`. If `T = 0.2`, it passes (barely) and
writes 0.21. If `T = 0.25`, it skips entirely — even though 2 pixels
were "confident," the *voxel's evidence from this view* is mostly low.

The per-voxel-mean gate is more conservative and prevents the cache
from learning misleadingly-high values on voxels that have only
sparse pixel evidence.

---

## 9. Asymmetric fusion

### Problem

The bidirectional fusion equation has two gates:

- `G` (semantic, like CA) for positive corrections.
- The cache's own confidence `C(q)` (now part of `α`) for both.

But asking semantic permission to *add* cache evidence is different from
asking permission to *subtract*:

- **Adding** mask says "this region is in the edit area." That's a
  semantic claim. It should require semantic confirmation.
- **Subtracting** mask says "this region is not in the edit area."
  That's a *consistency* claim — if multiple views agree the voxel is
  background, the cache should be able to clean up a per-view false
  positive without needing CA to also agree.

### Solution

The bidirectional equation already encodes this asymmetry: positive
corrections are gated by `G`, negative corrections are gated by
`λ_↓` only (not by `G`). The cache's confidence `C(q)` and warmup
factor `α` apply to both branches, but the semantic gate is on the
positive branch only.

### Numerical example

Background pixel: `M_2D = 0.6` (spurious false positive from one view),
`M_3D = 0.1` (multiple views agreed this is background). `ΔM = −0.5`,
`G = 0.1` (low semantic support — CA correctly doesn't fire on the
background), `C(q) = 0.9`, `α = 0.4`, `λ_↓ = 0.3`.

```
M_final = 0.6 + 0.4 · 0.9 · [ 0.1 · max(−0.5, 0) + 0.3 · min(−0.5, 0) ]
        = 0.6 + 0.36 · [ 0 + 0.3 · (−0.5) ]
        = 0.6 + 0.36 · (−0.15)
        = 0.6 − 0.054
        = 0.546
```

The cache pulled the background pixel from 0.6 to 0.55 *despite* the
semantic gate being weak — because the negative branch doesn't ask
for semantic permission. This is the cleanup mechanism.

Compare: foreground pixel `M_2D = 0.3` (under-detected by one view),
`M_3D = 0.8`, `ΔM = 0.5`, `G = 0.9` (CA fires strongly here),
`C(q) = 0.9`, `α = 0.4`:

```
M_final = 0.3 + 0.4 · 0.9 · [ 0.9 · 0.5 + 0.3 · 0 ]
        = 0.3 + 0.36 · 0.45
        = 0.3 + 0.162
        = 0.462
```

The cache pulled it up from 0.3 to 0.46. This is the support
mechanism — and it required `G = 0.9` to fire.

---

## 10. Raw-self input source

### Problem

What gets fed into the cache as "the 2D mask value"? Originally we used
the **internal mask** — the same one used by the DDS gradient. That mask
goes through several preprocessing steps:

1. Compute `||ε_tgt − ε_src||` per pixel.
2. **Percentile-normalize** per-sample (preserves spatial ranking but
   compresses values toward the median).
3. Temporal EMA per view.
4. Gamma/blur post-processing.
5. Fuse with cross-attention mask.

The percentile normalization is the culprit. It maps the *rank* of each
pixel's value, not the absolute magnitude. A scene with high
foreground/background contrast and a scene with low contrast end up
producing visually similar masks after percentile normalization. **The
cache needs absolute value information** to discriminate.

Symptom: the `update_hist` of per-voxel means was unimodal near zero
with a thin tail — no separable trough between foreground and background.
No threshold could partition them cleanly.

### Solution

Bypass percentile normalization for the cache input. Use `raw_self`:

```
raw_self = ||ε_tgt − ε_src|| / max_per_sample
```

Per-sample max normalization preserves within-frame absolute structure
(foreground pixels stay high, background pixels stay low) while keeping
values in [0, 1]. The DDS gradient still uses the percentile-normalized
mask (that's the right choice for the gradient because it's invariant
to noise-level scale), but the cache gets the absolute-scale version.

### Numerical example

A frame has raw `||ε_tgt − ε_src||` values ranging from 0.05 to 1.4.

**Percentile normalization** would map the 50th-percentile value to 0.5,
the 95th to 0.95, etc. The resulting distribution is approximately
uniform in [0, 1]. Foreground (top 20% of pixels) lands at 0.8–1.0,
background (bottom 80%) lands at 0.0–0.8. **Background gets a lot of
mass near 0.5–0.7**, because uniformity stretches it out.

**Max normalization** maps `1.4 → 1.0` (linear scaling, no rank). The
50th-percentile pixel might have value 0.2 in raw → 0.14 after
normalization. Background stays low, foreground stays high. The
distribution is now bimodal-like.

After this fix, the update histogram becomes bimodal and the threshold
gate works properly.

### 10b. Robust per-frame scale (refinement)

> **Knob: `gradient_mask_raw_norm_quantile`** (lives in `DCConfig`, not the cache config; `1.0` = legacy per-frame max, default; `0.95` = divide by p95).

Max normalization divides every pixel of a frame by that frame's single
brightest pixel. That divisor is a one-pixel statistic, so it is fragile:
if one frame contains an outlier pixel (a specular highlight, a denoising
artifact) at `||ε_tgt − ε_src|| = 1.4` while another frame's hottest pixel
is `0.9`, the *same* 3D point with true relevance `0.5` reads `0.5/1.4 ≈
0.36` in the first frame and `0.5/0.9 ≈ 0.56` in the second. That `Δ ≈ 0.20`
is **spurious cross-view variance manufactured by the divisor**, not real
edit disagreement — and `0.20` of spread contributes variance on the order
of `0.003–0.01`, which is a large fraction of a `max_variance ≈ 0.02–0.035`
gate budget. The cache then mistakes a normalizer artifact for view
inconsistency and over-suppresses.

The fix is to divide by a **robust per-frame quantile** (e.g. the 95th
percentile) instead of the max. It is still a monotone linear rescale, so it
preserves the absolute foreground/background contrast §10 restored — pixels
above the quantile simply saturate to `1.0` — but a single hot pixel can no
longer rescale the whole frame. Controlled by `gradient_mask_raw_norm_quantile`
(`dc.py:_build_gradient_relevance_mask`): `1.0` = legacy max, `0.95` = p95.
This is the single cheapest reduction of spurious cross-view variance, and it
must be applied *before* tightening the variance gate, or the gate is tuned
against the normalizer rather than the edit.

---

## 11. Angular-diversity factor

### Problem

Even with the confidence gate (`C(q) > 0` requiring enough views and
low variance), a voxel can be "confidently wrong" if all observing
cameras are clustered in viewpoint. Cross-view variance is only
meaningful when the views are *geometrically distinct*. If 10 cameras
all look at the helmet from nearly the same direction, they agree by
construction — but their agreement doesn't certify the cache value.

Diagnosed empirically: on the elf scene (65 cameras, clustered
viewpoints), variance ran at 0.005 and `confidence_coverage_0.5 = 0.99`
— near saturation. The gate was effectively off. But cache contribution
hurt the elf edit, erasing the subject. The variance signal couldn't
detect that the views weren't actually triangulating.

### Solution

A second gate that measures *angular* diversity, not just count and
variance. Use circular statistics (Fisher 1953, *Dispersion on a
sphere*). For each voxel `q`, maintain a running sum of unit ray
directions:

```
S(q) = Σ_v d̂_v(q)
```

Where `d̂_v(q)` is the unit direction from camera `v` to voxel `q`
(per-voxel mean direction of the pixels of view `v` that hit `q`).

The **resultant length** is

```
R(q) = ||S(q)|| / n(q)  ∈ [0, 1]
```

- `R(q) = 1`: all observing rays were parallel (no triangulation
  evidence).
- `R(q) = 0`: rays uniformly spread on the sphere (maximal triangulation).

The diversity factor is

```
A(q) = 1 − R(q)
```

It gets multiplied into the confidence:

```
C̃(q) = C(q) · A(q)
```

### Numerical example

Voxel `q1`, observed by 4 cameras with directions:

```
d̂_1 = (0, 0, −1)
d̂_2 = (0, 0, −1)
d̂_3 = (0.05, 0.01, −0.998)   (slightly different)
d̂_4 = (0, 0.02, −0.9998)     (also nearly the same direction)
```

Sum: `S ≈ (0.05, 0.03, −3.996)`. `||S|| ≈ 3.996`. `R = 3.996 / 4 = 0.999`.
`A = 1 − 0.999 = 0.001`.

This voxel was observed from a tight cone of viewpoints. The diversity
factor crushes its confidence to near zero — the gate identified it as
"not well triangulated."

Voxel `q2`, observed by 4 cameras from four sides:

```
d̂_1 = (1, 0, 0)
d̂_2 = (−1, 0, 0)
d̂_3 = (0, 1, 0)
d̂_4 = (0, −1, 0)
```

Sum: `S = (0, 0, 0)`. `R = 0`. `A = 1.0`. Maximal diversity — the gate
fully trusts this voxel.

---

## 11b. The geometry-vs-evidence misdiagnosis

### What we thought the problem was

The angular factor uses `n(q)` (the value-gated `unique_view_count`)
as its denominator. That count only increments when a view's per-voxel
mean clears `value_threshold`, so the angular factor accidentally
measures "cameras that contributed *confident mask evidence* to this
voxel," not "cameras that *geometrically saw* this voxel."

The hypothesis: on stormtrooper, helmet voxels' angular factor was
artificially low because most cameras observing the helmet failed the
value gate early in training (helmet is a generated feature with no
source-image mask signal), so the gate damped the helmet for the wrong
reason. The proposed fix: maintain a separate geometry-only counter
that fires regardless of mask value.

### What actually happened when we built it

We implemented the geometry/evidence split. On stormtrooper, the
helmet *did* return visually. But every scene also developed new
artifacts:

- Stormtrooper grew geometric flaws on hands and crotch — regions the
  diffusion model wasn't strongly editing.
- Elf eyes became blurry.
- Clown stayed roughly the same but slightly muddier.

The frozen denominator dropped 30–35 % on every scene, including
clown — but clown's behavior shouldn't change if the diagnosis was
that the geometry/evidence gap was scene-specific. Something deeper
was wrong with the framing.

### The actual diagnosis

The value-gated coupling we thought was a bug was **load-bearing**.
Pre-Fix-B, voxels with low cache mean (regions the model isn't editing
much) had small `view_observed` populations because their views often
failed the value gate. Small populations meant low angular factor (a
small set of cameras observing a voxel ends up clustered by chance),
which damped the cache's confidence on those voxels. The pre-Fix-B
gate was implicitly performing "damp cache contribution on voxels
without strong edit signal" — a sensible policy, just disguised
inside the angular math.

When we decoupled geometry from evidence, we *removed* that policy.
The angular factor became a clean geometric quantity, and voxels with
no edit activity now passed the gate at full strength. The cache
contributed to those regions, producing the artifacts we observed.

The lesson: the cache's confidence should not depend purely on
"is this voxel well-triangulated." It should also depend on "is
the model actually committing edit signal to this voxel." The
pre-Fix-B coupling enforced the second condition by accident.
The clean version of this policy is an explicit gate on the cache's
own cached mean value, which §11c introduces.

### What we kept from the experiment

The state added for geometry-only tracking (`view_observed_geom`,
`geom_view_count`) is retained as a *diagnostic*. Comparing it
against the evidence-gated `unique_view_count` tells you per scene
how much the value gate is restricting the trusted population —
useful for analysis but not load-bearing in the gate math.

---

## 11c. Mass gate (C_mass)

### Problem

After §11b's revert, the angular factor is once again coupled to
mask-value evidence by accident, via `unique_view_count`. That works
on the scenes we've tested, but it conflates three different things:

1. Triangulation quality (where the angular factor *should* live)
2. Statistical sample size (where the count gate `n(q) ≥ n_min` lives)
3. Edit-signal presence (where the model is actively committing edits)

Conflating (1) and (3) means we can't independently tune them. We
also can't reason cleanly about why the gate is damping a particular
voxel — it could be triangulation, count, or edit absence.

### Solution

Add an explicit per-voxel mass gate. The cache stores a mean mask
value `m(q)` at each voxel (the EMA of per-view contributions).
That mean is itself a measure of "how much edit signal has been
committed here" — low values mean the model has consistently said
"don't edit," high values mean "edit strongly." The gate is

```
C_mass(q) = min(1, m(q) / m_threshold)
```

with `m_threshold` controlling where damping starts (`m_threshold = 0`
disables the gate). The final confidence becomes

```
C̃(q) = 1[n(q) ≥ n_min]
       · max(0, 1 − σ²(q) / σ²_max)
       · min(1, (1 − R(q)) / Ā)
       · min(1, m(q) / m_threshold)
```

### Why this is cleaner than the implicit coupling

- The mass gate is monotone in `m(q)`. Same edit signal always produces
  the same gate behavior; no dependence on which cameras happened to
  observe the voxel first.
- The mass threshold is tunable independently of triangulation and
  variance. We can sweep it scene-by-scene if needed.
- The gate is interpretable: "voxels with cached value below 0.3 are
  treated as background and the cache doesn't contribute there." That's
  a sentence a reader can verify.
- It composes cleanly with the positive-only fusion: damping the
  confidence on low-mass voxels directly mutes the cache's added support
  exactly where the model isn't committing edit signal.

### Numerical example

Voxel `q1`: stormtrooper torso, cached mean `m(q) = 0.7`, `m_threshold = 0.3`.

```
C_mass = min(1, 0.7/0.3) = 1.0
```

Full contribution preserved.

Voxel `q2`: stormtrooper hand, cached mean `m(q) = 0.15`.

```
C_mass = min(1, 0.15/0.3) = 0.5
```

Half-damped — the cache contributes to the hand, but at reduced
strength.

Voxel `q3`: background near subject, cached mean `m(q) = 0.05`.

```
C_mass = min(1, 0.05/0.3) = 0.167
```

Strongly damped. The cache is mostly silent on this voxel.

(A steepening exponent on this ramp was explored — e.g. squaring would
take `q3` to 0.028 — but the linear ramp was adopted and the exponent
removed from the code.)

---

## 12. Scene-relative normalization

### Problem

The absolute `A(q) = 1 − R(q)` is bounded by the capture geometry,
not by [0, 1] in a scene-comparable way. Two examples:

- **Clown** (365 cameras orbiting horizontally at roughly waist level):
  every voxel is observed by cameras whose ray directions have y-component
  ≈ −0.95 and z-component ≈ 0. The mean direction is dominated by the
  y-component. `R ≈ 0.95`. `A = 0.05`. Even well-triangulated clown
  voxels look "untrustworthy" by absolute standards.
- **Forward-facing capture**: cameras all look in roughly the same
  direction. `R ≈ 0.99`. `A = 0.01`.

If we apply `A = 1 − R` directly, the gate collapses confidence on
*every* voxel in both scenes, even the well-triangulated ones. The gate
becomes a no-op damping multiplier.

### Solution

Scene-relative normalization. Divide by the scene-wide mean angular
factor:

```
Ã(q) = min(1, A(q) / Ā)
```

Where `Ā = E[A(q′) | trusted voxels q′]` — the average angular factor
over voxels that pass the confidence gate.

Now `Ã(q) = 1` for voxels at the scene's typical triangulation,
`Ã(q) < 1` for voxels below typical, `Ã(q) > 1` for voxels above
(clamped to 1). The gate damps the *worst* voxels of each scene
regardless of the rig's absolute diversity scale.

### Numerical example

Clown scene with `Ā = 0.10`:

- Voxel with `A(q) = 0.12` → `Ã = 1.0` (typical or better, no damping).
- Voxel with `A(q) = 0.04` → `Ã = 0.40` (below scene-typical, damped).
- Voxel with `A(q) = 0.02` → `Ã = 0.20` (much worse, strongly damped).

Elf scene with `Ā = 0.02` (smaller absolute, but the math is invariant):

- Voxel with `A(q) = 0.03` → `Ã = 1.0` (above scene-typical).
- Voxel with `A(q) = 0.005` → `Ã = 0.25` (below scene-typical).

Same gate behavior on both scenes, even though their absolute scales
differ by 5x.

---

## 13. Trusted-population denominator

### Problem

What population do we average over to compute `Ā`?

Naive choice: **all voxels with `n(q) ≥ 2`**. Sounds reasonable. But
this fails empirically: as training progresses, more voxels reach
`n = 2` — *including* edge voxels that just barely got their second
camera observation. These edge voxels have low diversity (only two
nearby cameras saw them), so they drag `Ā` down. After a few hundred
iterations, `Ā → 0.001` even on scenes where well-triangulated voxels
have `A = 0.12`. Every voxel's `Ã = A / Ā ≫ 1` clamps to 1.0. Gate
becomes a no-op.

### Solution

Restrict `Ā` to voxels at or above `min_observations`. These are the
*trusted* voxels — the ones the confidence gate will use anyway. Their
mean reflects "typical triangulation quality in well-observed regions
of this scene," which is the right reference for the relative gate.

```
Ā = E[A(q′) | n(q′) ≥ n_min]
```

### Numerical example

After 500 iterations on clown, the voxel population has:

- 1000 voxels with `n = 2` (edge voxels, `A` ranging 0.01–0.05).
- 200 voxels with `n ≥ 12` (trusted, `A` ranging 0.05–0.15).

Naive mean over all multi-view voxels: dominated by the 1000 edge
voxels, ≈ 0.025.

Trusted mean (over the 200 voxels with `n ≥ 12`): ≈ 0.10.

A typical trusted voxel with `A = 0.10`:

- Naive: `Ã = 0.10 / 0.025 = 4.0 → 1.0` (clamped, no damping).
- Trusted: `Ã = 0.10 / 0.10 = 1.0` (no damping — correct, it's typical).

A poor voxel with `A = 0.02`:

- Naive: `Ã = 0.02 / 0.025 = 0.80` (mild damping).
- Trusted: `Ã = 0.02 / 0.10 = 0.20` (strong damping — correct, it's
  below scene-typical).

The trusted version gives the gate real discriminative power.

---

## 14. Auto-freeze at the peak

### Problem

Even with the trusted-population denominator, `Ā` drifts over training.
As more edge voxels graduate into the trusted population, the mean
shifts. The peak of the trusted curve occurs early (e.g. edit-step 60
on elf, 80 on clown — when only "core" voxels with many observations
have qualified), and then decays as more edge voxels enter.

If we recompute `Ā` per query, the gate's behavior changes throughout
training: the same voxel gets different damping at different iterations.
That's inconsistent.

### Solution

**Freeze `Ā` once** at a representative value. The peak of the trusted
curve corresponds to the "core" voxel population — the well-observed,
well-triangulated reference voxels. We want to freeze there.

But the peak location varies by scene. Hardcoding "freeze at step
2500" fails (it's after the peak — drifted value). "Freeze at step
100" works for clown but maybe not for elf.

Auto-detection: track the running max of `Ā` across iterations. When
the max hasn't been beaten for `patience` consecutive edit-steps,
freeze at that max.

```
For each iteration t (after warmup):
    current ← mean_angular_factor_trusted()
    if current > peak_value:
        peak_value ← current
        peak_step ← t
    elif t − peak_step ≥ patience:
        freeze Ā ← peak_value
        stop tracking
```

The cache enters a no-improvement window after the curve starts
decaying. After `patience` iterations of no new peak, we conclude the
peak has passed and lock in.

### Numerical example

Edit-step 50: warmup ends. Tracking starts.
Edit-step 60: `current = 0.08`. peak_value = 0.08, peak_step = 60.
Edit-step 65: `current = 0.10`. peak_value = 0.10, peak_step = 65.
Edit-step 70: `current = 0.12`. peak_value = 0.12, peak_step = 70.
Edit-step 75: `current = 0.11`. No new peak.
Edit-step 80: `current = 0.10`. No new peak.
...
Edit-step 170: `current = 0.07`. 100 steps without new peak.
**Patience reached → freeze `Ā ← 0.12`.**

For the rest of training, `Ā = 0.12` constant. The gate behavior is now
consistent: a voxel with `A = 0.06` is *always* damped to 50%, not
sometimes to 20% (when `Ā = 0.08`) and sometimes to 100% (when
`Ā = 0.04`).

---

## 15. Summary

The final fusion equation, with all pieces (as implemented — positive-only;
the historical negative term `C̃^p_neg · λ_↓ · min(ΔM, 0)` is analyzed in §5b):

```
M_final = M_2D + α(t) · C̃(q) · G · max(ΔM, 0)

where:
    ΔM    = M_3D − M_2D
    M_3D  = readback of grid at p_pixel        (observed-weighted trilinear; §3b)
            then scale-matched to [0,1]         (§5c)
    p_pixel = ray_origin + depth · ray_direction
    α(t)  = warmup ramp from 0 to max_blend
    C̃(q) = 1[n(q) ≥ n_min]
            · max(0, 1 − σ²(q)/σ²_max)           (σ²: Welford OR EW, §6/§6b)
            · min(1, (1 − R(q)) / Ā_frozen)
            · min(1, m(q) / m_threshold)
    Ā_frozen = peak of mean_angular_factor_trusted over training,
               auto-frozen when no improvement seen for `patience` steps
    G     = semantic gate, max(M_attn, M_self)
    (the negative branch and its λ_↓ were removed from code; §5/§5b history)
```

### Knob → symbol map (what you're actually tuning)

Every tunable in the equations above, the symbol it controls, and where it's
explained. This is the table to keep open while sweeping.

| Knob | Symbol / role | Effect of increasing it | § |
|---|---|---|---|
| `mask_voxel_cache_max_blend` | ceiling of `α(t)` | stronger cache pull overall | 5, 14 |
| `mask_voxel_cache_warmup_start` / `_end` | ramp window of `α(t)` | later/slower phase-in | 5 |
| `mask_voxel_cache_scale_normalize_quantile` | `q` in the (always-on) `M_3D` re-ranging stretch | larger `q` = gentler stretch | 5c |
| `mask_voxel_cache_max_variance` | `σ²_max` (variance gate) | looser gate, more voxels pass | 7 |
| `mask_voxel_cache_variance_decay` | `σ²` estimator (`α` EW weight; `0.0` = Welford) | forgets stale edit-drift; flattens variance climb | 6b |
| `mask_voxel_cache_observation_fraction` / `_floor` / `_cap` | `n_min` (count gate, always camera-count-auto) | needs more views before a voxel counts | 7, 8 |
| `mask_voxel_cache_angular_freeze_patience` / `_warmup` | when `Ā_frozen` locks | later freeze captures a later peak | 14 |
| `mask_voxel_cache_mass_threshold` | `m_threshold` (mass gate) | higher = damp more low-edit-signal voxels | 11c |
| `mask_voxel_cache_ema_beta_camera_factor` | `c` in the (always-auto) EMA β = 1 − 1/(c·N_cam) | slower value updates / longer memory | 4 |
| `mask_voxel_cache_accumulation_threshold` | render-acc gate for backprojection | ignores low-opacity (sky/empty) rays | 3 |
| `mask_voxel_cache_resolution` | grid `V` | finer grid (but fewer views/voxel) | 2 |
| `gradient_mask_raw_norm_quantile` *(in `DCConfig`)* | per-frame scale of `raw_self` (the cache input) | robust divisor; less spurious variance | 10b |

Hardcoded design decisions (formerly knobs; removed 2026-07-05 once every
adopted config used a single value):

- **trilinear read** always on (`mask_voxel_cache_trilinear`, §3b);
- **scale-matching** always on (`mask_voxel_cache_scale_normalize`, §5c);
- semantic gate fixed at `G = max(M_attn, M_self)`
  (`external_mask_screen_attn_gate_strength`, `_self_boost_lambda`, §5/§9);
- **negative branch deleted** — fusion is positive-only
  (`external_mask_interp_suppression_ratio`, `external_mask_negative_variance_power`, §5/§5b);
- angular factor always **scene-relative**, no floor
  (`mask_voxel_cache_angular_relative`, `_min_angular_factor`, §11/§12);
- count gate always camera-count-auto (`mask_voxel_cache_min_observations`, `_auto`, §7);
- EMA β always camera-count-auto (`mask_voxel_cache_ema_beta`, `_auto`, §4);
- **update value gate deleted** (`mask_voxel_cache_update_threshold`, §8);
- bbox always from observed points (`mask_voxel_cache_bbox_source`, §3);
- passive measure-only mode removed (`mask_voxel_cache_measure_only`);
- angular and mass gates fixed **linear** — the steepness exponents
  (`mask_voxel_cache_angular_power`, `mask_voxel_cache_mass_power`) were
  `1.0` in every adopted config and were removed (§11, §11c).

Removed earlier as negative results (2026-06-15): `external_mask_contested_suppression_ratio`
(active-distrust damp; never beat the base rate, §5d) and
`mask_voxel_cache_variance_peak_decay` (peak-held variance gate; warmup
transient dominated it, §6c).

And the per-voxel state:

```
grid[q]               # EMA mean (running average) — β auto: 1 − 1/(c·N_cam)
observed[q]           # bool: first-observation flag
unique_view_count[q]  # n(q): number of distinct views
running_mean[q]       # μ (Welford or EW mean)
running_m2[q]         # Welford: M2 → σ² = M2/(n−1);  EW: holds σ² directly (§6b)
view_observed[q, v]   # bool: which views have observed each voxel
view_dir_sum[q]       # Σ d̂_v for angular factor R(q)
```

### Reading the equation

The fusion is a **gated additive correction**. The 2D mask `M_2D` is
the baseline. The cache contributes either positive (adding edit
support, gated by `G`) or negative (cleaning up false positives, ungated
by `G`) corrections. Both contributions are scaled by:

- The warmup blend `α(t)`: lets the cache phase in gradually as it
  accumulates evidence.
- The per-voxel confidence `C̃(q)`: count, variance, and angular
  diversity must all pass.

### The development arc, in one paragraph

The cache started as a simple EMA-aggregated voxel grid (`§4`) with
bidirectional fusion (`§5`). A variance-gated negative branch (`§5b`)
later reconciled fine-detail preservation with per-view-noise cleanup
on different scenes. Welford statistics added cross-view variance (`§6`), which became part of a confidence gate (`§7`).
Asymmetric fusion (`§9`) decoupled the semantic gate from the cleanup
mechanism. Switching the cache input to raw-self values (`§10`) restored
absolute foreground/background contrast that percentile normalization
had been destroying. The angular-diversity factor (`§11`) caught a
failure mode the variance gate alone couldn't: views agreeing because
they're parallel. A geometry-vs-evidence decoupling attempt (`§11b`)
exposed that the value-gated angular factor was load-bearing — it
implicitly damped cache contribution on low-edit-signal voxels — and
the mass gate (`§11c`) made that policy explicit and tunable.
Scene-relative normalization (`§12`) made the angular gate work
across capture geometries. The trusted-population denominator (`§13`)
prevented the relative gate from collapsing to a no-op. Auto-freeze
at the peak (`§14`) made the denominator a true constant across
training, ensuring gate consistency.

The most recent round targeted the helps-clown/hurts-faces split directly,
by making the lifted signal and its readback more *view-invariant*: a robust
per-frame scale for `raw_self` (`§10b`) removed single-pixel-max jitter; a
positive-only fusion (`§5`, `λ_↓ = 0`) stopped the cache eroding fine detail;
scale-matching (`§5c`) stopped the down-branch firing on the edit as a units
artifact; observed-weighted trilinear readback (`§3b`) converted the cache's
mask-level consistency into rendered multi-view consistency without a
resolution/density penalty; and a decayed (exponentially-weighted) variance
(`§6b`) stopped the cross-view variance estimate from climbing as the edit
drifts over training. The clown arm over-edit then exposed the decay's blind
spot — a consolidating over-edit makes late views agree, collapsing the
recency-biased variance below any gate value — and the peak-held variance
(`§6c`) was built to latch the early disagreement so the gate stays shut
(later retired: the warmup transient dominated the run-wide peak).

Each step traces to a specific empirical failure: a wandb panel that
showed unexpected behavior, a CLIP_direction number that regressed, a
visual artifact in a confidence mask. The cache as it stands is not the
result of design from first principles — it's the accumulated response
to roughly ten failure modes uncovered by iterative experimentation
across five scenes (face, stormtrooper, clown, bear, einstein) with
different camera distributions and edit types.

---

## 16. What's load-bearing vs. optional in the current config

Not all of the mechanisms above are *active* in the final default
configuration. Some are kept on the instance for diagnostics; others
are off-by-default knobs we plan to revisit. This section maps each
mechanism to its status.

Live status as of 2026-06-15 (matches `method_config.py`):

| Mechanism | §  | Current | Status |
|---|---|---|---|
| Voxel grid + depth backprojection + observed bbox | 2, 3 | on | core; required |
| EMA aggregation across views (camera-count-aware β) | 4 | on | core |
| Fusion machinery (bidirectional up/down branches) | 5, 9 | up only — down branch **off** via `interp_suppression_ratio=0` | core machinery; **positive-only** active |
| Variance-gated negative branch (`p_neg`) | 5b | **off** (`p_neg = 0`) | weakens suppression — don't use for over-edits |
| Contested-region suppression (`external_mask_contested_suppression_ratio`) | 5d | **removed from code 2026-06-15** | no effect at safe gain (2/5 fails ≈ base rate), harmful at 2.0 (self-extinction rebound); never beat the base rate |
| Scale-matching cache value (`_scale_normalize_quantile`) | 5c | **always on** (`q = 0.95`, hardcoded 2026-07-05) | core; makes `M_3D` comparable to the sharp 2D mask |
| Cross-view variance — Welford **or EW/decayed** | 6, 6b | **EW on** (`variance_decay = 0.2`) | core; recency-weighted, kept (helped elf) |
| Peak-held variance latch (`variance_peak_decay`) | 6c | **removed from code 2026-06-15** | tested on clown: warmup transient dominates the peak → confidence zeroed body-wide (cache no-ops); the late-warmup config already protects the stats |
| Confidence gate (count + variance) | 7 | on (`max_variance = 0.015` clown override, **tune per scene**) | core |
| Per-voxel-mean update gate (`update_threshold`) | 8 | **removed from code 2026-07-05** (was `0.0`/off in every adopted config) | — |
| Raw-self input source | 10 | on | core |
| Robust per-frame scale (`gradient_mask_raw_norm_quantile`) | 10b | on (`0.95`) | core; reduces spurious variance |
| Trilinear cache read | 3b | **always on** (hardcoded 2026-07-05) | core; delivers the rendered MV-consistency gain (small editability cost) |
| Angular-diversity factor | 11 | **always on** (linear, scene-relative) | core |
| Geometry-vs-evidence decoupling (§11b state) | 11b | **diagnostic only** | counters kept; gate uses evidence-gated count |
| Mass gate (`C_mass`) | 11c | **on** (`m_thr = 0.18`, linear) | core; damps low-edit-force voxels |
| Scene-relative normalization | 12 | on | core |
| Trusted-population denominator | 13 | on | core |
| Auto-freeze at the peak | 14 | on (`patience = 100`) | core |

The "core" mechanisms compose into the current config. **Removed from the code
(2026-07-05)** after being off in every adopted config: the negative fusion
branch and its variance-gated variant `p_neg` (5/5b — redundant when the
over-edit region is variance-distinguished, see clown) and the
`update_threshold` gate (8). They remain documented here as design history.
§11b is kept in the code as a diagnostic only.

> `max_variance` is **scene-tuned**, not a fixed default: read it off
> `dc_debug/voxel_cache_variance_map` and set it in the gap between the wanted
> edit (low variance) and the over-edit region (high variance), staying above the
> former. With `variance_decay` on, read it in the decayed scale (the map already is).
> Read it around mid-run on a healthy run: once an over-edit *consolidates*
> (all views agreeing on the wrong edit), its variance collapses and the map
> can no longer separate it. (The retired §6c peak-hold latch was an attempt
> to lift exactly this caveat.)

### How to read this when re-orienting

If you come back to this document after a break and want to understand
"what is actually running when I launch the default config":

1. Read §2-4 (grid + depth + EMA) — these are always on. **§3b** = how the
   value is read back (observed-weighted trilinear, always on).
2. Read §5 + §5c + §9 (bidirectional fusion + scale-matching + asymmetric
   gating) — the fusion math and how `M_3D` is prepped before differencing.
3. Read §6 + §6b + §6c + §7 (Welford / decayed variance + peak-hold latch +
   confidence gate) — the per-voxel trust and which variance estimator feeds
   `σ²_max`.
4. Read §10 + §10b (raw-self input + robust per-frame scale) — what's fed into
   the cache and how it's normalized.
5. Read §11 + §12 + §13 + §14 (angular factor + scene-relative + trusted
   denominator + auto-freeze) — the angular-gating chain.
6. Keep the **§15 knob → symbol map** open while sweeping — it's the index of
   every tunable and the symbol it moves.

Skip §5b, §11b, and §11c on a first re-read — they're "history of
attempts" sections useful for thesis context but not for understanding
what the running code does.
