#!/usr/bin/env python3
"""
Evaluate a DreamCatalyst editing experiment.

Renders all views from a trained checkpoint, computes metrics against
the original (unedited) images, and saves results to JSON.

Metrics:
  - CLIP_text_sim:   cosine similarity between edited image and target prompt
  - CLIP_direction:  directional CLIP similarity (editing faithfulness)
  - CLIP_img_sim:    cosine similarity between original and edited image (identity)
  - SSIM:            structural similarity (identity preservation)
  - LPIPS:           perceptual distance (lower = more similar to original)
  - MultiView_consistency_std:  std of the per-view CLIP embeddings
                                (lower = more consistent)
  - MultiView_pairwise_cos_sim: mean cosine similarity between the CLIP
                                embeddings of every pair of distinct views
                                (higher = more consistent)
  - EditMaskVariance_3D: variance of a rendered edit-magnitude mask after
    backprojection into world-space voxels (lower = more 3D-consistent masks).
    Reported next to EditMaskVariance_3D_normalized (the same variance divided
    by the squared mean edit mass, so it is invariant to how hard the config
    edited) and the grid provenance keys _mean_value, _num_voxels,
    _mean_observations, _bbox_min, _bbox_max, _bbox_source. Comparing two runs
    requires the same partition: pass --mask-variance-bbox-min/-max.

Region-localized metrics (PIE-Bench-style; all computed against a soft region
mask R segmented from the SOURCE image only, so the region is identical for
every method compared on a scene).

R is anchored on the SOURCE noun by default (--region-anchor union segments the
source frame for both the source noun and the target phrase and takes the max).
That default matters: every target prompt in this project is an instruction
naming something the source does not contain yet ("Turn him into a
stormtrooper", "Put a knight's helmet on him"), so segmenting the source for the
target concept asks for something provably absent. The thing being REPLACED is
always there. See the anchor table further down for which scene needs what.
  - CLIP_direction_local:     CLIP-direction with out-of-region change removed.
                              The editability number that is not inflated by
                              repainting the background.
  - CLIP_direction_bg:        CLIP-direction with in-region change removed.
                              Near zero for a well-localized edit; a large
                              positive value quantifies background leakage.
  - Edit_Localization_Ratio:  area-normalized share of the total change that
                              landed inside R (0.5 = uniform, higher = better
                              localized).
  - Background_LPIPS:         perceptual distance outside R (lower = better).
  - Background_PSNR:          PSNR outside R (higher = better).
  - Edit_Magnitude_in_region / _out_region:
                              mean |edit - src| on each side of R. NOT scores.
                              CLIP_direction_local is a cosine and
                              Edit_Localization_Ratio is a contrast, so both are
                              blind to how much actually changed; these two make
                              that checkable.
  - Region_valid_views:       how many views produced a usable region. Read it
                              against num_views: the region-restricted means are
                              over that subset, while CLIP_direction is over
                              every view.
  - Region_mask_mean_coverage: mean soft area of R. Report it next to the
                              metrics; a degenerate value invalidates them.

Every region-restricted entry also carries "n" (the number of views its mean is
over) in metrics.json, exported by collect_metrics.py as eval/<key>_n, and a
per-view value (or null) under results["per_view"]. The provenance of R lives in
the top-level "region" block.

CLIP_direction (global) is still reported: the local/bg pair explains it, it
does not replace it. The new metrics are skipped cleanly (with a warning) when
the CLIPSeg segmenter is unavailable, and are turned off by
--no-localized-metrics.

This is NOT a novel metric family. It follows the established PIE-Bench
protocol (background-restricted preservation plus region-restricted CLIP
similarity), applied per view to multi-view renders.

Three caveats that belong in any write-up of these numbers:

  1. CLIP_direction_bg near zero means the background did not move TOWARD the
     target text. It does not mean the background was preserved: floaters,
     blur and random drift are large changes that are roughly orthogonal to
     text_delta and score near zero too. Read it with Background_LPIPS and
     Background_PSNR, which do measure preservation.
  2. The pixel split is exact, the CLIP split is not. See region_composites:
     (edit_fg - src) + (edit_bg - src) = edit - src holds for any R, but CLIP
     is nonlinear, so local and bg are a diagnostic decomposition of the global
     number, not an algebraic identity on it.
  3. The whole suite rests on R. R is segmented from the source frames, so it
     is method-agnostic, but it is not necessarily CORRECT. Inspect the cached
     masks in <output_dir>/region_masks/ before trusting a result, and report
     Region_mask_mean_coverage next to the metrics.

Recommended protocol for a fair comparison
------------------------------------------
Evaluate ONE run of a scene, then point every other run of that scene at its
masks:

  python scripts/evaluate.py eval --config <runA>/config.yml ... \
      --output-dir eval_results/<scene>_runA
  python scripts/evaluate.py eval --config <runB>/config.yml ... \
      --output-dir eval_results/<scene>_runB \
      --region-mask-dir eval_results/<scene>_runA/region_masks

Re-running the segmenter per method would also give the same masks (same
source frames, same prompts, deterministic model), but sharing the folder makes
the region provably the same bytes for every method instead of merely the same
in principle, which is the version a reviewer can check.

Then evaluate the UNEDITED reconstruction as one more row, with the same
prompts and the same masks. It is the empirical zero of this suite. A NeRF
render never equals its source frames, so the floor of CLIP_direction_local is
not 0 but whatever the reconstruction error alone scores, and the floor of
Edit_Magnitude_in_region is not 0 either. Without that row there is no way to
tell a real edit from reconstruction noise that happened to point the right
way, and CLIP_direction_local is a cosine, so it does not shrink when the
change does.

--region-anchor and --region-dilate-frac must be fixed per SCENE and shared by
every method on it. Sharing the mask folder enforces that mechanically. The
part that code cannot enforce is choosing them from the source frames before
looking at any metric, which is the honest way to use them.

Usage:
  python scripts/evaluate.py eval \
      --config outputs/bicycle/dc_splat/<timestamp>/config.yml \
      --src-prompt "a photo of a bicycle" \
      --tgt-prompt "a photo of a motorcycle" \
      [--output-dir eval_results/bicycle_exp001]

  # Attach the metrics to the original WandB run if available:
  python scripts/evaluate.py eval ... --log-wandb

  # Override the auto-derived region phrase, or supply your own masks:
  python scripts/evaluate.py eval ... --region-prompt "cake"
  python scripts/evaluate.py eval ... --region-mask-dir path/to/masks

  # Which phrase the SOURCE frame is segmented for. Default 'union' (source
  # noun OR target phrase) covers every scene here; the other two are ablations:
  python scripts/evaluate.py eval ... --region-anchor source

  # Reproduce the pre-region metric set exactly:
  python scripts/evaluate.py eval ... --no-localized-metrics

  # Edits that grow geometry outside the source object (fangzhou helmet,
  # stormtrooper armor) need headroom, or the new geometry counts as leakage:
  python scripts/evaluate.py eval ... --region-dilate-frac 0.03
"""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml


WANDB_RUN_DIR_RE = re.compile(r"(?:offline-)?run-\d{8}_\d{6}-([a-z0-9]+)", re.IGNORECASE)
WANDB_RUN_FILE_RE = re.compile(r"(?:offline-)?run-([a-z0-9]+)\.wandb$", re.IGNORECASE)


def load_clip_model(device):
    """Load CLIP model for text-image similarity."""
    import clip
    model, _ = clip.load("ViT-L/14", device=device)
    model.eval()
    return model


def clip_encode_image(model, images, device):
    """Encode a batch of PIL images with CLIP. Images in [0,1] tensor [B,C,H,W]."""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    images = F.interpolate(images.float(), size=224, mode="bicubic", align_corners=False)
    images = (images - mean) / std
    with torch.no_grad():
        features = model.encode_image(images)
    return features / features.norm(dim=1, keepdim=True)


def clip_encode_text(model, texts, device):
    """Encode text prompts with CLIP."""
    import clip
    tokens = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
    return features / features.norm(dim=1, keepdim=True)


def compute_ssim(img1, img2):
    """Compute SSIM between two [H,W,3] float32 numpy arrays in [0,1]."""
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, channel_axis=2, data_range=1.0)


def compute_lpips(img1_tensor, img2_tensor, lpips_model):
    """Compute LPIPS between two [1,C,H,W] tensors in [0,1]."""
    # LPIPS expects [-1, 1]
    with torch.no_grad():
        return lpips_model(img1_tensor * 2 - 1, img2_tensor * 2 - 1).item()


# ==============================================================================
#  Region-localized edit metrics (PIE-Bench-style)
# ==============================================================================
#  CLIP_direction is computed over the WHOLE image, so it rewards a method that
#  changes everything: recolouring the background toward the target concept
#  moves the global image embedding in the same direction as the text edit and
#  scores higher than an edit that only touches the intended object. Nothing in
#  the global number says WHERE the change happened.
#
#  The metrics below split that one number in two, using a soft region mask R:
#  the change inside R (what was asked for) and the change outside R (leakage).
#  R is segmented from the SOURCE frame only. The reconstruction checkpoint is
#  held fixed per scene across every method being compared, so the GT/source
#  frames are byte-identical across methods and R is method-agnostic by
#  construction. NEVER derive R from an edited render or from a method's own
#  internal mask: that would be circular and would favour whichever method
#  produced the mask.
#
#  This is not a novel metric. It mirrors the established PIE-Bench protocol
#  (background-restricted preservation + region-restricted CLIP similarity).
#  CLIP_direction (global) is still reported: the claim is that the pair
#  (local, bg) EXPLAINS the global number, not that the global one is wrong.
#
#  WHICH ANCHOR EACH SCENE NEEDS (--region-anchor, --region-dilate-frac)
#  ---------------------------------------------------------------------
#  Every target prompt in this project is an INSTRUCTION naming something that
#  does not exist in the source yet, so segmenting the source for the target
#  phrase asks for something provably absent. The default anchor is therefore
#  the source noun, unioned with the target phrase so the rule stays universal.
#
#    scene / edit                 source noun      target phrase   notes
#    bear -> panda                "bear statue"    "panda"         source anchor;
#                                                                  target phrase
#                                                                  may partly fire
#                                                                  (both are bears)
#    face -> Tolkien elf          "face"           "tolkien elf"   source anchor
#    face -> Einstein             "face"           "einstein"      source anchor
#    person -> clown              "person"         "clown"         source anchor
#    person -> stormtrooper       "person"         "stormtrooper"  source anchor +
#                                                                  dilation: armor
#                                                                  grows past the
#                                                                  silhouette
#    fangzhou -> knight helmet    "face"           "knight helmet" source anchor +
#                                                                  dilation: the
#                                                                  helmet sits
#                                                                  ABOVE the head,
#                                                                  outside any
#                                                                  face mask
#    yuseung -> Batman            "man"            "batman"        source anchor
#    yuseung -> Joker             "man"            "joker"         source anchor
#    velvet cake -> chocolate     "velvet cake"    "chocolate      either anchor
#      cake                                         cake"          works here: the
#                                                                  cake IS in the
#                                                                  source, which is
#                                                                  why this edit is
#                                                                  the easy case
#
#  Read that table as: without dilation, fangzhou and stormtrooper will report
#  worse localization than they deserve, for EVERY method equally. That is a
#  conservative bias, not a favourable one, but it should be disclosed rather
#  than left for a reviewer to find.

DEFAULT_CLIPSEG_MODEL = "CIDAS/clipseg-rd64-refined"
REGION_PHRASE_MODES = ("diff_anchored", "diff", "head_noun", "target_prompt")
REGION_ANCHOR_MODES = ("union", "source", "target")
REGION_MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
# Minimum mask weight, measured in "pixels worth of R", for a side of the
# partition to count as present. Below this a ratio/PSNR denominator is
# degenerate and the view is dropped rather than reported as a fake value.
REGION_MIN_AREA = 1.0
# Reported ceiling for a numerically perfect background (mse == 0).
BACKGROUND_PSNR_CAP = 100.0
# Absolute floor on the divisor used to rescale a CLIPSeg probability map.
# WITHOUT this, a map that found nothing (every probability ~0.02, which is what
# CLIPSeg returns when the queried concept is absent from the image) is divided
# by its own p95 ~0.02 and comes back as a near-all-ones mask. An all-ones R
# makes edit_fg == edit and edit_bg == src, i.e. it reports "the edit was
# perfectly localized and leaked nothing" for EVERY method. That is a silent
# fabrication in the flattering direction, so the divisor never drops below a
# probability that a genuine detection clears easily.
REGION_MIN_PROB_SCALE = 0.25
# A view whose final R covers essentially none or essentially all of the frame
# carries no partition, so it is dropped from the localized metrics instead of
# producing the degenerate numbers described above.
REGION_DEGENERATE_MIN_COVERAGE = 0.002
REGION_DEGENERATE_MAX_COVERAGE = 0.98
# Stopwords used when the training-time helper cannot be imported. Kept in sync
# with DEFAULT_CROSS_ATTENTION_STOPWORDS in nerfstudio/dc/localization_utils.py,
# plus the instruction verbs and the possessive fragment that the word regex
# splits off ("a knight's helmet" -> knight, s, helmet). Those extra entries
# matter here and not during training: a CLIPSeg text query is a noun phrase,
# so "put knight s helmet" is a materially worse query than "knight helmet",
# whereas cross-attention only ever reweights the tokens it is handed.
REGION_EXTRA_STOPWORDS = {
    "put", "add", "give", "change", "replace", "convert", "transform", "swap",
    "s", "it", "its", "he", "she", "they", "who", "which", "some", "any",
}


_PROMPT_TOKEN_HELPERS = None


def _load_prompt_token_helpers():
    """Import the training-time prompt-token helpers (single source of truth).

    The target phrase is derived by the same rule the cross-attention mask uses
    during editing, so the evaluation region and the training-time keywords
    cannot drift apart. Returns ``(derive_cross_attention_keywords, stopwords)``
    with the keyword function set to ``None`` when the ``dc`` package is not
    importable (evaluate.py run outside the training environment); the stopword
    set is always usable, so the source anchor keeps working either way. Cached
    so the import warning is printed at most once per process.
    """
    global _PROMPT_TOKEN_HELPERS
    if _PROMPT_TOKEN_HELPERS is not None:
        return _PROMPT_TOKEN_HELPERS
    _PROMPT_TOKEN_HELPERS = _import_prompt_token_helpers()
    return _PROMPT_TOKEN_HELPERS


def _import_prompt_token_helpers():
    try:
        from dc.localization_utils import (
            DEFAULT_CROSS_ATTENTION_STOPWORDS,
            derive_cross_attention_keywords,
        )
    except Exception as exc:
        print(
            f"WARNING: Could not import dc.localization_utils ({exc}); "
            "the region phrases fall back to a local copy of the stopword list."
        )
        return None, set(REGION_EXTRA_STOPWORDS) | {
            "a", "an", "the", "of", "to", "into", "in", "on", "at", "for", "with",
            "from", "photo", "image", "picture", "turn", "make", "him", "her",
            "them", "his", "their", "this", "that", "is", "as", "be", "and", "or",
        }
    return derive_cross_attention_keywords, (
        set(DEFAULT_CROSS_ATTENTION_STOPWORDS) | REGION_EXTRA_STOPWORDS
    )


def _content_words(prompt, stopwords):
    """Content words of a prompt, in order, with stopwords and duplicates gone."""
    words = re.findall(r"[a-zA-Z0-9]+", (prompt or "").lower())
    content = []
    for word in words:
        if word in stopwords or word in content:
            continue
        content.append(word)
    return content


def _target_head_noun(tgt_prompt, stopwords):
    """Last content word of the target prompt, used as its head noun."""
    content = _content_words(tgt_prompt, stopwords)
    return content[-1] if content else None


def derive_source_region_phrase(src_prompt):
    """Name the thing the edit REPLACES, from the source prompt alone.

    This is the anchor that works when the target concept is absent from the
    source image, which is the normal case for this project: every target
    prompt here is an instruction naming something that does not exist yet
    ("Turn him into a stormtrooper", "Put a knight's helmet on him"), so
    querying the SOURCE frame for the target concept asks the segmenter to find
    something that is provably not there. The source noun is always present by
    construction, and it is still derived from the source prompt only, so it
    stays method-agnostic.

    "a photo of a bear statue" -> "bear statue"; "a photo of a person" ->
    "person"; "a photo of a velvet cake" -> "velvet cake". Returns ``None`` when
    the source prompt has no content words at all.
    """
    _, stopwords = _load_prompt_token_helpers()
    content = _content_words(src_prompt, stopwords or set())
    if not content:
        return None
    # A source prompt is a short noun phrase, so the last few content words are
    # the noun phrase; keeping at most three avoids dragging in scene chatter.
    return " ".join(content[-3:])


def derive_region_phrase(src_prompt, tgt_prompt, mode="diff_anchored"):
    """Pick the phrase that names the region the edit is supposed to touch.

    Both prompts are shared by every method compared on a scene, so the phrase
    is method-agnostic like the mask it produces.

      - ``diff_anchored`` (default): the target-only content words, plus the
        target head noun when the diff did not already include it. The bare
        diff is often an attribute (velvet cake -> chocolate cake diffs to just
        "chocolate"), which is a poor query against a SOURCE image that
        contains no chocolate; anchoring it to the noun it modifies gives
        "chocolate cake", which still localizes the cake.
      - ``diff``: the target-only content words alone (the literal prompt diff).
      - ``head_noun``: the target head noun alone.
      - ``target_prompt``: the whole target prompt.

    Every mode falls back down the chain when its own step yields nothing.
    ``--region-prompt`` overrides all of this.
    """
    tgt_prompt = (tgt_prompt or "").strip()
    if mode == "target_prompt" or not tgt_prompt:
        return tgt_prompt

    derive_keywords, stopwords = _load_prompt_token_helpers()
    head_noun = _target_head_noun(tgt_prompt, stopwords)
    if mode == "head_noun":
        return head_noun or tgt_prompt
    if derive_keywords is None:
        return head_noun or tgt_prompt

    # The training-time helper strips only DEFAULT_CROSS_ATTENTION_STOPWORDS.
    # Re-filter with the extended set so instruction verbs and the possessive
    # fragment do not end up in a segmentation query ("Put a knight's helmet on
    # him" would otherwise ask CLIPSeg for "put knight s helmet").
    keywords = [word for word in derive_keywords(src_prompt, tgt_prompt) if word not in stopwords]
    if not keywords:
        return head_noun or tgt_prompt
    if mode == "diff":
        return " ".join(keywords)

    phrase_words = list(keywords)
    if head_noun and head_noun not in phrase_words:
        phrase_words.append(head_noun)
    return " ".join(phrase_words)


class CLIPSegRegionSegmenter:
    """Soft region masks from a source image and a text phrase, via CLIPSeg.

    CLIPSeg is chosen over SAM because ``transformers`` is already a dependency
    of this project, so this adds a weight download and no new package.
    """

    def __init__(self, model_name=DEFAULT_CLIPSEG_MODEL, device="cuda"):
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self.device = device
        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name)
        self.model = self.model.to(device).eval()

    def segment(self, image, phrase):
        """Segment one [1,3,H,W] image in [0,1]; returns [1,1,H,W] probabilities.

        The returned map is at the INPUT resolution: CLIPSeg predicts at 352x352
        and the logits are bilinearly upsampled before the sigmoid.
        """
        height, width = image.shape[-2:]
        array = image[0].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0
        pil_image = Image.fromarray(array.round().astype(np.uint8))

        inputs = self.processor(
            text=[phrase],
            images=[pil_image],
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # transformers returns [352,352] for a single prompt and [B,352,352]
        # for a batch, depending on version; normalize to [1,1,h,w].
        logits = logits.float().reshape(1, 1, *logits.shape[-2:])
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        return torch.sigmoid(logits).detach().cpu()

    def close(self):
        """Release the segmenter before the LPIPS/CLIP models are loaded."""
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()


def _normalize_region_mask(
    prob,
    norm_quantile=0.95,
    binarize_threshold=None,
    min_prob_scale=REGION_MIN_PROB_SCALE,
):
    """Robustly rescale a probability map to [0,1] and keep it soft.

    Divides by a per-view quantile (p95 by default, the same convention as
    ``gradient_mask_raw_norm_quantile`` in the editing code) so one hot pixel
    cannot deflate the whole mask, then clamps. Binarization is opt-in.

    The divisor is floored at ``min_prob_scale`` (see REGION_MIN_PROB_SCALE):
    a purely relative rescaling cannot tell "the concept fills the frame" from
    "the concept is absent and every probability is noise near zero", and it
    turns the second case into an all-ones mask that silently flatters every
    method. With the floor, an absent concept stays near zero and is caught by
    the coverage check instead.
    """
    flat = prob.flatten()
    if flat.numel() > 4_000_000:
        # torch.quantile refuses very large inputs; a strided subsample is a
        # perfectly good estimate of a p95 and keeps this deterministic.
        stride = -(-flat.numel() // 4_000_000)
        flat = flat[::stride]
    quantile = float(min(max(float(norm_quantile), 0.5), 0.999))
    scale = torch.quantile(flat, quantile).clamp_min(float(max(min_prob_scale, 1e-6)))
    mask = (prob / scale).clamp(0.0, 1.0)
    if binarize_threshold is not None:
        mask = (mask >= float(binarize_threshold)).to(mask.dtype)
    return mask


def _mask_radius_pixels(mask, fraction):
    """Turn a fraction of the image diagonal into an integer pixel radius.

    Clamped to one pixel short of the frame's shorter side. The feather step
    pads with ``mode="replicate"``, which raises when the padding reaches the
    input extent, so a generous fraction on a small render would otherwise be
    a hard crash rather than a wide blur.
    """
    if fraction is None or float(fraction) <= 0.0:
        return 0
    height, width = mask.shape[-2:]
    diagonal = float((height ** 2 + width ** 2) ** 0.5)
    radius = int(max(1, round(float(fraction) * diagonal)))
    return max(0, min(radius, int(min(height, width)) - 1))


def _dilate_region_mask(mask, fraction):
    """Grow a soft mask outward by a fraction of the image diagonal.

    A max filter, which is the soft-mask generalization of a morphological
    dilation. Needed for edits that GROW geometry outside the source object:
    a knight's helmet sits above the head and stormtrooper armor extends past
    the person's silhouette, so a region segmented from the source alone would
    score that legitimate new geometry as background leakage.
    """
    radius = _mask_radius_pixels(mask, fraction)
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    # Separable: a square max filter is the horizontal max filter composed with
    # the vertical one, exactly. At a 0.03 fraction on a 1000px frame the kernel
    # is ~85 wide, so doing it as one square window would cost 85x more work per
    # pixel than it needs to.
    mask = F.max_pool2d(mask, kernel_size=(1, kernel), stride=1, padding=(0, radius))
    mask = F.max_pool2d(mask, kernel_size=(kernel, 1), stride=1, padding=(radius, 0))
    return mask.clamp(0.0, 1.0)


def _feather_region_mask(mask, fraction):
    """Blur the mask edge so the composite has no hard seam.

    ``edit_fg = R * edit + (1 - R) * src`` cuts along R. With a hard R that cut
    is a step discontinuity, and CLIP's patch embedding can read a step as
    texture. A soft R already avoids this, which is why masks stay soft by
    default; feathering exists for the binarized case.
    """
    radius = _mask_radius_pixels(mask, fraction)
    if radius <= 0:
        return mask
    sigma = max(float(radius) / 2.0, 1e-3)
    kernel_size = 2 * radius + 1
    coordinates = torch.arange(kernel_size, dtype=mask.dtype, device=mask.device) - radius
    weights = torch.exp(-(coordinates ** 2) / (2.0 * sigma ** 2))
    weights = weights / weights.sum()
    horizontal = weights.view(1, 1, 1, kernel_size)
    vertical = weights.view(1, 1, kernel_size, 1)
    blurred = F.conv2d(F.pad(mask, (radius, radius, 0, 0), mode="replicate"), horizontal)
    blurred = F.conv2d(F.pad(blurred, (0, 0, radius, radius), mode="replicate"), vertical)
    return blurred.clamp(0.0, 1.0)


def _postprocess_region_mask(mask, dilate_fraction=0.0, feather_fraction=0.0):
    """Dilate, feather, then quantize a normalized mask.

    The quantization to 1/255 is not cosmetic. The cache writes 8-bit pngs, so
    without it the run that COMPUTES a mask scores against a float R while
    every run that later loads the same png scores against a rounded R, and the
    shared-mask protocol in the module docstring would only be approximately
    fair. Rounding here makes the in-memory mask bit-identical to the one on
    disk, so computing and loading give the same numbers.
    """
    mask = _dilate_region_mask(mask, dilate_fraction)
    mask = _feather_region_mask(mask, feather_fraction)
    mask = torch.round(mask.clamp(0.0, 1.0) * 255.0) / 255.0
    return mask


def _region_coverage_ok(mask):
    """False when R covers essentially none or essentially all of the frame."""
    coverage = float(mask.mean().item())
    return REGION_DEGENERATE_MIN_COVERAGE <= coverage <= REGION_DEGENERATE_MAX_COVERAGE


def _find_region_mask_file(directory, name):
    """Locate a precomputed mask for one view under any common image suffix."""
    directory = Path(directory)
    for extension in REGION_MASK_EXTENSIONS:
        candidate = directory / f"{name}{extension}"
        if candidate.exists():
            return candidate
    return None


def _load_region_mask_file(path, size):
    """Read a grayscale mask file as a [1,1,H,W] float tensor in [0,1]."""
    array = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(array)[None, None]
    if tuple(mask.shape[-2:]) != tuple(size):
        mask = F.interpolate(mask, size=tuple(size), mode="bilinear", align_corners=False)
    return mask.clamp(0.0, 1.0)


def _save_region_mask_file(path, mask):
    """Cache one soft mask as an 8-bit grayscale png (quantized to 1/255)."""
    array = (mask[0, 0].clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def _region_cache_signature(
    phrases,
    anchor,
    clipseg_model,
    norm_quantile,
    binarize_threshold,
    min_prob_scale,
    dilate_fraction,
    feather_fraction,
):
    """Settings a cached mask was produced with; a mismatch forces a recompute.

    Every knob that changes the stored pixels is in here, so the cached png in
    ``region_masks/`` is always exactly the R that was measured against. That
    is what makes the recommended protocol sound: evaluate one run of a scene,
    then point every other run of that scene at its ``region_masks/`` with
    ``--region-mask-dir`` and the region is provably the same bytes for all of
    them, rather than merely the same by re-running a deterministic segmenter.
    """
    return {
        "phrases": list(phrases),
        "anchor": anchor,
        "clipseg_model": clipseg_model,
        "norm_quantile": float(norm_quantile),
        "binarize_threshold": None if binarize_threshold is None else float(binarize_threshold),
        "min_prob_scale": float(min_prob_scale),
        "dilate_fraction": float(dilate_fraction),
        "feather_fraction": float(feather_fraction),
        "format": "uint8_png_soft_mask_v2",
    }


def resolve_region_phrases(src_prompt, tgt_prompt, region_prompt, region_phrase_mode, region_anchor):
    """Decide which phrase(s) to segment the SOURCE frame for.

    Returns ``(phrases, detail)``. ``detail`` records what each channel
    resolved to so ``metrics.json`` can be audited later.

    The anchor choice is the single most important validity knob in this suite,
    because the region is segmented from the SOURCE image:

      - ``target``  queries the source for the phrase that names the EDIT
                    ("panda", "stormtrooper", "knight helmet"). Correct only
                    when the target concept is already visible in the source,
                    e.g. a recolour like velvet cake -> chocolate cake where
                    "cake" survives the diff-anchoring. It is the wrong query
                    for every edit that INVENTS an object, and every target
                    prompt in this project is an instruction of exactly that
                    kind, so this mode is kept for ablation, not as a default.
      - ``source``  queries the source for the thing being replaced ("bear
                    statue", "person", "face", "cake"). Always present by
                    construction.
      - ``union``   (default) takes the pixelwise max of the two. When the
                    target concept is absent its channel normalizes to ~0 (see
                    REGION_MIN_PROB_SCALE) and the union collapses to the
                    source channel automatically, so one universal rule covers
                    both regimes without a per-scene branch.

    Every input here is the prompt pair, which is shared by all methods on a
    scene, so the phrase is method-agnostic exactly like the mask it produces.
    """
    manual = (region_prompt or "").strip()
    if manual:
        return [manual], {
            "region_phrase_source": "manual",
            "region_anchor": "manual",
            "region_target_phrase": None,
            "region_source_phrase": None,
        }

    target_phrase = (derive_region_phrase(src_prompt, tgt_prompt, region_phrase_mode) or "").strip()
    source_phrase = (derive_source_region_phrase(src_prompt) or "").strip()
    detail = {
        "region_phrase_source": region_phrase_mode,
        "region_anchor": region_anchor,
        "region_target_phrase": target_phrase or None,
        "region_source_phrase": source_phrase or None,
    }

    if region_anchor == "target":
        # Fall back to the source noun rather than to nothing: a missing target
        # phrase would otherwise segment the whole target instruction.
        candidates = [target_phrase, source_phrase]
    elif region_anchor == "source":
        candidates = [source_phrase, target_phrase]
    else:  # union: both channels, source first
        candidates = [source_phrase, target_phrase]

    phrases = []
    for phrase in candidates:
        if phrase and phrase not in phrases:
            phrases.append(phrase)
        if phrases and region_anchor != "union":
            break
    return (phrases or [(tgt_prompt or "").strip()]), detail


def build_region_masks(
    gt_images,
    render_sizes,
    image_names,
    src_prompt,
    tgt_prompt,
    output_dir,
    device="cuda",
    region_mask_dir=None,
    region_prompt=None,
    region_phrase_mode="diff_anchored",
    region_anchor="union",
    norm_quantile=0.95,
    binarize_threshold=None,
    min_prob_scale=REGION_MIN_PROB_SCALE,
    dilate_fraction=0.0,
    feather_fraction=0.0,
    clipseg_model=DEFAULT_CLIPSEG_MODEL,
):
    """Build one soft region mask R in [0,1] per view, at render resolution.

    R comes from the SOURCE frames (``gt_images``) and the prompt pair only.
    ``render_sizes`` is a list of ``(H, W)`` target resolutions, NOT the edited
    renders: the edited pixels must never reach this function, or the region
    would be a function of the method being scored and the comparison would be
    circular. Passing sizes rather than tensors makes that checkable by reading
    the signature. The resolution itself is shared across methods by the
    project's fixed-downscale invariant.

    Masks are cached to ``<output_dir>/region_masks/<view>.png`` so a re-run is
    cheap and so they can be eyeballed; ``region_mask_dir`` loads precomputed
    (for example hand-drawn) masks instead and never touches CLIPSeg. Masks
    loaded that way are used verbatim, so what sits in the folder is exactly
    what was measured against.

    Returns ``(masks, info)``. ``masks`` is a list aligned with ``image_names``
    of [1,1,H,W] CPU tensors, with ``None`` for any view whose mask could not be
    produced or came out degenerate, or ``None`` outright when no mask at all
    could be produced. The caller treats a ``None`` list as "skip the localized
    metrics, finish the rest of the evaluation normally".
    """
    phrases, phrase_detail = resolve_region_phrases(
        src_prompt, tgt_prompt, region_prompt, region_phrase_mode, region_anchor
    )

    info = {
        "region_phrase": " | ".join(phrases),
        "region_phrases": list(phrases),
        "region_norm_quantile": float(norm_quantile),
        "region_binarize_threshold": (
            None if binarize_threshold is None else float(binarize_threshold)
        ),
        "region_min_prob_scale": float(min_prob_scale),
        "region_dilate_fraction": float(dilate_fraction),
        "region_feather_fraction": float(feather_fraction),
    }
    info.update(phrase_detail)
    sizes = [tuple(size) for size in render_sizes]

    # -- Precomputed masks supplied by the author --
    if region_mask_dir is not None:
        info["region_mask_source"] = f"dir:{region_mask_dir}"
        # These masks are used verbatim (see the docstring), so none of the
        # knobs that only exist inside the CLIPSeg path touched them. Blank
        # them out of the provenance block instead of echoing values that were
        # never applied: collect_metrics.py carries this block into the CSV as
        # the audit trail for "what was R", and a recorded dilate_fraction that
        # did not run would make that trail wrong.
        ignored = [
            flag
            for flag, was_set in (
                ("--region-binarize-threshold", binarize_threshold is not None),
                ("--region-dilate-frac", float(dilate_fraction) > 0.0),
                ("--region-feather-frac", float(feather_fraction) > 0.0),
                ("--region-min-prob-scale", float(min_prob_scale) != REGION_MIN_PROB_SCALE),
            )
            if was_set
        ]
        if ignored:
            print(
                f"NOTE: {', '.join(ignored)} ignored. --region-mask-dir masks are "
                "used exactly as stored, so apply those knobs when the masks are "
                "generated, not when they are read back."
            )
        for key in (
            "region_norm_quantile",
            "region_binarize_threshold",
            "region_min_prob_scale",
            "region_dilate_fraction",
            "region_feather_fraction",
        ):
            info[key] = None
        masks = []
        missing = []
        degenerate = []
        for name, size in zip(image_names, sizes):
            path = _find_region_mask_file(region_mask_dir, name)
            if path is None:
                masks.append(None)
                missing.append(name)
                continue
            try:
                mask = _load_region_mask_file(path, size)
            except Exception as exc:
                print(f"WARNING: Failed to read region mask {path}: {exc}")
                masks.append(None)
                missing.append(name)
                continue
            if not _region_coverage_ok(mask):
                masks.append(None)
                degenerate.append(name)
                continue
            masks.append(mask)
        if missing:
            print(
                f"WARNING: {len(missing)} of {len(image_names)} view(s) have no region "
                f"mask in {region_mask_dir} (first: {missing[0]}); they are excluded "
                "from the localized metrics."
            )
        if degenerate:
            print(
                f"WARNING: {len(degenerate)} region mask(s) in {region_mask_dir} cover "
                f"essentially none or essentially all of the frame (first: {degenerate[0]}); "
                "those views are excluded rather than scored against a mask that carries "
                "no partition."
            )
        if all(mask is None for mask in masks):
            print(f"WARNING: No usable region masks found in {region_mask_dir}.")
            return None, info
        info["region_mask_mean_coverage"] = float(
            np.mean([float(mask.mean().item()) for mask in masks if mask is not None])
        )
        return masks, info

    # -- CLIPSeg, with an on-disk cache keyed by the phrase and settings --
    cache_dir = Path(output_dir) / "region_masks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "region_mask_meta.json"
    signature = _region_cache_signature(
        phrases,
        info["region_anchor"],
        clipseg_model,
        norm_quantile,
        binarize_threshold,
        min_prob_scale,
        dilate_fraction,
        feather_fraction,
    )
    cached_meta = None
    if meta_path.exists():
        try:
            cached_meta = json.loads(meta_path.read_text())
        except Exception:
            cached_meta = None
    cache_valid = bool(cached_meta) and cached_meta.get("signature") == signature

    info["region_mask_source"] = "clipseg"
    info["region_mask_cache_dir"] = str(cache_dir)
    info["clipseg_model"] = clipseg_model

    masks = [None] * len(image_names)
    pending = []
    for i, name in enumerate(image_names):
        path = cache_dir / f"{name}.png"
        if cache_valid and path.exists():
            try:
                masks[i] = _load_region_mask_file(path, sizes[i])
                continue
            except Exception as exc:
                print(f"WARNING: Failed to read cached region mask {path}: {exc}")
        pending.append(i)

    degenerate = []
    if pending:
        print(
            f"Segmenting region {phrases} on {len(pending)} SOURCE view(s) "
            f"with {clipseg_model} (anchor={info['region_anchor']})..."
        )
        segmenter = None
        try:
            segmenter = CLIPSegRegionSegmenter(clipseg_model, device)
        except Exception as exc:
            print(
                f"WARNING: Could not load the CLIPSeg segmenter ({type(exc).__name__}: {exc}). "
                "The localized metrics (CLIP_direction_local, CLIP_direction_bg, "
                "Edit_Localization_Ratio, Background_LPIPS, Background_PSNR) are skipped; "
                "every other metric is unaffected. Pass --region-mask-dir to supply masks "
                "yourself, or --no-localized-metrics to silence this."
            )

        probability_means = {phrase: [] for phrase in phrases}
        if segmenter is not None:
            try:
                for i in pending:
                    try:
                        source = gt_images[i]
                        if tuple(source.shape[-2:]) != sizes[i]:
                            source = F.interpolate(
                                source, size=sizes[i], mode="bilinear", align_corners=False
                            )
                        source = source.clamp(0.0, 1.0)
                        mask = None
                        for phrase in phrases:
                            probabilities = segmenter.segment(source, phrase)
                            # Pre-normalization mean, kept as a sanity read: a
                            # channel that found nothing sits near zero here,
                            # which is how an absent target concept shows up.
                            probability_means[phrase].append(
                                float(probabilities.mean().item())
                            )
                            channel = _normalize_region_mask(
                                probabilities, norm_quantile, binarize_threshold, min_prob_scale
                            )
                            mask = channel if mask is None else torch.maximum(mask, channel)
                        mask = _postprocess_region_mask(mask, dilate_fraction, feather_fraction)
                    except Exception as exc:
                        # One bad view must not abort an evaluation run; it is
                        # dropped from the localized metrics instead.
                        print(
                            f"WARNING: Region segmentation failed for {image_names[i]} "
                            f"({type(exc).__name__}: {exc}); view excluded."
                        )
                        continue
                    if not _region_coverage_ok(mask):
                        degenerate.append(image_names[i])
                        continue
                    masks[i] = mask
                    try:
                        _save_region_mask_file(cache_dir / f"{image_names[i]}.png", mask)
                    except Exception as exc:
                        print(f"WARNING: Failed to cache region mask for {image_names[i]}: {exc}")
            finally:
                segmenter.close()
                segmenter = None
                torch.cuda.empty_cache()
            try:
                meta_path.write_text(
                    json.dumps(
                        {
                            "signature": signature,
                            "src_prompt": src_prompt,
                            "tgt_prompt": tgt_prompt,
                        },
                        indent=2,
                    )
                )
            except Exception as exc:
                print(f"WARNING: Failed to write {meta_path}: {exc}")
        recorded = {
            phrase: float(np.mean(values))
            for phrase, values in probability_means.items()
            if values
        }
        if recorded:
            info["region_prob_mean_per_phrase"] = recorded
            info["region_prob_mean"] = float(np.mean(list(recorded.values())))

    if degenerate:
        print(
            f"WARNING: {len(degenerate)} view(s) produced a region covering essentially "
            f"none or essentially all of the frame (first: {degenerate[0]}) and were "
            "excluded. An all-frame region would report perfect localization and zero "
            "leakage for every method, so it is dropped rather than scored."
        )

    if all(mask is None for mask in masks):
        print(
            "WARNING: No view produced a usable region. The phrase probably names "
            "something absent from the source frames; try --region-anchor source, or "
            "--region-prompt with the object being edited."
        )
        return None, info

    coverage = float(np.mean([float(mask.mean().item()) for mask in masks if mask is not None]))
    info["region_mask_mean_coverage"] = coverage
    print(
        f"Region masks ready (phrases={phrases}, anchor={info['region_anchor']}, "
        f"mean soft coverage {coverage:.4f})."
    )
    if coverage > 0.9 or coverage < 0.005:
        print(
            "WARNING: That coverage is degenerate (the region is nearly the whole "
            "image, or nearly empty), which usually means the phrase did not match "
            "anything in the source views. Inspect the cached masks and consider "
            "--region-prompt."
        )
    return masks, info


def region_composites(edit, source, region_mask):
    """Split an edited view so exactly one side of the region can differ.

        edit_fg = R * edit + (1 - R) * source   (only in-region change survives)
        edit_bg = (1 - R) * edit + R * source   (only out-of-region change survives)

    The split is EXACT in pixel space for any R, soft or hard:

        (edit_fg - source) + (edit_bg - source)
            = R * (edit - source) + (1 - R) * (edit - source)
            = edit - source

    so no part of the change is double counted or lost. The corresponding
    statement about CLIP_direction_local and CLIP_direction_bg is only
    approximate, because the CLIP encoder is not linear: the pair is a
    diagnostic decomposition of the global number, not an algebraic identity
    on it. Say that plainly rather than claiming an additive split.

    All tensors are [1,3,H,W] in [0,1] except ``region_mask``, which is
    [1,1,H,W] in [0,1] and broadcasts over the channels.
    """
    return (
        region_mask * edit + (1.0 - region_mask) * source,
        (1.0 - region_mask) * edit + region_mask * source,
    )


def compute_edit_localization_ratio(edit, source, region_mask):
    """Area-normalized share of the edit that landed inside the region.

        d      = |edit - source| summed over channels
        d_in   = sum(R * d) / sum(R)
        d_out  = sum((1 - R) * d) / sum(1 - R)
        ELR    = d_in / (d_in + d_out)

    Dividing by the mask areas is what keeps region SIZE from biasing the
    number: 0.5 means the change is spread uniformly over the image (no
    localization at all), values toward 1 mean it is concentrated in the
    region, values toward 0 mean it is concentrated outside it.

    Equivalently ELR = rho / (1 + rho) with rho = d_in / d_out, so ELR is a
    function of the in/out CONTRAST alone. Two consequences to state openly:

      - Area cancels. A 5% region with d_in = 0.60 against d_out = 0.02 and a
        50% region with d_in = 0.06 against d_out = 0.002 both give 0.9677.
      - Magnitude cancels too. A method that barely edits anything, but edits
        it inside R, scores near 1. ELR must therefore be read next to
        ``Edit_Magnitude_in_region``, which is why d_in and d_out are returned
        and reported rather than thrown away.

    Returns ``(ELR, d_in, d_out)``, or ``(None, None, None)`` for a degenerate
    view (an all-zero or all-one mask, or no change anywhere) so it can be
    dropped from the mean instead of reported as a fabricated 0.5.
    """
    area_in = float(region_mask.sum().item())
    area_out = float((1.0 - region_mask).sum().item())
    if area_in < REGION_MIN_AREA or area_out < REGION_MIN_AREA:
        return None, None, None

    delta = (edit - source).abs().sum(dim=1, keepdim=True)
    d_in = float((region_mask * delta).sum().item()) / area_in
    d_out = float(((1.0 - region_mask) * delta).sum().item()) / area_out
    total = d_in + d_out
    if not np.isfinite(total) or total <= 1e-12:
        return None, None, None
    return float(d_in / total), float(d_in), float(d_out)


def compute_background_psnr(edit, source, region_mask):
    """PSNR restricted to the pixels outside the region.

        mse  = sum((1 - R) * (edit - source)^2) / (3 * sum(1 - R))
        PSNR = 10 * log10(1 / mse)

    Mirrors the PIE-Bench background-preservation family. Returns ``None`` when
    the background is degenerate (an all-one mask), and caps a numerically
    perfect background at ``BACKGROUND_PSNR_CAP`` dB instead of returning inf.
    """
    weight = 1.0 - region_mask
    area_out = float(weight.sum().item())
    if area_out < REGION_MIN_AREA:
        return None

    squared_error = (edit - source) ** 2
    mse = float((weight * squared_error).sum().item()) / (3.0 * area_out)
    if not np.isfinite(mse):
        return None
    if mse <= 0.0:
        return float(BACKGROUND_PSNR_CAP)
    return float(min(BACKGROUND_PSNR_CAP, 10.0 * np.log10(1.0 / mse)))


def _clip_direction(image_feat, reference_feat, text_delta):
    """Directional CLIP similarity between an image change and the text change."""
    img_delta = image_feat - reference_feat
    if img_delta.norm() <= 1e-8 or text_delta.norm() <= 1e-8:
        return 0.0
    return float(F.cosine_similarity(img_delta, text_delta).item())


def _mean_std(values):
    """Mean/std over the views where a metric was defined; ``None`` if none were.

    Views whose region mask was degenerate or missing contribute ``None`` and
    are dropped here rather than being filled with a placeholder value.

    ``n`` records how many views survived that drop. It matters: CLIP_direction
    is a mean over EVERY view while CLIP_direction_local is a mean over the
    views that had a usable region, so the two are only comparable when ``n``
    matches ``num_views``. Without the count, a local number averaged over
    three of a hundred views would look exactly like one averaged over all of
    them. ``flatten_wandb_metrics`` ignores the field, so no WandB panel
    changes; collect_metrics.py exports it as an extra ``eval/<key>_n`` column
    alongside the existing ones.
    """
    defined = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not defined:
        return None
    return {
        "mean": float(np.mean(defined)),
        "std": float(np.std(defined)),
        "n": len(defined),
    }


def _optional_float(value):
    """JSON-friendly float that preserves an undefined per-view measurement."""
    return None if value is None else float(value)


@contextmanager
def temporary_env_var(name, value):
    """Temporarily set an environment variable for the duration of a context."""
    previous = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def load_experiment_config(config_path):
    """Load the serialized Nerfstudio config object for an experiment."""
    return yaml.load(Path(config_path).read_text(), Loader=yaml.Loader)


def infer_experiment_name(config_path, config=None):
    """Best-effort experiment name, preferring the serialized config."""
    if config is not None:
        experiment_name = getattr(config, "experiment_name", None)
        if experiment_name:
            return experiment_name

    config_path = Path(config_path)
    if len(config_path.parents) >= 3:
        return config_path.parents[2].name
    return config_path.parent.name


def infer_project_name(config=None, default="dreamcatalyst-pfc"):
    """Best-effort WandB project name, preferring the serialized config."""
    if config is not None:
        project_name = getattr(config, "project_name", None)
        if project_name:
            return project_name
    return default


def load_wandb_run_metadata(config_path):
    """Load per-run WandB metadata written during training, when available."""
    metadata_path = Path(config_path).parent / "wandb_run.json"
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text())
    except Exception as exc:
        print(f"WARNING: Failed to parse {metadata_path}: {exc}")
        return None


def extract_wandb_run_id_from_path(path):
    """Parse a WandB run id from common local directory/file naming schemes."""
    path = Path(path)
    for candidate in (str(path), path.name):
        match = WANDB_RUN_DIR_RE.search(candidate)
        if match:
            return match.group(1)
        match = WANDB_RUN_FILE_RE.search(candidate)
        if match:
            return match.group(1)
    return None


def discover_wandb_run_id(wandb_dir):
    """Best-effort local WandB run id discovery from a run storage directory."""
    if wandb_dir is None:
        return None

    wandb_dir = Path(wandb_dir)
    if not wandb_dir.exists():
        return None

    candidate_paths = []
    if wandb_dir.is_file():
        candidate_paths.append(wandb_dir)
    else:
        patterns = [
            "latest-run",
            "run-*",
            "offline-run-*",
            "**/run-*.wandb",
            "**/offline-run-*.wandb",
            "**/wandb-metadata.json",
            "**/wandb-settings.json",
        ]
        for pattern in patterns:
            candidate_paths.extend(wandb_dir.glob(pattern))

    candidate_paths = sorted(
        {path.resolve() if path.exists() else path for path in candidate_paths},
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )

    for path in candidate_paths:
        run_id = extract_wandb_run_id_from_path(path)
        if run_id:
            return run_id
        if path.suffix == ".json":
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            run_id = payload.get("run_id") or payload.get("id")
            if run_id:
                return run_id

    return None


def infer_wandb_dir(config_path):
    """Infer the method-level WandB directory used by edit.sh for this run."""
    candidate = Path(config_path).parent.parent / "wandb"
    return candidate if candidate.exists() else None


def flatten_wandb_metrics(metrics, num_views):
    """Flatten nested metrics into WandB-friendly scalar keys."""
    flattened = {"eval/num_views": num_views}
    for key, value in metrics.items():
        # A localized metric is None when no view had a usable region mask, and
        # its mean/std are None when every view was degenerate. wandb.log
        # rejects null, so those entries never leave metrics.json.
        if value is None:
            continue
        if isinstance(value, dict) and "mean" in value:
            if value["mean"] is None:
                continue
            flattened[f"eval/{key}"] = value["mean"]
            if value.get("std") is not None:
                flattened[f"eval/{key}_std"] = value["std"]
        elif isinstance(value, bool) or isinstance(value, (int, float)):
            flattened[f"eval/{key}"] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(v, (int, float)) for v in value
        ):
            # e.g. the EMV3D bbox 3-vectors -> key_0/_1/_2 scalar columns.
            for i, v in enumerate(value):
                flattened[f"eval/{key}_{i}"] = v
        # else: non-numeric values (e.g. the bbox-source string) stay in
        # metrics.json only — never sent to wandb.log, which rejects them.
    return flattened


def log_results_to_wandb(results, config_path, config, metrics_path, run_id=None, wandb_project=None):
    """Log evaluation metrics into an existing WandB run when possible."""
    try:
        import wandb
    except Exception as exc:
        print(f"WARNING: Failed to import wandb: {exc}")
        return

    project_name = wandb_project or infer_project_name(config)
    experiment_name = infer_experiment_name(config_path, config)
    flattened_metrics = flatten_wandb_metrics(results["metrics"], results["num_views"])

    run = None
    attached = False
    if run_id:
        try:
            print(f"Attaching evaluation metrics to WandB run {run_id}...")
            run = wandb.init(
                project=project_name,
                id=run_id,
                resume="must",
                job_type="evaluation",
            )
            attached = True
        except Exception as exc:
            print(f"WARNING: Failed to attach to WandB run {run_id}: {exc}")

    if run is None:
        run_name = f"{experiment_name}_eval"
        print(f"Logging evaluation metrics to a new WandB run: {run_name}")
        run = wandb.init(
            project=project_name,
            name=run_name,
            job_type="evaluation",
        )

    run.summary["eval/config"] = str(config_path)
    run.summary["eval/metrics_path"] = str(metrics_path)
    run.summary["eval/attached_to_existing_run"] = attached
    for key, value in flattened_metrics.items():
        run.summary[key] = value

    wandb.finish()


def _resize_image_like(tensor, size, mode="bilinear"):
    """Resize [1,C,H,W] tensor to size while preserving value range."""
    kwargs = {"mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(tensor, size=size, **kwargs)


def _hwc_to_bchw(tensor):
    """Convert [H,W,C] or [H,W] to [1,C,H,W]."""
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(-1)
    return tensor.permute(2, 0, 1).unsqueeze(0)


def _choose_mask_metric_size(height, width, max_side):
    """Return a smaller H,W preserving aspect ratio for geometry metrics."""
    if max_side <= 0:
        return int(height), int(width)
    scale = min(1.0, float(max_side) / float(max(height, width)))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _sample_edit_mask_points(
    rendered,
    gt,
    outputs,
    camera,
    max_side=128,
    accumulation_threshold=0.3,
):
    """Build one view's world-space samples for the 3D edit-mask variance metric.

    The metric cannot recover the training-time final gradient mask from a
    checkpoint, so it uses a rendered edit-magnitude proxy:
        M_edit = robust_norm(||I_edit - I_src||_2).
    Those mask values are backprojected with rendered depth and camera rays.
    """
    depth = outputs.get("depth", None)
    if depth is None:
        return None

    device = rendered.device
    height, width = rendered.shape[:2]
    metric_h, metric_w = _choose_mask_metric_size(height, width, max_side)

    rendered_bchw = _hwc_to_bchw(rendered.float())
    gt_bchw = _hwc_to_bchw(gt.float()).to(device)
    if gt_bchw.shape[-2:] != rendered_bchw.shape[-2:]:
        gt_bchw = _resize_image_like(gt_bchw, rendered_bchw.shape[-2:])

    edit_delta = torch.linalg.norm(rendered_bchw - gt_bchw, dim=1, keepdim=True)
    edit_delta = _resize_image_like(edit_delta, (metric_h, metric_w))
    q95 = torch.quantile(edit_delta.flatten(), 0.95).clamp_min(1e-6)
    edit_mask = (edit_delta / q95).clamp(0.0, 1.0)

    depth_bchw = _hwc_to_bchw(depth.float().to(device))
    depth_bchw = _resize_image_like(depth_bchw, (metric_h, metric_w))

    accumulation = outputs.get("accumulation", None)
    if accumulation is None:
        accumulation_bchw = torch.ones_like(depth_bchw)
    else:
        accumulation_bchw = _hwc_to_bchw(accumulation.float().to(device))
        accumulation_bchw = _resize_image_like(accumulation_bchw, (metric_h, metric_w))

    rays = camera.generate_rays(camera_indices=0, keep_shape=True)
    origins = rays.origins.to(device).float()
    directions = rays.directions.to(device).float()
    origins_bchw = _resize_image_like(_hwc_to_bchw(origins), (metric_h, metric_w))
    directions_bchw = _resize_image_like(_hwc_to_bchw(directions), (metric_h, metric_w))
    directions_bchw = F.normalize(directions_bchw, dim=1)

    points = origins_bchw + depth_bchw * directions_bchw
    valid = (
        torch.isfinite(points).all(dim=1, keepdim=True)
        & torch.isfinite(depth_bchw)
        & torch.isfinite(edit_mask)
        & (depth_bchw > 0)
        & (accumulation_bchw > float(accumulation_threshold))
    )

    return {
        "points": points.permute(0, 2, 3, 1).reshape(-1, 3).detach().cpu(),
        "values": edit_mask.reshape(-1).detach().cpu(),
        "valid": valid.reshape(-1).detach().cpu(),
    }


def compute_3d_edit_mask_variance(
    mask_samples,
    voxel_resolution=64,
    min_observations=3,
    bbox_quantile=0.05,
    bbox_inflation=0.2,
    bbox_min=None,
    bbox_max=None,
):
    """Measure per-voxel disagreement of rendered edit masks across views.

    Each view first contributes one mean edit-mask value per occupied voxel.
    The final metric is the variance of those per-view values over voxels
    observed by at least `min_observations` views.

    Two reported variants:
      - ``EditMaskVariance_3D``: the raw mean per-voxel cross-view variance.
        This is confounded by edit magnitude/sparsity — a config that edits
        harder or more bimodally scores higher purely from the wider value
        range, and a weak/uniform edit scores "better" for the wrong reason.
      - ``EditMaskVariance_3D_normalized``: the raw variance divided by the
        squared mean edit mass over eligible voxels (a squared coefficient of
        variation). This is scale-invariant in edit magnitude, so it isolates
        *relative* cross-view disagreement — the apples-to-apples consistency
        number to compare across configs.

    Cross-config comparability also requires a SHARED voxel partition. Pass an
    explicit ``bbox_min``/``bbox_max`` (each a length-3 sequence) so every
    config of a scene voxelizes on the same grid; otherwise the bbox is derived
    per-run from this run's own point cloud (``bbox_quantile``/
    ``bbox_inflation``), which means two runs are partitioned differently and
    their raw numbers are not strictly comparable. The bbox actually used is
    returned (and printed) so it can be captured from a reference run and reused
    verbatim across the other configs of the same scene.
    """
    mask_samples = [sample for sample in mask_samples if sample is not None]
    valid_points = [
        sample["points"][sample["valid"]]
        for sample in mask_samples
        if bool(sample["valid"].any())
    ]
    if not valid_points:
        return None

    all_points = torch.cat(valid_points, dim=0)
    if all_points.numel() == 0:
        return None

    if bbox_min is not None and bbox_max is not None:
        # Fixed shared partition (cross-config comparable). The caller is
        # expected to pass an already-inflated bbox — typically copied from a
        # reference run's returned EditMaskVariance_3D_bbox_min/max — so no
        # per-run inflation is applied here.
        bbox_min = torch.as_tensor(bbox_min, dtype=all_points.dtype).reshape(3)
        bbox_max = torch.as_tensor(bbox_max, dtype=all_points.dtype).reshape(3)
        bbox_source = "fixed"
    else:
        q = min(max(float(bbox_quantile), 0.0), 0.49)
        bbox_min = torch.quantile(all_points, q, dim=0)
        bbox_max = torch.quantile(all_points, 1.0 - q, dim=0)
        span = (bbox_max - bbox_min).clamp_min(1e-6)
        bbox_min = bbox_min - float(bbox_inflation) * span
        bbox_max = bbox_max + float(bbox_inflation) * span
        bbox_source = "per_run_quantile"
    span = (bbox_max - bbox_min).clamp_min(1e-6)
    print(
        f"[EMV3D] bbox_source={bbox_source} "
        f"bbox_min={[round(float(v), 5) for v in bbox_min]} "
        f"bbox_max={[round(float(v), 5) for v in bbox_max]}"
    )

    resolution = int(voxel_resolution)
    n_voxels = resolution ** 3
    obs_count = torch.zeros(n_voxels, dtype=torch.float32)
    mean = torch.zeros(n_voxels, dtype=torch.float32)
    m2 = torch.zeros(n_voxels, dtype=torch.float32)

    for sample in mask_samples:
        valid = sample["valid"]
        if not bool(valid.any()):
            continue

        points = sample["points"][valid]
        values = sample["values"][valid].float()
        coords = torch.floor((points - bbox_min) / span * resolution).long()
        in_bounds = ((coords >= 0) & (coords < resolution)).all(dim=-1)
        if not bool(in_bounds.any()):
            continue

        coords = coords[in_bounds]
        values = values[in_bounds]
        flat = coords[:, 0] * (resolution * resolution) + coords[:, 1] * resolution + coords[:, 2]

        per_voxel_sum = torch.bincount(flat, weights=values, minlength=n_voxels)
        per_voxel_count = torch.bincount(flat, minlength=n_voxels).float()
        touched = per_voxel_count > 0
        if not bool(touched.any()):
            continue

        idx = touched.nonzero(as_tuple=False).squeeze(-1)
        view_values = per_voxel_sum[idx] / per_voxel_count[idx].clamp_min(1.0)

        old_count = obs_count[idx]
        new_count = old_count + 1.0
        delta = view_values - mean[idx]
        mean[idx] = mean[idx] + delta / new_count
        delta2 = view_values - mean[idx]
        m2[idx] = m2[idx] + delta * delta2
        obs_count[idx] = new_count

    eligible = obs_count >= float(min_observations)
    if not bool(eligible.any()):
        return None

    variance = m2[eligible] / (obs_count[eligible] - 1.0).clamp_min(1.0)
    raw_mean = float(variance.mean().item())
    # Foreground edit mass over eligible voxels = the mean per-voxel edit value.
    # Normalizing the variance by its square removes the "amount of editing"
    # scaling, so the result reflects relative cross-view disagreement rather
    # than how hard the config edited. eps guards the near-zero-edit degenerate.
    mean_value = float(mean[eligible].mean().item())
    eps = 1e-6
    normalized = raw_mean / (mean_value * mean_value + eps)
    return {
        "EditMaskVariance_3D": {
            "mean": raw_mean,
            "std": float(variance.std(unbiased=False).item()),
        },
        "EditMaskVariance_3D_normalized": normalized,
        "EditMaskVariance_3D_mean_value": mean_value,
        "EditMaskVariance_3D_num_voxels": int(eligible.sum().item()),
        "EditMaskVariance_3D_mean_observations": float(obs_count[eligible].mean().item()),
        "EditMaskVariance_3D_bbox_min": [float(v) for v in bbox_min],
        "EditMaskVariance_3D_bbox_max": [float(v) for v in bbox_max],
        "EditMaskVariance_3D_bbox_source": bbox_source,
    }


def load_pipeline_from_experiment(config_path, device, disable_wandb_during_load=True):
    """Load the edited checkpoint directly from the experiment folder.

    This intentionally avoids Nerfstudio's eval_setup helper because some
    installed versions may reuse the original training load_dir and spin up a
    fresh trainer/W&B session instead of loading the edited run checkpoint.
    """
    from nerfstudio.configs.method_configs import all_methods

    config_path = Path(config_path)
    config = load_experiment_config(config_path)

    # Restore the datamanager target in the same way Nerfstudio eval_setup does.
    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target

    # Force evaluation to load the checkpoint produced by this edited run.
    checkpoint_dir = config_path.parent / "nerfstudio_models"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    config.load_dir = checkpoint_dir

    # Be extra defensive about side effects during evaluation.
    if hasattr(config, "vis"):
        config.vis = "tensorboard"

    env_value = "disabled" if disable_wandb_during_load else None
    with temporary_env_var("WANDB_MODE", env_value):
        pipeline = config.pipeline.setup(device=device, test_mode="test")
        pipeline.eval()

        checkpoint_steps = sorted(
            int(path.stem.split("-")[1]) for path in checkpoint_dir.glob("step-*.ckpt")
        )
        if not checkpoint_steps:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

        load_step = checkpoint_steps[-1]
        load_path = checkpoint_dir / f"step-{load_step:09d}.ckpt"
        loaded_state = torch.load(load_path, map_location="cpu")
        pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
    return pipeline, load_path, load_step, config


def render_all_views(
    config_path,
    device,
    disable_wandb_during_load=True,
    compute_mask_variance=True,
    mask_variance_image_max_side=128,
    mask_variance_voxel_resolution=64,
    mask_variance_min_observations=3,
    mask_variance_accumulation_threshold=0.3,
    mask_variance_bbox_min=None,
    mask_variance_bbox_max=None,
):
    """Load an edited nerfstudio checkpoint and render all training views.
    Returns list of (rendered_image_tensor, gt_image_tensor) pairs.
    rendered images are [1,C,H,W] in [0,1].
    """
    pipeline, checkpoint_path, load_step, config = load_pipeline_from_experiment(
        config_path,
        device,
        disable_wandb_during_load=disable_wandb_during_load,
    )
    print(f"Loaded edited checkpoint from {checkpoint_path} (step {load_step}).")

    rendered_images = []
    gt_images = []
    image_names = []
    mask_variance_samples = []

    dataset = pipeline.datamanager.train_dataset

    for i in range(len(dataset)):
        camera = dataset.cameras[i : i + 1].to(device)

        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera)

        # Rendered image: [H, W, 3]
        rendered = outputs["rgb"].cpu()
        # GT image: [H, W, 3]
        gt = dataset[i]["image"].cpu()

        # Convert to [1, C, H, W]
        rendered_tensor = rendered.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
        gt_tensor = gt.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)

        if compute_mask_variance:
            sample = _sample_edit_mask_points(
                outputs["rgb"].detach(),
                dataset[i]["image"].to(device),
                outputs,
                camera,
                max_side=mask_variance_image_max_side,
                accumulation_threshold=mask_variance_accumulation_threshold,
            )
            mask_variance_samples.append(sample)

        rendered_images.append(rendered_tensor)
        gt_images.append(gt_tensor)

        fname = Path(dataset.image_filenames[i]).stem if hasattr(dataset, "image_filenames") else f"view_{i:04d}"
        image_names.append(fname)

    mask_variance_metrics = None
    if compute_mask_variance:
        print("Computing 3D edit-mask variance...")
        mask_variance_metrics = compute_3d_edit_mask_variance(
            mask_variance_samples,
            voxel_resolution=mask_variance_voxel_resolution,
            min_observations=mask_variance_min_observations,
            bbox_min=mask_variance_bbox_min,
            bbox_max=mask_variance_bbox_max,
        )
        if mask_variance_metrics is None:
            print("WARNING: 3D edit-mask variance unavailable; depth/ray samples were insufficient.")

    return rendered_images, gt_images, image_names, config, mask_variance_metrics


def evaluate_experiment(
    config_path,
    src_prompt,
    tgt_prompt,
    output_dir,
    device="cuda",
    log_wandb=False,
    wandb_run_id=None,
    wandb_dir=None,
    wandb_project=None,
    compute_mask_variance=True,
    mask_variance_image_max_side=128,
    mask_variance_voxel_resolution=64,
    mask_variance_min_observations=3,
    mask_variance_accumulation_threshold=0.3,
    mask_variance_bbox_min=None,
    mask_variance_bbox_max=None,
    compute_localized_metrics=True,
    region_mask_dir=None,
    region_prompt=None,
    region_phrase_mode="diff_anchored",
    region_anchor="union",
    region_norm_quantile=0.95,
    region_binarize_threshold=None,
    region_min_prob_scale=REGION_MIN_PROB_SCALE,
    region_dilate_fraction=0.0,
    region_feather_fraction=0.0,
    clipseg_model=DEFAULT_CLIPSEG_MODEL,
):
    """Run full evaluation on a single experiment."""
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rendered").mkdir(exist_ok=True)

    print(f"Loading checkpoint from {config_path}...")
    rendered_images, gt_images, image_names, config, mask_variance_metrics = render_all_views(
        config_path,
        device,
        disable_wandb_during_load=True,
        compute_mask_variance=compute_mask_variance,
        mask_variance_image_max_side=mask_variance_image_max_side,
        mask_variance_voxel_resolution=mask_variance_voxel_resolution,
        mask_variance_min_observations=mask_variance_min_observations,
        mask_variance_accumulation_threshold=mask_variance_accumulation_threshold,
        mask_variance_bbox_min=mask_variance_bbox_min,
        mask_variance_bbox_max=mask_variance_bbox_max,
    )
    num_views = len(rendered_images)
    print(f"Rendered {num_views} views.")

    # Every per-view artefact is keyed by the image stem: results["per_view"],
    # rendered/<name>.png and region_masks/<name>.png. Two dataset images that
    # share a stem therefore overwrite each other's mask and collapse into one
    # per_view entry, which is silent data loss rather than an error.
    if len(set(image_names)) != len(image_names):
        duplicates = sorted({n for n in image_names if image_names.count(n) > 1})
        print(
            f"WARNING: {len(duplicates)} view name(s) are not unique (first: "
            f"{duplicates[0]}). Per-view entries, rendered pngs and region masks "
            "are keyed by name, so duplicates overwrite each other. The aggregate "
            "means are still over all views."
        )

    # Save rendered images
    for name, rendered in zip(image_names, rendered_images):
        img = (rendered.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(output_dir / f"rendered/{name}.png")

    # ── Region masks (source-derived, so identical across methods) ──
    # Built before the LPIPS/CLIP models are loaded so the segmenter's memory is
    # released before they arrive.
    region_masks = None
    region_info = None
    if compute_localized_metrics:
        print("Building region masks for the localized metrics...")
        region_masks, region_info = build_region_masks(
            gt_images,
            # Render RESOLUTIONS only. The edited pixels are deliberately not
            # passed: the region must be a function of the source frames and
            # the prompts alone, or it would favour the method it was built
            # from.
            [tuple(rendered.shape[-2:]) for rendered in rendered_images],
            image_names,
            src_prompt,
            tgt_prompt,
            output_dir,
            device=device,
            region_mask_dir=region_mask_dir,
            region_prompt=region_prompt,
            region_phrase_mode=region_phrase_mode,
            region_anchor=region_anchor,
            norm_quantile=region_norm_quantile,
            binarize_threshold=region_binarize_threshold,
            min_prob_scale=region_min_prob_scale,
            dilate_fraction=region_dilate_fraction,
            feather_fraction=region_feather_fraction,
            clipseg_model=clipseg_model,
        )
        if region_masks is None:
            print("WARNING: No region masks available; localized metrics are skipped.")

    # ── LPIPS ──
    print("Computing LPIPS...")
    import lpips as lpips_lib
    lpips_model = lpips_lib.LPIPS(net="vgg").to(device)
    lpips_scores = []
    bg_lpips_scores = []
    bg_psnr_scores = []
    elr_scores = []
    edit_magnitude_in = []
    edit_magnitude_out = []
    for i, (rendered, gt) in enumerate(zip(rendered_images, gt_images)):
        # Resize gt to match rendered if needed
        if gt.shape[2:] != rendered.shape[2:]:
            gt = F.interpolate(gt, size=rendered.shape[2:], mode="bilinear", align_corners=False)
        score = compute_lpips(rendered.to(device), gt.to(device), lpips_model)
        lpips_scores.append(score)

        # Everything below is measured against the region mask, at render
        # resolution, on the already-resized source frame.
        region_mask = region_masks[i] if region_masks is not None else None
        if region_mask is None:
            bg_lpips_scores.append(None)
            bg_psnr_scores.append(None)
            elr_scores.append(None)
            edit_magnitude_in.append(None)
            edit_magnitude_out.append(None)
            continue

        _, edit_bg = region_composites(rendered, gt, region_mask)
        if float((1.0 - region_mask).sum().item()) < REGION_MIN_AREA:
            # An all-one mask leaves no background to preserve.
            bg_lpips_scores.append(None)
        else:
            bg_lpips_scores.append(
                compute_lpips(edit_bg.to(device), gt.to(device), lpips_model)
            )
        bg_psnr_scores.append(compute_background_psnr(rendered, gt, region_mask))
        elr, d_in, d_out = compute_edit_localization_ratio(rendered, gt, region_mask)
        elr_scores.append(elr)
        edit_magnitude_in.append(d_in)
        edit_magnitude_out.append(d_out)
    del lpips_model
    torch.cuda.empty_cache()

    # ── SSIM ──
    print("Computing SSIM...")
    ssim_scores = []
    for rendered, gt in zip(rendered_images, gt_images):
        r_np = rendered.squeeze(0).permute(1, 2, 0).numpy()
        if gt.shape[2:] != rendered.shape[2:]:
            gt = F.interpolate(gt, size=rendered.shape[2:], mode="bilinear", align_corners=False)
        g_np = gt.squeeze(0).permute(1, 2, 0).numpy()
        ssim_scores.append(compute_ssim(r_np, g_np))

    # ── CLIP metrics ──
    print("Computing CLIP metrics...")
    clip_model = load_clip_model(device)

    # Encode text prompts
    src_text_feat = clip_encode_text(clip_model, [src_prompt], device)
    tgt_text_feat = clip_encode_text(clip_model, [tgt_prompt], device)

    clip_text_sims = []      # edited image vs target text
    clip_directions = []     # directional similarity (whole image)
    clip_directions_local = []  # directional similarity, in-region change only
    clip_directions_bg = []     # directional similarity, out-of-region change only
    clip_img_sims = []       # original vs edited image similarity
    edited_features_all = [] # for multi-view consistency

    text_delta = tgt_text_feat - src_text_feat

    for i, (rendered, gt) in enumerate(zip(rendered_images, gt_images)):
        edited_feat = clip_encode_image(clip_model, rendered.to(device), device)
        if gt.shape[2:] != rendered.shape[2:]:
            gt = F.interpolate(gt, size=rendered.shape[2:], mode="bilinear", align_corners=False)
        orig_feat = clip_encode_image(clip_model, gt.to(device), device)

        # Text similarity: edited image vs target prompt
        clip_text_sims.append(
            F.cosine_similarity(edited_feat, tgt_text_feat).item()
        )

        # Directional similarity: (img_edit - img_orig) vs (text_tgt - text_src)
        clip_directions.append(_clip_direction(edited_feat, orig_feat, text_delta))

        # Same quantity, but on the two composites that isolate one side of the
        # region: edit_fg keeps only the in-region change, edit_bg only the
        # out-of-region change. Their sum of evidence explains the global
        # number above.
        region_mask = region_masks[i] if region_masks is not None else None
        if region_mask is None:
            clip_directions_local.append(None)
            clip_directions_bg.append(None)
        else:
            edit_fg, edit_bg = region_composites(rendered, gt, region_mask)
            fg_feat = clip_encode_image(clip_model, edit_fg.to(device), device)
            bg_feat = clip_encode_image(clip_model, edit_bg.to(device), device)
            clip_directions_local.append(_clip_direction(fg_feat, orig_feat, text_delta))
            clip_directions_bg.append(_clip_direction(bg_feat, orig_feat, text_delta))

        # Image similarity: original vs edited (identity preservation)
        clip_img_sims.append(
            F.cosine_similarity(orig_feat, edited_feat).item()
        )

        edited_features_all.append(edited_feat)

    # ── Multi-view consistency ──
    # Measure how consistent the CLIP embeddings are across views.
    # Lower std = more consistent editing across views.
    all_feats = torch.cat(edited_features_all, dim=0)  # [N, D]
    mv_consistency_std = all_feats.std(dim=0).mean().item()

    # Also compute pairwise cosine similarity mean
    cos_sim_matrix = F.cosine_similarity(
        all_feats.unsqueeze(0), all_feats.unsqueeze(1), dim=2
    )
    # Exclude diagonal
    mask = ~torch.eye(num_views, dtype=torch.bool, device=device)
    mv_pairwise_mean = cos_sim_matrix[mask].mean().item()

    del clip_model
    torch.cuda.empty_cache()

    # ── Aggregate results ──
    results = {
        "config": str(config_path),
        "src_prompt": src_prompt,
        "tgt_prompt": tgt_prompt,
        "num_views": num_views,
        "metrics": {
            "CLIP_text_sim": {
                "mean": float(np.mean(clip_text_sims)),
                "std": float(np.std(clip_text_sims)),
            },
            "CLIP_direction": {
                "mean": float(np.mean(clip_directions)),
                "std": float(np.std(clip_directions)),
            },
            "CLIP_img_sim": {
                "mean": float(np.mean(clip_img_sims)),
                "std": float(np.std(clip_img_sims)),
            },
            "SSIM": {
                "mean": float(np.mean(ssim_scores)),
                "std": float(np.std(ssim_scores)),
            },
            "LPIPS": {
                "mean": float(np.mean(lpips_scores)),
                "std": float(np.std(lpips_scores)),
            },
            "MultiView_consistency_std": float(mv_consistency_std),
            "MultiView_pairwise_cos_sim": float(mv_pairwise_mean),
        },
        "per_view": {
            name: {
                "CLIP_text_sim": float(clip_text_sims[i]),
                "CLIP_direction": float(clip_directions[i]),
                "CLIP_img_sim": float(clip_img_sims[i]),
                "SSIM": float(ssim_scores[i]),
                "LPIPS": float(lpips_scores[i]),
            }
            for i, name in enumerate(image_names)
        },
    }
    if mask_variance_metrics is not None:
        results["metrics"].update(mask_variance_metrics)

    # ── Region-localized metrics ──
    # Entries whose every view was degenerate stay out of `metrics` entirely
    # rather than being reported as a null, so the old metric set is byte-for-
    # byte what a run without a segmenter produces.
    if region_masks is not None:
        localized_entries = {
            "CLIP_direction_local": _mean_std(clip_directions_local),
            "CLIP_direction_bg": _mean_std(clip_directions_bg),
            "Edit_Localization_Ratio": _mean_std(elr_scores),
            "Background_LPIPS": _mean_std(bg_lpips_scores),
            "Background_PSNR": _mean_std(bg_psnr_scores),
            # Companions, not scores. CLIP_direction_local is a cosine and
            # Edit_Localization_Ratio is a contrast, so both are invariant to
            # the SIZE of the change: a method that edits almost nothing, in a
            # lucky direction, inside R can match a method that edits properly.
            # These two say how much actually moved, and make that check
            # mechanical instead of a matter of trust.
            "Edit_Magnitude_in_region": _mean_std(edit_magnitude_in),
            "Edit_Magnitude_out_region": _mean_std(edit_magnitude_out),
        }
        for key, value in localized_entries.items():
            if value is not None:
                results["metrics"][key] = value
        results["metrics"]["Region_valid_views"] = int(
            sum(rm is not None for rm in region_masks)
        )
        if region_info is not None and "region_mask_mean_coverage" in region_info:
            results["metrics"]["Region_mask_mean_coverage"] = float(
                region_info["region_mask_mean_coverage"]
            )
        for i, name in enumerate(image_names):
            results["per_view"][name].update(
                {
                    "CLIP_direction_local": _optional_float(clip_directions_local[i]),
                    "CLIP_direction_bg": _optional_float(clip_directions_bg[i]),
                    "Edit_Localization_Ratio": _optional_float(elr_scores[i]),
                    "Background_LPIPS": _optional_float(bg_lpips_scores[i]),
                    "Background_PSNR": _optional_float(bg_psnr_scores[i]),
                    "Edit_Magnitude_in_region": _optional_float(edit_magnitude_in[i]),
                    "Edit_Magnitude_out_region": _optional_float(edit_magnitude_out[i]),
                }
            )
    if region_info is not None:
        results["region"] = region_info

    # Save
    results_path = output_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── W&B Logging ──
    if log_wandb:
        metadata = load_wandb_run_metadata(config_path) or {}
        resolved_wandb_dir = Path(wandb_dir) if wandb_dir else infer_wandb_dir(config_path)
        resolved_run_id = wandb_run_id or metadata.get("run_id") or discover_wandb_run_id(resolved_wandb_dir)
        resolved_project_name = wandb_project or metadata.get("project") or infer_project_name(config)
        log_results_to_wandb(
            results,
            config_path,
            config,
            results_path,
            run_id=resolved_run_id,
            wandb_project=resolved_project_name,
        )

    # Print summary
    print("\n" + "=" * 60)
    print(f"  Evaluation: {infer_experiment_name(config_path, config)}")
    print("=" * 60)
    m = results["metrics"]
    print(f"  CLIP text sim (edit quality):  {m['CLIP_text_sim']['mean']:.4f} +/- {m['CLIP_text_sim']['std']:.4f}")
    print(f"  CLIP direction (edit faithf.): {m['CLIP_direction']['mean']:.4f} +/- {m['CLIP_direction']['std']:.4f}")
    print(f"  CLIP img sim (identity):       {m['CLIP_img_sim']['mean']:.4f} +/- {m['CLIP_img_sim']['std']:.4f}")
    print(f"  SSIM (identity):               {m['SSIM']['mean']:.4f} +/- {m['SSIM']['std']:.4f}")
    print(f"  LPIPS (perceptual dist):       {m['LPIPS']['mean']:.4f} +/- {m['LPIPS']['std']:.4f}")
    print(f"  MV consistency (feat std):     {m['MultiView_consistency_std']:.6f}")
    print(f"  MV pairwise cos sim:           {m['MultiView_pairwise_cos_sim']:.4f}")
    if region_info is not None:
        print("-" * 60)
        print(f"  Region phrase ({region_info.get('region_phrase_source', 'n/a')}, "
              f"anchor={region_info.get('region_anchor', 'n/a')}): "
              f"'{region_info.get('region_phrase', '')}'")
        if "Region_mask_mean_coverage" in m:
            print(f"  Region mean soft coverage:     {m['Region_mask_mean_coverage']:.4f}")
        if "Region_valid_views" in m:
            print(f"  Region valid views:            {m['Region_valid_views']} / {num_views}")
    if "CLIP_direction_local" in m:
        print(
            f"  CLIP dir LOCAL (in region):    {m['CLIP_direction_local']['mean']:.4f} "
            f"+/- {m['CLIP_direction_local']['std']:.4f}"
        )
    if "CLIP_direction_bg" in m:
        print(
            f"  CLIP dir BG (leak, ~0 is good):{m['CLIP_direction_bg']['mean']:.4f} "
            f"+/- {m['CLIP_direction_bg']['std']:.4f}"
        )
    if "Edit_Localization_Ratio" in m:
        print(
            f"  Edit localization ratio:       {m['Edit_Localization_Ratio']['mean']:.4f} "
            f"+/- {m['Edit_Localization_Ratio']['std']:.4f} (0.5 = uniform)"
        )
    if "Background_LPIPS" in m:
        print(
            f"  Background LPIPS (lower=bett): {m['Background_LPIPS']['mean']:.4f} "
            f"+/- {m['Background_LPIPS']['std']:.4f}"
        )
    if "Background_PSNR" in m:
        print(
            f"  Background PSNR (higher=bett): {m['Background_PSNR']['mean']:.2f} "
            f"+/- {m['Background_PSNR']['std']:.2f} dB"
        )
    if "Edit_Magnitude_in_region" in m and "Edit_Magnitude_out_region" in m:
        print(
            f"  Edit magnitude in / out of R:  "
            f"{m['Edit_Magnitude_in_region']['mean']:.4f} / "
            f"{m['Edit_Magnitude_out_region']['mean']:.4f} "
            "(sanity: a high local score with a near-zero in-region magnitude "
            "means nothing was really edited)"
        )
    if "EditMaskVariance_3D" in m:
        print(
            f"  3D edit-mask variance:         {m['EditMaskVariance_3D']['mean']:.6f} "
            f"+/- {m['EditMaskVariance_3D']['std']:.6f}"
        )
        if "EditMaskVariance_3D_normalized" in m:
            print(
                f"  3D edit-mask variance (norm):  {m['EditMaskVariance_3D_normalized']:.6f} "
                f"(mean edit mass {m['EditMaskVariance_3D_mean_value']:.4f})"
            )
        print(f"  3D edit-mask voxels:           {m['EditMaskVariance_3D_num_voxels']}")
        if "EditMaskVariance_3D_bbox_source" in m:
            print(f"  3D edit-mask bbox source:      {m['EditMaskVariance_3D_bbox_source']}")
    print("=" * 60)
    print(f"  Results saved to: {results_path}")

    return results
def main():
    parser = argparse.ArgumentParser(description="Evaluate DreamCatalyst editing experiments")
    subparsers = parser.add_subparsers(dest="command")

    # Evaluate a single experiment
    eval_parser = subparsers.add_parser("eval", help="Evaluate a single experiment")
    eval_parser.add_argument("--config", type=str, required=True, help="Path to config.yml")
    eval_parser.add_argument("--src-prompt", type=str, required=True, help="Source prompt")
    eval_parser.add_argument("--tgt-prompt", type=str, required=True, help="Target prompt")
    eval_parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    eval_parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    eval_parser.add_argument("--log-wandb", action="store_true", help="Log evaluation metrics to Weights & Biases")
    eval_parser.add_argument("--wandb-run-id", type=str, default=None, help="Attach metrics to an existing WandB run id")
    eval_parser.add_argument("--wandb-dir", type=str, default=None, help="Directory containing local WandB run files")
    eval_parser.add_argument("--wandb-project", type=str, default=None, help="Override the WandB project name")
    eval_parser.add_argument(
        "--disable-mask-variance",
        action="store_true",
        help="Disable the 3D edit-mask variance metric",
    )
    eval_parser.add_argument(
        "--mask-variance-image-max-side",
        type=int,
        default=128,
        help="Max rendered-image side used for 3D edit-mask variance samples",
    )
    eval_parser.add_argument(
        "--mask-variance-voxel-resolution",
        type=int,
        default=64,
        help="Voxel grid resolution used by the 3D edit-mask variance metric",
    )
    eval_parser.add_argument(
        "--mask-variance-min-observations",
        type=int,
        default=3,
        help="Minimum number of views observing a voxel for 3D edit-mask variance",
    )
    eval_parser.add_argument(
        "--mask-variance-accumulation-threshold",
        type=float,
        default=0.3,
        help="Rendered accumulation threshold for valid 3D edit-mask variance samples",
    )
    eval_parser.add_argument(
        "--mask-variance-bbox-min",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Fixed shared voxel-grid min corner for EMV3D (3 floats). Pass the "
             "SAME value across all configs of a scene for cross-config "
             "comparable EMV3D; copy it from a reference run's "
             "EditMaskVariance_3D_bbox_min in metrics.json. If omitted, the "
             "bbox is derived per-run (NOT comparable across configs).",
    )
    eval_parser.add_argument(
        "--mask-variance-bbox-max",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Fixed shared voxel-grid max corner for EMV3D (3 floats). "
             "See --mask-variance-bbox-min.",
    )
    eval_parser.add_argument(
        "--no-localized-metrics",
        action="store_true",
        help="Disable the region-localized metrics (CLIP_direction_local, "
             "CLIP_direction_bg, Edit_Localization_Ratio, Background_LPIPS, "
             "Background_PSNR). Everything else is unaffected.",
    )
    eval_parser.add_argument(
        "--region-mask-dir",
        type=str,
        default=None,
        help="Load precomputed region masks from this directory instead of "
             "running CLIPSeg. One grayscale image per view, named <view>.png "
             "(jpg/bmp/tif also accepted); white = inside the region. Use this "
             "for hand-drawn regions or when CLIPSeg is unavailable.",
    )
    eval_parser.add_argument(
        "--region-prompt",
        type=str,
        default=None,
        help="Manual region phrase for the segmenter, overriding the phrase "
             "auto-derived from the src/tgt prompt diff (e.g. \"cake\").",
    )
    eval_parser.add_argument(
        "--region-phrase-mode",
        type=str,
        default="diff_anchored",
        choices=list(REGION_PHRASE_MODES),
        help="How to derive the region phrase from the prompts when "
             "--region-prompt is not given. 'diff_anchored' (default) is the "
             "prompt diff plus the target head noun; 'diff' is the bare prompt "
             "diff; 'head_noun' and 'target_prompt' are the two fallbacks.",
    )
    eval_parser.add_argument(
        "--region-anchor",
        type=str,
        default="union",
        choices=list(REGION_ANCHOR_MODES),
        help="WHICH phrase the SOURCE frame is segmented for. 'union' "
             "(default) takes the max of the source-noun mask and the "
             "target-phrase mask; 'source' segments only the thing being "
             "replaced (bear statue, person, face, cake), which is the mode "
             "that works when the edit invents an object that is not in the "
             "source; 'target' segments only the edit phrase, which is correct "
             "only when the target concept is already visible in the source. "
             "All three read the source image, never the edited render.",
    )
    eval_parser.add_argument(
        "--region-binarize-threshold",
        type=float,
        default=None,
        help="Threshold the normalized region mask into a hard 0/1 mask. "
             "Omitted by default: the mask stays soft. If you do binarize, "
             "pass --region-feather-frac as well, or the composite is cut "
             "along a step edge that CLIP can read as texture.",
    )
    eval_parser.add_argument(
        "--region-min-prob-scale",
        type=float,
        default=REGION_MIN_PROB_SCALE,
        help="Absolute floor on the divisor used to rescale the CLIPSeg "
             f"probability map (default {REGION_MIN_PROB_SCALE}). Without it, a "
             "map that found nothing is divided by its own near-zero quantile "
             "and becomes an all-ones region, which reports perfect "
             "localization for every method. Set to 0 only to reproduce that "
             "older, unsafe behaviour.",
    )
    eval_parser.add_argument(
        "--region-dilate-frac",
        type=float,
        default=0.0,
        help="Grow the region outward by this fraction of the image diagonal "
             "(default 0 = off). Needed when the edit GROWS geometry outside "
             "the source object, e.g. a knight's helmet above the head or "
             "stormtrooper armor past the person's silhouette; without it that "
             "legitimate new geometry is scored as background leakage. Around "
             "0.02-0.03 is a reasonable starting point. Use the SAME value for "
             "every method compared on a scene.",
    )
    eval_parser.add_argument(
        "--region-feather-frac",
        type=float,
        default=0.0,
        help="Blur the region edge by this fraction of the image diagonal "
             "(default 0 = off, which is safe because the mask is soft "
             "already). Use it together with --region-binarize-threshold.",
    )
    eval_parser.add_argument(
        "--region-norm-quantile",
        type=float,
        default=0.95,
        help="Robust per-view quantile used to rescale the region mask to "
             "[0,1] (default 0.95, matching gradient_mask_raw_norm_quantile).",
    )
    eval_parser.add_argument(
        "--clipseg-model",
        type=str,
        default=DEFAULT_CLIPSEG_MODEL,
        help="HuggingFace CLIPSeg checkpoint used to segment the region.",
    )

    args = parser.parse_args()

    if args.command == "eval":
        evaluate_experiment(
            args.config,
            args.src_prompt,
            args.tgt_prompt,
            args.output_dir,
            args.device,
            log_wandb=args.log_wandb,
            wandb_run_id=args.wandb_run_id,
            wandb_dir=args.wandb_dir,
            wandb_project=args.wandb_project,
            compute_mask_variance=not args.disable_mask_variance,
            mask_variance_image_max_side=args.mask_variance_image_max_side,
            mask_variance_voxel_resolution=args.mask_variance_voxel_resolution,
            mask_variance_min_observations=args.mask_variance_min_observations,
            mask_variance_accumulation_threshold=args.mask_variance_accumulation_threshold,
            mask_variance_bbox_min=args.mask_variance_bbox_min,
            mask_variance_bbox_max=args.mask_variance_bbox_max,
            compute_localized_metrics=not args.no_localized_metrics,
            region_mask_dir=args.region_mask_dir,
            region_prompt=args.region_prompt,
            region_phrase_mode=args.region_phrase_mode,
            region_anchor=args.region_anchor,
            region_norm_quantile=args.region_norm_quantile,
            region_binarize_threshold=args.region_binarize_threshold,
            region_min_prob_scale=args.region_min_prob_scale,
            region_dilate_fraction=args.region_dilate_frac,
            region_feather_fraction=args.region_feather_frac,
            clipseg_model=args.clipseg_model,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
