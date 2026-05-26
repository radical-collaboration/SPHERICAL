#!/usr/bin/env python3
"""
Generate SPHERICAL benchmark PowerPoint presentation.

Usage:
    python make_presentation.py [--out spherical_benchmark.pptx]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Colour constants ──────────────────────────────────────────────────────────

WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
BLACK       = RGBColor(0x00, 0x00, 0x00)
DARK_BG     = RGBColor(0x1A, 0x1A, 0x2E)
ACCENT      = RGBColor(0x0F, 0x3C, 0x78)
LIGHT_PANEL = RGBColor(0xF0, 0xF4, 0xFF)
GRAY_TEXT   = RGBColor(0x55, 0x55, 0x55)
BASELINE_C  = RGBColor(0x9E, 0x9E, 0x9E)
SHARD_C     = RGBColor(0x4C, 0xAF, 0x50)
BANDIT_C    = RGBColor(0x9C, 0x27, 0xB0)
ALLOPT_C    = RGBColor(0xF4, 0x43, 0x36)
HIGHLIGHT   = RGBColor(0xFF, 0xC1, 0x07)
TEAL_C      = RGBColor(0x00, 0x83, 0x8F)
ORANGE_C    = RGBColor(0xE6, 0x51, 0x00)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _bg(slide, color: RGBColor):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def _box(slide, l, t, w, h, text="", font_size=18, bold=False,
         color=WHITE, bg=None, align=PP_ALIGN.LEFT,
         font_name="Calibri", italic=False, wrap=True):
    txBox = slide.shapes.add_textbox(l, t, w, h)
    tf    = txBox.text_frame
    tf.word_wrap = wrap
    p  = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size      = Pt(font_size)
    run.font.bold      = bold
    run.font.italic    = italic
    run.font.color.rgb = color
    run.font.name      = font_name
    if bg is not None:
        txBox.fill.solid()
        txBox.fill.fore_color.rgb = bg
    return txBox


def _rect(slide, l, t, w, h, fill_color: RGBColor, line_color=None, line_width=0):
    shape = slide.shapes.add_shape(1, l, t, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width     = Pt(line_width)
    else:
        shape.line.fill.background()
    return shape


def _img(slide, path, l, t, w, h=None):
    if h is not None:
        slide.shapes.add_picture(str(path), l, t, w, h)
    else:
        slide.shapes.add_picture(str(path), l, t, w)


def _title_bar(slide, title: str, subtitle: str = ""):
    _rect(slide, Inches(0), Inches(0), SLIDE_W, Inches(1.1), ACCENT)
    _box(slide, Inches(0.25), Inches(0.08), Inches(12.5), Inches(0.6),
         title, font_size=28, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        _box(slide, Inches(0.25), Inches(0.65), Inches(12.5), Inches(0.35),
             subtitle, font_size=13, color=RGBColor(0xBB, 0xCC, 0xFF),
             align=PP_ALIGN.LEFT)


def _stat_box(slide, l, t, w, h, value, label, val_color=HIGHLIGHT,
              lbl_color=WHITE, bg=ACCENT):
    _rect(slide, l, t, w, h, bg)
    _box(slide, l, t + Inches(0.05), w, Inches(0.55),
         value, font_size=36, bold=True, color=val_color, align=PP_ALIGN.CENTER)
    _box(slide, l, t + Inches(0.6), w, Inches(0.35),
         label, font_size=11, color=lbl_color, align=PP_ALIGN.CENTER)


def _section_panel(slide, l, t, w, h, title, bullets, title_bg, title_color=WHITE,
                   body_bg=None, bullet_color=None, title_size=11, bullet_size=10):
    """Titled panel with bullet items."""
    if body_bg is None:
        body_bg = RGBColor(0xF8, 0xF8, 0xF8)
    if bullet_color is None:
        bullet_color = BLACK
    _rect(slide, l, t, w, Inches(0.32), title_bg)
    _box(slide, l + Inches(0.06), t + Inches(0.02), w - Inches(0.12), Inches(0.30),
         title, font_size=title_size, bold=True, color=title_color)
    _rect(slide, l, t + Inches(0.32), w, h - Inches(0.32), body_bg)
    y = t + Inches(0.36)
    per = (h - Inches(0.40)) / max(len(bullets), 1)
    for b in bullets:
        _box(slide, l + Inches(0.1), y, w - Inches(0.15), per,
             f"• {b}", font_size=bullet_size, color=bullet_color)
        y += per


# ── Slide builders ────────────────────────────────────────────────────────────

PLOT_DIR = Path(__file__).parent / "plots" / "optimizations"


def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, DARK_BG)
    _rect(slide, Inches(0), Inches(0), SLIDE_W, Inches(2.8), ACCENT)
    _box(slide, Inches(0.5), Inches(0.3), Inches(12), Inches(0.7),
         "SPHERICAL", font_size=18, bold=True,
         color=RGBColor(0xBB, 0xCC, 0xFF), align=PP_ALIGN.LEFT)
    _box(slide, Inches(0.5), Inches(0.85), Inches(12), Inches(1.1),
         "Adaptive HPC Campaign Optimisation", font_size=40, bold=True,
         color=WHITE, align=PP_ALIGN.LEFT)
    _box(slide, Inches(0.5), Inches(1.9), Inches(12), Inches(0.6),
         "Benchmark Results: 4-Configuration Drug-Discovery Pipeline Study",
         font_size=18, color=RGBColor(0xBB, 0xCC, 0xFF), align=PP_ALIGN.LEFT)

    _rect(slide, Inches(0.5), Inches(3.1), Inches(5.5), Inches(2.0),
          RGBColor(0x0A, 0x2A, 0x52))
    _box(slide, Inches(0.6), Inches(3.2), Inches(5.2), Inches(0.5),
         "Key Result", font_size=14, bold=True, color=HIGHLIGHT, align=PP_ALIGN.LEFT)
    _box(slide, Inches(0.6), Inches(3.6), Inches(5.2), Inches(0.8),
         "10.9× faster time-to-target", font_size=28, bold=True,
         color=WHITE, align=PP_ALIGN.LEFT)
    _box(slide, Inches(0.6), Inches(4.25), Inches(5.2), Inches(0.7),
         "82.6 s  →  7.6 s  to find 5 drug leads\nfrom 10,000 candidate ligands",
         font_size=13, color=RGBColor(0xBB, 0xCC, 0xFF), align=PP_ALIGN.LEFT)

    for i, (label, col) in enumerate([
        ("Quality Routing",     SHARD_C),
        ("Adaptive Scheduling", BANDIT_C),
        ("Combined",            ALLOPT_C),
    ]):
        x = Inches(6.4 + i * 2.2)
        _rect(slide, x, Inches(3.1), Inches(2.0), Inches(0.5), col)
        _box(slide, x, Inches(3.15), Inches(2.0), Inches(0.45),
             label, font_size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    _box(slide, Inches(0.5), Inches(6.9), Inches(12), Inches(0.4),
         "5 independent runs per configuration  ·  10,000 s1 ligands  ·  target = 5 s5 FEP completions",
         font_size=10, color=RGBColor(0x77, 0x88, 0xAA), align=PP_ALIGN.CENTER)


def slide_pipeline_overview(prs):
    """5-stage drug-discovery pipeline — no redundant optimization-axis preview."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, WHITE)
    _title_bar(slide, "Drug-Discovery Pipeline",
               "5-stage cascade — each stage refines quality and filters candidates")

    stages = [
        ("S1\nLigand Filter",  "~3,200 start\n(10,000 queued)",  "#42a5f5", "score > 0.60"),
        ("S2\nML Affinity",    "~313 enter",          "#66bb6a", "score > 0.65"),
        ("S3\nDocking",        "~77 enter",            "#ffa726", "score > 0.70"),
        ("S4\nMD Refinement",  "~21 enter",            "#ef5350", "score > 0.75"),
        ("S5\nFEP Ranking",    "5 hit target",         "#ab47bc", "score > 0.80"),
    ]
    bw  = Inches(2.35)
    gap = Inches(0.14)
    for i, (name, count, col, filt) in enumerate(stages):
        x = Inches(0.25) + i * (bw + gap)
        c = RGBColor(int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16))
        _rect(slide, x, Inches(1.5), bw, Inches(2.6), c)
        _box(slide, x, Inches(1.58), bw, Inches(0.8),
             name, font_size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        _box(slide, x, Inches(2.4), bw, Inches(0.55),
             count, font_size=11, color=WHITE, align=PP_ALIGN.CENTER)
        _box(slide, x, Inches(2.95), bw, Inches(0.85),
             filt, font_size=10, italic=True,
             color=RGBColor(0xEE, 0xEE, 0xEE), align=PP_ALIGN.CENTER)
        if i < 4:
            ax = x + bw
            _box(slide, ax, Inches(2.35), gap + Inches(0.05), Inches(0.45),
                 "▶", font_size=22, color=GRAY_TEXT, align=PP_ALIGN.CENTER)

    # Footnote: where counts come from
    _box(slide, Inches(0.25), Inches(4.1), Inches(12.8), Inches(0.22),
         "† Counts are averages from baseline benchmark runs (5 independent runs).  "
         "Campaign stops when 5th s5 hit is found — most of the 10,000 s1 candidates "
         "never execute (early termination).  Counts vary by configuration; see Cascade Funnel slide.",
         font_size=8.5, italic=True, color=GRAY_TEXT)

    # Resource model description
    _rect(slide, Inches(0.25), Inches(4.35), Inches(12.85), Inches(2.8),
          RGBColor(0xF3, 0xF4, 0xFF))
    _box(slide, Inches(0.4), Inches(4.42), Inches(12.5), Inches(0.38),
         "How SPHERICAL executes this pipeline", font_size=14, bold=True,
         color=ACCENT)

    cols = [
        ("Each stage = a workflow group",
         ["Replicas run in parallel within each group",
          "GPU slots assigned per replica (required_gpus)",
          "min_replicas ensures downstream stages always have slots",
          "max_replicas caps concurrent usage per group"]),
        ("Dependencies drive execution order",
         ["Downstream groups start when upstream signals done",
          "_signal_done() or _trigger_dependent() from workflow code",
          "Sharder buffers upstream results before dispatching",
          "Backpressure prevents queue flooding"]),
        ("Adaptive resource allocation",
         ["Thompson-sampling bandit allocates GPUs cross-stage",
          "Warm-start priors favour terminal stages (s5 > s1)",
          "Learns from backpressure feedback each scheduling cycle",
          "Combined: routes best candidates to best-resourced stage"]),
    ]
    for ci, (title, bullets) in enumerate(cols):
        x = Inches(0.4 + ci * 4.3)
        _box(slide, x, Inches(4.85), Inches(4.1), Inches(0.35),
             title, font_size=11, bold=True, color=ACCENT)
        for bi, b in enumerate(bullets):
            _box(slide, x + Inches(0.1), Inches(5.25 + bi * 0.4), Inches(4.0), Inches(0.38),
                 f"• {b}", font_size=9.5, color=GRAY_TEXT)


def slide_spherical_architecture(prs, diag_dir: Path):
    """Merged architecture slide: CM class hierarchy + scheduler description."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, WHITE)
    _title_bar(slide, "SPHERICAL — System Architecture",
               "AsyncCampaignManager: mixin-based design with optional feature flags")

    _img(slide, diag_dir / "cm_architecture.png",
         Inches(0.15), Inches(1.1), Inches(8.7), Inches(5.9))

    # Right: Scheduler + key design notes
    _rect(slide, Inches(9.05), Inches(1.1), Inches(4.1), Inches(5.9),
          RGBColor(0xF5, 0xF5, 0xF5))
    _box(slide, Inches(9.15), Inches(1.15), Inches(3.9), Inches(0.38),
         "Scheduling Algorithm", font_size=13, bold=True, color=BLACK)

    sched_items = [
        (RGBColor(0x2E, 0x7D, 0x32), "Pass 1 — Fairness",
         "For every eligible group: allocate until running == min_replicas.  "
         "Highest priority first.  Prevents s1 from monopolising all GPUs."),
        (ORANGE_C, "Pass 2 — Throughput",
         "After min_replicas satisfied: fill remaining capacity up to max_replicas.  "
         "Highest priority (or bandit-ranked) stage gets extras first."),
        (BANDIT_C, "Bandit override",
         "When bandit=true: Thompson-sample Beta arm per stage to replace "
         "static priority sort.  Learns downstream-first allocation."),
        (RGBColor(0x01, 0x57, 0x9B), "Dependency eligibility",
         "Group eligible when: deps called _signal_done()  OR  "
         "dep.finished_replicas ≥ dep_threshold."),
    ]
    for i, (col, title, body) in enumerate(sched_items):
        y = Inches(1.6 + i * 1.35)
        _rect(slide, Inches(9.05), y, Inches(4.1), Inches(0.3), col)
        _box(slide, Inches(9.1), y + Inches(0.02), Inches(4.0), Inches(0.28),
             title, font_size=10, bold=True, color=WHITE)
        _box(slide, Inches(9.1), y + Inches(0.33), Inches(4.0), Inches(0.88),
             body, font_size=9, color=GRAY_TEXT)

    _box(slide, Inches(0.15), Inches(7.1), Inches(13.1), Inches(0.3),
         "Feature flags: cm.features.sharder / backpressure / bandit / monitor — all disabled by default",
         font_size=9, italic=True, color=GRAY_TEXT, align=PP_ALIGN.CENTER)


def slide_stage_profiles(prs, diag_dir: Path):
    """NEW: Candidate ranking profiles used by the sharder."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, WHITE)
    _title_bar(slide, "Candidate Ranking Profiles",
               "The sharder scores candidates as: priority = Σ(weight × signal) — choose profile per campaign stage")

    _img(slide, diag_dir / "profiles.png",
         Inches(0.15), Inches(1.1), Inches(8.0), Inches(5.9))

    # Right panel: when to use each profile
    _rect(slide, Inches(8.35), Inches(1.1), Inches(4.8), Inches(5.9),
          RGBColor(0xF3, 0xF4, 0xFF))
    _box(slide, Inches(8.45), Inches(1.15), Inches(4.6), Inches(0.38),
         "When to use each profile", font_size=13, bold=True, color=ACCENT)

    profiles_guide = [
        (SHARD_C,   "pure_promise",
         "Best when upstream scores are reliable.\nPure quality routing — top candidates only.\nUsed in this benchmark (sharding+bp config)."),
        (BANDIT_C,  "active_learning",
         "Early campaign: model needs diverse data.\nMaximises uncertainty reduction.\nSacrifices short-term quality for model accuracy."),
        (TEAL_C,    "explore_exploit",
         "Mid-campaign with calibrated surrogate.\nBalances score, model prediction, uncertainty.\nDefault for most drug-discovery campaigns."),
        (ALLOPT_C,  "diverse_top",
         "Late campaign: avoid chemical echoes.\nQuality-weighted + scaffold novelty bonus.\nPrevents converging on one chemical series."),
        (BASELINE_C, "round_robin",
         "Initial screening / diversity mandate.\nRound-robin across scaffold classes.\nIgnores quality — maximises chemical diversity."),
    ]
    for i, (col, name, desc) in enumerate(profiles_guide):
        y = Inches(1.6 + i * 1.07)
        _rect(slide, Inches(8.35), y, Inches(4.8), Inches(0.28), col)
        _box(slide, Inches(8.42), y + Inches(0.02), Inches(4.6), Inches(0.26),
             name, font_size=10, bold=True, color=WHITE)
        _box(slide, Inches(8.42), y + Inches(0.30), Inches(4.65), Inches(0.72),
             desc, font_size=9, color=GRAY_TEXT)

    _box(slide, Inches(0.15), Inches(7.1), Inches(13.1), Inches(0.3),
         "Profiles are configured per-group in the YAML.  The sharder evaluates all buffered candidates each dispatch cycle.",
         font_size=9, italic=True, color=GRAY_TEXT, align=PP_ALIGN.CENTER)


def slide_benchmark_design(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, WHITE)
    _title_bar(slide, "Benchmark Design",
               "Same workload, different optimisation features — 5 independent runs per config")

    configs = [
        ("baseline",          BASELINE_C, "No features",
         ["Pipelined FIFO — no quality routing", "No resource allocation learning",
          "s1 monopolises GPUs; downstream starved", "Result: 82.6 s ± 19.3 s"]),
        ("sharding+bp",       SHARD_C,    "Quality Routing",
         ["Sharder: highest-score candidates dispatched first",
          "Backpressure: THROTTLE/WIDEN queue control",
          "min_replicas floor keeps s2–s5 slots open",
          "Result: 17.0 s ± 2.5 s  (4.9× faster)"]),
        ("scheduling_bandit", BANDIT_C,   "Adaptive Scheduling",
         ["Thompson-sampling bandit per stage",
          "Learns downstream-first GPU allocation",
          "No quality routing — FIFO dispatch",
          "Result: 26.7 s ± 5.8 s  (3.1× faster)"]),
        ("all_optimizations", ALLOPT_C,   "Both Combined",
         ["Quality routing + adaptive scheduling",
          "Bandit + sharder + backpressure all active",
          "Most consistent: σ=0.9 s on 7.6 s mean",
          "Result: 7.6 s ± 0.9 s  (10.9× faster)"]),
    ]

    for i, (name, color, tag, bullets) in enumerate(configs):
        x = Inches(0.25 + i * 3.27)
        _rect(slide, x, Inches(1.2), Inches(3.1), Inches(0.45), color)
        _box(slide, x, Inches(1.22), Inches(3.1), Inches(0.42),
             f"{name}", font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        _rect(slide, x, Inches(1.65), Inches(3.1), Inches(0.3),
              RGBColor(0xEE, 0xEE, 0xEE))
        _box(slide, x, Inches(1.66), Inches(3.1), Inches(0.3),
             tag, font_size=11, italic=True, color=GRAY_TEXT, align=PP_ALIGN.CENTER)
        _rect(slide, x, Inches(1.95), Inches(3.1), Inches(2.5),
              RGBColor(0xF8, 0xF8, 0xF8))
        for j, b in enumerate(bullets):
            bold_last = (j == len(bullets) - 1)
            col = color if bold_last else BLACK
            _box(slide, x + Inches(0.05), Inches(2.0 + j * 0.58),
                 Inches(3.0), Inches(0.55),
                 f"{'→' if bold_last else '•'} {b}",
                 font_size=11, bold=bold_last, color=col)

    _rect(slide, Inches(0.25), Inches(4.6), Inches(12.8), Inches(0.65),
          RGBColor(0xE3, 0xF2, 0xFD))
    _box(slide, Inches(0.35), Inches(4.65), Inches(12.5), Inches(0.55),
         "Setup:  10,000 s1 candidates  ·  early termination when s5_fep_ranking hits 5 completions  "
         "·  same random seed per run index across all configs  ·  concurrent asyncio backend (no real GPU hardware)",
         font_size=11, color=RGBColor(0x0D, 0x47, 0xA1))

    _box(slide, Inches(0.25), Inches(5.35), Inches(12.8), Inches(1.95),
         "Note: 'baseline' is pipelined FIFO (not a true sequential waterfall) — stages start as "
         "soon as the first upstream replica completes. This makes baseline HARDER to beat than a "
         "pure waterfall, so the measured speedups are conservative.",
         font_size=11, italic=True, color=GRAY_TEXT)


def slide_cascade_funnel(prs):
    """Standalone cascade funnel — shows pipeline compute cost per config."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, WHITE)
    _title_bar(slide, "Cascade Funnel — Total Pipeline Work per Configuration",
               "How many replicas were launched at each stage to find 5 s5 FEP hits")

    _img(slide, PLOT_DIR / "3_cascade_funnel.png",
         Inches(0.15), Inches(1.15), Inches(8.7), Inches(5.0))

    # Right: key numbers
    _rect(slide, Inches(9.05), Inches(1.15), Inches(4.1), Inches(5.0),
          RGBColor(0xF5, 0xF5, 0xF5))
    _box(slide, Inches(9.15), Inches(1.2), Inches(3.9), Inches(0.38),
         "Total replicas launched", font_size=12, bold=True, color=BLACK)

    funnel_stats = [
        (BASELINE_C,  "baseline",          "3,622 replicas\ns1 monopolises pipeline"),
        (SHARD_C,     "sharding+bp",       "726 replicas\n5× less compute"),
        (BANDIT_C,    "scheduling_bandit", "1,109 replicas\n3.3× less compute"),
        (ALLOPT_C,    "all_optimizations", "119 replicas\n30× less s1 work"),
    ]
    for i, (col, name, stat) in enumerate(funnel_stats):
        y = Inches(1.65 + i * 1.1)
        _rect(slide, Inches(9.05), y, Inches(4.1), Inches(1.0), col)
        _box(slide, Inches(9.12), y + Inches(0.04), Inches(3.9), Inches(0.34),
             name, font_size=11, bold=True, color=WHITE)
        _box(slide, Inches(9.12), y + Inches(0.40), Inches(3.9), Inches(0.52),
             stat, font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    _box(slide, Inches(0.15), Inches(6.2), Inches(13.0), Inches(1.0),
         "Left: stacked bars show absolute replica counts per config — s1 dominates baseline.  "
         "Right: log scale reveals all 5 stages.  Sharding dispatches only the top-scoring ~20% of s1 "
         "results downstream — drastically shrinking every subsequent stage.",
         font_size=9.5, italic=True, color=GRAY_TEXT)


def slide_main_result(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, WHITE)
    _title_bar(slide, "Main Result — Wall Time to Target",
               "Time from campaign start until 5th s5_fep_ranking completion  ·  5 runs per config")

    _img(slide, PLOT_DIR / "1_wall_time.png",
         Inches(0.2), Inches(1.15), Inches(7.8), Inches(5.5))

    stats = [
        ("82.6 s",  "baseline\n(±19.3 s)",            BASELINE_C),
        ("17.0 s",  "sharding+bp\n4.9× faster",        SHARD_C),
        ("26.7 s",  "scheduling_bandit\n3.1× faster",  BANDIT_C),
        ("7.6 s",   "all_optimizations\n10.9× faster", ALLOPT_C),
    ]
    for i, (val, lbl, col) in enumerate(stats):
        y = Inches(1.2 + i * 1.5)
        _rect(slide, Inches(8.3), y, Inches(4.8), Inches(1.3), col)
        _box(slide, Inches(8.3), y + Inches(0.08), Inches(4.8), Inches(0.65),
             val, font_size=38, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        _box(slide, Inches(8.3), y + Inches(0.72), Inches(4.8), Inches(0.5),
             lbl, font_size=12, color=WHITE, align=PP_ALIGN.CENTER)

    _box(slide, Inches(8.3), Inches(7.1), Inches(4.8), Inches(0.3),
         "Lower is better  ·  white dots = individual runs",
         font_size=9, italic=True, color=GRAY_TEXT, align=PP_ALIGN.CENTER)


def slide_sharding_bp(prs, diag_dir: Path = None):
    """Optimisation 1 — fully pptx-native layout (no embedded image)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, WHITE)
    _title_bar(slide, "Optimisation 1 — Quality Routing",
               "Sharder + BackpressureNegotiator + Shard Bandit  ·  17.0 s ± 2.5 s  (4.9× faster)")

    # ── Left column: visual diagrams ──────────────────────────────────────

    # --- Data flow section label ---
    _rect(slide, Inches(0.15), Inches(1.18), Inches(8.55), Inches(0.28),
          RGBColor(0xE8, 0xF5, 0xE9))
    _box(slide, Inches(0.22), Inches(1.20), Inches(8.4), Inches(0.25),
         "SHARDER  —  upstream trigger  →  buffer  →  rank  →  dispatch",
         font_size=9.5, bold=True, color=RGBColor(0x1B, 0x5E, 0x20))

    # Data flow boxes (4 boxes + arrows)
    flow_y  = Inches(1.52)
    flow_h  = Inches(1.6)
    box_w   = Inches(1.88)
    arr_w   = Inches(0.36)
    gap     = arr_w
    flow_items = [
        (Inches(0.15), "S1 Replica\nDone",    SHARD_C,
         "trigger(cid,\nscore, surr,\nuncertainty,\nscaffold)"),
        (Inches(0.15) + box_w + gap, "BUFFER",    RGBColor(0x1B, 0x5E, 0x20),
         "all candidates\nenqueued\nsorted by\npriority score"),
        (Inches(0.15) + 2*(box_w + gap), "Ranking\nEngine",  BANDIT_C,
         "priority =\nΣ weight\n× signal\n(profile)"),
        (Inches(0.15) + 3*(box_w + gap), "S2 Queue", RGBColor(0x01, 0x57, 0x9B),
         "highest-score\ncandidates\ndispatched\nfirst"),
    ]
    for i, (x, name, col, sub) in enumerate(flow_items):
        _rect(slide, x, flow_y, box_w, flow_h, col)
        _box(slide, x + Inches(0.05), flow_y + Inches(0.07),
             box_w - Inches(0.1), Inches(0.44),
             name, font_size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        _box(slide, x + Inches(0.05), flow_y + Inches(0.52),
             box_w - Inches(0.1), Inches(0.98),
             sub, font_size=9, color=RGBColor(0xDD, 0xFF, 0xDD), align=PP_ALIGN.CENTER)
        if i < 3:
            arr_x = x + box_w
            _box(slide, arr_x, flow_y + Inches(0.65), arr_w, Inches(0.35),
                 "▶", font_size=18, color=GRAY_TEXT, align=PP_ALIGN.CENTER)

    # Adaptive batch sizing note (below flow)
    _rect(slide, Inches(0.15), Inches(3.18), Inches(8.55), Inches(0.28),
          RGBColor(0x00, 0x60, 0x64))
    _box(slide, Inches(0.22), Inches(3.20), Inches(8.4), Inches(0.25),
         "Batch sizing:  stratify=soft → adaptive;  stratify=strict → hold until full batch;  "
         "shard bandit learns multiplier arms [0.5 0.75 1.0 1.25 1.5]",
         font_size=9, color=WHITE)

    # --- Backpressure state machine ---
    _rect(slide, Inches(0.15), Inches(3.55), Inches(8.55), Inches(0.28),
          RGBColor(0xFF, 0xE0, 0xB2))
    _box(slide, Inches(0.22), Inches(3.57), Inches(8.4), Inches(0.25),
         "BACKPRESSURE NEGOTIATOR  —  hysteresis state machine controlling dispatch rate",
         font_size=9.5, bold=True, color=RGBColor(0xE6, 0x51, 0x00))

    # Three state boxes
    BP_Y   = Inches(3.9)
    BP_H   = Inches(1.55)
    BP_W   = Inches(2.35)
    states = [
        (Inches(0.15),   "HOLD",     "normal operation",
         "dispatch proceeds\nat normal rate",    RGBColor(0x43, 0xA0, 0x47)),
        (Inches(2.97),   "THROTTLE", "queue ≥ high_water",
         "dispatch = 0\npipeline paused",         RGBColor(0xE5, 0x39, 0x35)),
        (Inches(5.8),    "WIDEN",    "queue ≤ low_water",
         "dispatch × mult\nqueue drained",        RGBColor(0x1E, 0x88, 0xE5)),
    ]
    for x, title, cond, action, col in states:
        _rect(slide, x, BP_Y, BP_W, BP_H, col)
        _box(slide, x + Inches(0.05), BP_Y + Inches(0.06),
             BP_W - Inches(0.1), Inches(0.38),
             title, font_size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        _box(slide, x + Inches(0.05), BP_Y + Inches(0.44),
             BP_W - Inches(0.1), Inches(0.3),
             cond, font_size=8.5, italic=True,
             color=RGBColor(0xFF, 0xFF, 0xCC), align=PP_ALIGN.CENTER)
        _box(slide, x + Inches(0.05), BP_Y + Inches(0.78),
             BP_W - Inches(0.1), Inches(0.6),
             action, font_size=10, color=WHITE, align=PP_ALIGN.CENTER)

    # Transition arrows between states (text-based)
    _box(slide, Inches(2.52), BP_Y + Inches(0.58), Inches(0.45), Inches(0.4),
         "▶", font_size=18, color=RGBColor(0xE5, 0x39, 0x35), align=PP_ALIGN.CENTER)
    _box(slide, Inches(5.35), BP_Y + Inches(0.58), Inches(0.45), Inches(0.4),
         "▶", font_size=18, color=RGBColor(0x1E, 0x88, 0xE5), align=PP_ALIGN.CENTER)

    # Return path label
    _rect(slide, Inches(0.15), BP_Y + BP_H + Inches(0.05),
          Inches(8.55), Inches(0.3), RGBColor(0x43, 0xA0, 0x47))
    _box(slide, Inches(0.22), BP_Y + BP_H + Inches(0.07),
         Inches(8.4), Inches(0.25),
         "◀─── when queue returns to normal range, state resets to HOLD ───────────────────────",
         font_size=9, color=WHITE)

    # Config params
    _box(slide, Inches(0.15), BP_Y + BP_H + Inches(0.42), Inches(8.55), Inches(0.28),
         "Config:  backpressure_high (high_water mark)  ·  backpressure_low (low_water mark)  "
         "·  queue_depth = group.replicas − group.started_count",
         font_size=8.5, italic=True, color=GRAY_TEXT)

    # ── Right column: bullet descriptions ─────────────────────────────────
    RW = Inches(4.3)
    RX = Inches(8.95)

    _section_panel(
        slide, RX, Inches(1.18), RW, Inches(1.85),
        "Sharder  (sharder.py)",
        ["Buffer upstream trigger results (score, surrogate, uncertainty, scaffold)",
         "Rank candidates: priority = Σ(weight × signal) via ProfileWeights",
         "Dispatch best candidates to s2 first  (stratify=soft: adaptive batch size)",
         "Flush buffer when upstream stage completes"],
        title_bg=SHARD_C,
        body_bg=RGBColor(0xE8, 0xF5, 0xE9),
        bullet_color=RGBColor(0x1B, 0x5E, 0x20),
        bullet_size=9.5,
    )

    _section_panel(
        slide, RX, Inches(3.13), RW, Inches(1.95),
        "BackpressureNegotiator  (backpressure.py)",
        ["HOLD  → queue depth between thresholds → dispatch normal",
         "THROTTLE  → depth ≥ high_water → pause dispatch (return 0)",
         "WIDEN  → depth ≤ low_water → dispatch × multiplier (> 1)",
         "Hysteresis prevents rapid oscillation between states",
         "Config: backpressure_high / backpressure_low per group"],
        title_bg=RGBColor(0xE6, 0x51, 0x00),
        body_bg=RGBColor(0xFF, 0xF3, 0xE0),
        bullet_color=RGBColor(0x7F, 0x3B, 0x00),
        bullet_size=9.5,
    )

    _section_panel(
        slide, RX, Inches(5.18), RW, Inches(1.42),
        "Shard Bandit  (optional, bandit.py)",
        ["Arms = [0.5, 0.75, 1.0, 1.25, 1.5]  (dispatch multipliers)",
         "Thompson-samples arm to adjust batch size each dispatch cycle",
         "Reward = throughput improvement over last window",
         "Learns optimal batch size for current pipeline state"],
        title_bg=BANDIT_C,
        body_bg=RGBColor(0xF3, 0xE5, 0xF5),
        bullet_color=RGBColor(0x4A, 0x14, 0x8C),
        bullet_size=9.5,
    )

    # Result callout
    _rect(slide, RX, Inches(6.7), RW, Inches(0.68), SHARD_C)
    _box(slide, RX + Inches(0.08), Inches(6.73), RW - Inches(0.15), Inches(0.28),
         "Result:  17.0 s ± 2.5 s  ·  4.9× faster wall time",
         font_size=11, bold=True, color=WHITE)
    _box(slide, RX + Inches(0.08), Inches(7.01), RW - Inches(0.15), Inches(0.28),
         "5× fewer total replicas launched  (3,622 → 726)", font_size=11, color=WHITE)


def slide_bandit(prs):
    """Optimisation 2 — Scheduling Bandit: GPU utilization + algorithm description."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, WHITE)
    _title_bar(slide, "Optimisation 2 — Adaptive Scheduling  (Thompson-sampling Bandit)",
               "Learns to allocate freed GPU slots to the highest-value stage  ·  26.7 s ± 5.8 s  (3.1× faster)")

    # Left: GPU utilization plot — full height
    _img(slide, PLOT_DIR / "4_gpu_utilization.png",
         Inches(0.15), Inches(1.15), Inches(8.6), Inches(5.85))

    # Right: Algorithm description
    RW = Inches(4.3)
    RX = Inches(8.95)

    _rect(slide, RX, Inches(1.15), RW, Inches(0.35), BANDIT_C)
    _box(slide, RX + Inches(0.08), Inches(1.17), RW - Inches(0.15), Inches(0.32),
         "Thompson Sampling Algorithm", font_size=11, bold=True, color=WHITE)

    algo_steps = [
        "1.  GPU slot freed → collect eligible stages",
        "2.  Sample θᵢ ~ Beta(αᵢ, βᵢ) for each stage",
        "3.  Assign slot to stage with highest θ",
        "4.  Replica runs → measure downstream BP state",
        "5.  Compute reward r ∈ [0,1] (see below)",
        "6.  Update posterior:  αᵢ += r,  βᵢ += (1-r)",
    ]
    _rect(slide, RX, Inches(1.5), RW, Inches(2.2), RGBColor(0xF3, 0xE5, 0xF5))
    for j, step in enumerate(algo_steps):
        _box(slide, RX + Inches(0.1), Inches(1.53 + j * 0.35), RW - Inches(0.15), Inches(0.34),
             step, font_size=9.5, color=RGBColor(0x4A, 0x14, 0x8C))

    # Warm-start priors
    _rect(slide, RX, Inches(3.8), RW, Inches(0.3), RGBColor(0x4A, 0x14, 0x8C))
    _box(slide, RX + Inches(0.08), Inches(3.82), RW - Inches(0.15), Inches(0.28),
         "Warm-start priors", font_size=10, bold=True, color=WHITE)
    _rect(slide, RX, Inches(4.1), RW, Inches(0.95), RGBColor(0xED, 0xE7, 0xF6))
    priors = [
        ("s5 FEP Ranking",   "Beta(5,1)", BANDIT_C),
        ("s4 MD Refine",     "Beta(4,1)", RGBColor(0xEF, 0x53, 0x50)),
        ("s3 Docking",       "Beta(3,1)", RGBColor(0xFF, 0xA7, 0x26)),
        ("s2 ML Affinity",   "Beta(2,1)", RGBColor(0x66, 0xBB, 0x6A)),
        ("s1 Ligand Filter", "Beta(1,1) neutral", BASELINE_C),
    ]
    for j, (stage, prior, col) in enumerate(priors):
        x_off = j * (RW / 5)
        _box(slide, RX + x_off, Inches(4.13), RW / 5, Inches(0.42),
             f"{prior}\n{stage.split()[0]}", font_size=8, bold=False,
             color=col, align=PP_ALIGN.CENTER)

    # Reward signal
    _rect(slide, RX, Inches(5.15), RW, Inches(0.3), RGBColor(0x4A, 0x14, 0x8C))
    _box(slide, RX + Inches(0.08), Inches(5.17), RW - Inches(0.15), Inches(0.28),
         "Reward signal (from BP state)", font_size=10, bold=True, color=WHITE)
    _rect(slide, RX, Inches(5.45), RW, Inches(0.85), RGBColor(0xED, 0xE7, 0xF6))
    for j, (label, r, col) in enumerate([
        ("THROTTLE (queue flooded)", "r = 0.2", RGBColor(0xC6, 0x28, 0x28)),
        ("HOLD (queue healthy)",      "r = 0.5-0.8", RGBColor(0x2E, 0x7D, 0x32)),
        ("WIDEN (queue drained)",     "r = 0.8", RGBColor(0x15, 0x65, 0xC0)),
    ]):
        _box(slide, RX + Inches(0.1), Inches(5.48 + j * 0.28), RW - Inches(0.2), Inches(0.27),
             f"• {label}  →  {r}", font_size=9.5, color=col)

    # Result callout
    _rect(slide, RX, Inches(6.4), RW, Inches(0.6), BANDIT_C)
    _box(slide, RX + Inches(0.08), Inches(6.43), RW - Inches(0.15), Inches(0.27),
         "s5 first start: baseline 14.5 s  →  bandit 6.9 s  (2.1×)", font_size=10,
         bold=True, color=WHITE)
    _box(slide, RX + Inches(0.08), Inches(6.7), RW - Inches(0.15), Inches(0.27),
         "Wall time 3.1× faster  ·  3.3× fewer total replicas", font_size=10, color=WHITE)


def slide_all_opt(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, WHITE)
    _title_bar(slide, "Optimisation 3 — Combined  (all_optimizations)",
               "Quality routing + adaptive scheduling — both axes active simultaneously")

    _img(slide, PLOT_DIR / "7_time_to_target.png",
         Inches(0.15), Inches(1.15), Inches(8.0), Inches(5.1))

    _rect(slide, Inches(8.3), Inches(1.15), Inches(4.8), Inches(5.1),
          RGBColor(0xFF, 0xEB, 0xEE))
    _box(slide, Inches(8.4), Inches(1.2), Inches(4.6), Inches(0.4),
         "Why combined > each alone", font_size=14, bold=True,
         color=RGBColor(0xB7, 0x1C, 0x1C))

    for j, txt in enumerate([
        "Sharding routes high-quality candidates\n  to s2–s5 FIRST",
        "Bandit allocates s5 GPUs from t~1 s\n  (warm-start prior)",
        "Together: s5 gets the BEST candidates\n  with the MOST GPU resources",
        "s5 hits target with only 10 s5 starts\n  from just 119 s1 replicas",
        "15× less total compute vs baseline",
    ]):
        _box(slide, Inches(8.4), Inches(1.65 + j * 0.72), Inches(4.6), Inches(0.7),
             f"• {txt}", font_size=11, color=RGBColor(0xB7, 0x1C, 0x1C))

    _rect(slide, Inches(8.3), Inches(5.4), Inches(4.8), Inches(0.85), ALLOPT_C)
    _box(slide, Inches(8.35), Inches(5.42), Inches(4.7), Inches(0.42),
         "10.9×  faster  ·  15×  less compute", font_size=22, bold=True,
         color=WHITE, align=PP_ALIGN.CENTER)
    _box(slide, Inches(8.35), Inches(5.82), Inches(4.7), Inches(0.38),
         "7.6 s ± 0.9 s  (most consistent of all configs)",
         font_size=11, color=WHITE, align=PP_ALIGN.CENTER)

    _box(slide, Inches(0.15), Inches(6.3), Inches(13.0), Inches(0.85),
         "Time-to-target step curves: faint lines = all 5 runs per config; bold = median run.  "
         "▼ markers show when each config crosses N=5.  "
         "Expected independent speedup = 4.9×3.1=15.2×; actual 10.9× shows moderate overlap "
         "(both optimisations reduce wasted compute, so savings partially overlap).",
         font_size=9, italic=True, color=GRAY_TEXT)


def slide_gantt(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, WHITE)
    _title_bar(slide, "Pipeline Stage Overlap — Gantt View",
               "Average first-start to last-finish per stage; more overlap = better pipeline utilisation")

    _img(slide, PLOT_DIR / "2_pipeline_gantt.png",
         Inches(0.15), Inches(1.15), Inches(9.5), Inches(5.8))

    _rect(slide, Inches(9.8), Inches(1.15), Inches(3.3), Inches(5.8),
          RGBColor(0xF5, 0xF5, 0xF5))
    _box(slide, Inches(9.9), Inches(1.2), Inches(3.1), Inches(0.4),
         "What to look for", font_size=13, bold=True, color=BLACK)

    insights = [
        (BASELINE_C,  "baseline",
         "All stages sequential — s1 finishes before s2 fills up."),
        (SHARD_C,     "sharding+bp",
         "s2–s5 bars start within seconds of s1 due to min_replicas floor."),
        (BANDIT_C,    "scheduling_bandit",
         "s3/s4/s5 overlap deeply with s1 — bandit feeds terminal stages while s1 still runs."),
        (ALLOPT_C,    "all_optimizations",
         "All 5 bars nearly co-incident; campaign ends at t=7.6 s."),
    ]
    for j, (col, name, desc) in enumerate(insights):
        y = Inches(1.7 + j * 1.3)
        _rect(slide, Inches(9.85), y, Inches(0.18), Inches(0.28), col)
        _box(slide, Inches(10.1), y - Inches(0.02), Inches(2.9), Inches(0.32),
             name, font_size=11, bold=True, color=col)
        _box(slide, Inches(10.0), y + Inches(0.3), Inches(3.0), Inches(0.75),
             desc, font_size=10, color=GRAY_TEXT)


def slide_planner_execution(prs, diag_dir: Path):
    """NEW: SPHERICAL execution model — from YAML config to running replicas."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, WHITE)
    _title_bar(slide, "SPHERICAL — Execution Model",
               "From campaign YAML config to asyncflow engine: how the planner and executor relate")

    _img(slide, diag_dir / "planner.png",
         Inches(0.15), Inches(1.1), Inches(8.7), Inches(5.9))

    # Right: key concepts
    _rect(slide, Inches(9.05), Inches(1.1), Inches(4.1), Inches(5.9),
          RGBColor(0xF5, 0xF5, 0xF5))
    _box(slide, Inches(9.15), Inches(1.15), Inches(3.9), Inches(0.38),
         "Execution Plan Concepts", font_size=13, bold=True, color=BLACK)

    plan_items = [
        (RGBColor(0x01, 0x57, 0x9B), "YAML Config = Plan",
         "Each workflow group is a named pool of replicas.  "
         "Dependencies define the DAG.  "
         "Resources (CPUs/GPUs) cap concurrent execution."),
        (SHARD_C, "Dynamic plan update",
         "_trigger_dependent(name, N) lets an upstream workflow "
         "add N replicas to a downstream group at runtime — "
         "the plan adapts to intermediate results."),
        (BANDIT_C, "Scheduler as planner",
         "Every state change re-runs _schedule_locked().  "
         "The two-pass algorithm is the 'planner' that decides "
         "which groups get resources each cycle."),
        (ALLOPT_C, "asyncflow = executor",
         "CM creates asyncio tasks; asyncflow engine manages "
         "the event loop, task lifecycle, and backend "
         "(concurrent or Dragon HPC)."),
    ]
    for i, (col, title, body) in enumerate(plan_items):
        y = Inches(1.6 + i * 1.35)
        _rect(slide, Inches(9.05), y, Inches(4.1), Inches(0.3), col)
        _box(slide, Inches(9.1), y + Inches(0.02), Inches(4.0), Inches(0.28),
             title, font_size=10, bold=True, color=WHITE)
        _box(slide, Inches(9.1), y + Inches(0.33), Inches(4.0), Inches(0.88),
             body, font_size=9, color=GRAY_TEXT)

    _box(slide, Inches(0.15), Inches(7.1), Inches(13.1), Inches(0.3),
         "Sync wrapper (CampaignManager) provides a blocking API for non-async callers; "
         "AsyncCampaignManager is the native async class.",
         font_size=9, italic=True, color=GRAY_TEXT, align=PP_ALIGN.CENTER)


def slide_methodology(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, WHITE)
    _title_bar(slide, "Methodology — What's Real vs What to Watch",
               "Verified findings and known caveats")

    confirmed = [
        "10.9× wall-time speedup is correctly measured (time from start to 5th s5 finish)",
        "Cascade funnel reduction (15× less total work) uses n_started — accurate",
        "Same random seed per run index ensures consistent score distributions across configs",
        "All 5 runs per config completed successfully (no timeouts or failures)",
        "all_optimizations has the lowest run-to-run variance (σ/mean = 12% vs 23% for baseline)",
    ]
    caveats = [
        "Baseline label says 'waterfall' but dep_threshold_override is commented out — "
        "baseline is actually pipelined FIFO, making speedups MORE conservative",
        "all_optimizations over-provisions s5: ~10 s5 replicas start to find 5 hits "
        "(bandit warm-start Beta(5,1) is aggressive — 2× s5 waste)",
        "BP fractions show 100% WIDEN for all runs — queue never hit high-water mark "
        "(BP controlled dispatch rate but never fully throttled in these short runs)",
        "Runs use asyncio concurrent backend (no real GPU hardware) — timing models "
        "stage durations with simulated sleep + jitter, not actual compute",
    ]

    _rect(slide, Inches(0.25), Inches(1.15), Inches(6.2), Inches(5.5),
          RGBColor(0xE8, 0xF5, 0xE9))
    _box(slide, Inches(0.35), Inches(1.2), Inches(5.9), Inches(0.4),
         "✓  Confirmed — results are real", font_size=13, bold=True,
         color=RGBColor(0x1B, 0x5E, 0x20))
    for j, txt in enumerate(confirmed):
        _box(slide, Inches(0.4), Inches(1.65 + j * 0.95), Inches(5.9), Inches(0.9),
             f"✓  {txt}", font_size=10.5, color=RGBColor(0x1B, 0x5E, 0x20))

    _rect(slide, Inches(6.7), Inches(1.15), Inches(6.4), Inches(5.5),
          RGBColor(0xFF, 0xF9, 0xC4))
    _box(slide, Inches(6.8), Inches(1.2), Inches(6.1), Inches(0.4),
         "⚠  Caveats — known limitations", font_size=13, bold=True,
         color=RGBColor(0xE6, 0x5C, 0x00))
    for j, txt in enumerate(caveats):
        _box(slide, Inches(6.85), Inches(1.65 + j * 1.22), Inches(6.1), Inches(1.1),
             f"⚠  {txt}", font_size=10.5, color=RGBColor(0x7F, 0x3B, 0x00))


def slide_summary(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(slide, DARK_BG)
    _rect(slide, Inches(0), Inches(0), SLIDE_W, Inches(1.1), ACCENT)
    _box(slide, Inches(0.3), Inches(0.12), Inches(12.5), Inches(0.85),
         "Summary", font_size=32, bold=True, color=WHITE)

    stats = [
        ("10.9×",  "wall-time speedup\nall_optimizations vs baseline",  ALLOPT_C),
        ("15×",    "less total compute\n(replicas launched)",             SHARD_C),
        ("7.6 s",  "median time-to-5-hits\nall_optimizations (±0.9 s)", ALLOPT_C),
        ("4.9× / 3.1×", "sharding+bp / bandit\nindividual gains",       BANDIT_C),
    ]
    for i, (val, lbl, col) in enumerate(stats):
        x = Inches(0.25 + i * 3.27)
        _rect(slide, x, Inches(1.25), Inches(3.1), Inches(1.7), col)
        _box(slide, x, Inches(1.3), Inches(3.1), Inches(0.9),
             val, font_size=34, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        _box(slide, x, Inches(2.15), Inches(3.1), Inches(0.65),
             lbl, font_size=11, color=WHITE, align=PP_ALIGN.CENTER)

    takeaways = [
        ("Quality routing wins on compute", SHARD_C,
         "Routing high-score candidates first cuts total pipeline work by 5×.  "
         "The cascade stays narrow — only the best leads reach expensive downstream stages."),
        ("Bandit wins on latency", BANDIT_C,
         "Thompson-sampling allocation gets s5 slots filled 2× earlier than baseline.  "
         "Even without quality filtering, earlier resource allocation cuts wall time 3×."),
        ("Combined is super-additive", ALLOPT_C,
         "Best candidates reach a well-resourced s5 simultaneously.  "
         "Result: 10.9× speedup and 12% run-to-run variance — the most reliable configuration."),
    ]
    for i, (title, col, body) in enumerate(takeaways):
        x = Inches(0.25 + i * 4.37)
        _rect(slide, x, Inches(3.1), Inches(4.1), Inches(0.38), col)
        _box(slide, x + Inches(0.05), Inches(3.12), Inches(4.0), Inches(0.35),
             title, font_size=12, bold=True, color=WHITE)
        _rect(slide, x, Inches(3.48), Inches(4.1), Inches(2.5),
              RGBColor(0x0A, 0x2A, 0x52))
        _box(slide, x + Inches(0.08), Inches(3.52), Inches(3.95), Inches(2.42),
             body, font_size=11, color=RGBColor(0xCC, 0xDD, 0xFF))

    _box(slide, Inches(0.3), Inches(6.1), Inches(12.5), Inches(0.3),
         "Experiment: 10,000 s1 ligands · target = 5 s5 FEP completions · 5 independent runs · asyncio concurrent backend",
         font_size=9, color=RGBColor(0x77, 0x88, 0xAA), align=PP_ALIGN.CENTER)


# ── Architecture diagram generators ──────────────────────────────────────────

DIAG_DIR = Path(__file__).parent / "plots" / "diagrams"

_BG   = "#F4F6FB"
_NAVY = "#1A237E"
_BLUE = "#1565C0"
_GRN  = "#2E7D32"
_PRP  = "#6A1B9A"
_ORG  = "#E65100"
_RED  = "#B71C1C"
_GRY  = "#37474F"
_LBL  = "#546E7A"
_TEAL = "#00838F"


def _fbox(ax, x, y, w, h, label, sublabel="", fc="#1565C0", tc="white",
          fs=11, sfs=8.5, radius=0.05, lw=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad={radius}",
                                facecolor=fc, edgecolor="white", linewidth=lw, zorder=3))
    ly = y + h * (0.62 if sublabel else 0.5)
    ax.text(x + w / 2, ly, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=tc, zorder=4)
    if sublabel:
        ax.text(x + w / 2, y + h * 0.28, sublabel, ha="center", va="center",
                fontsize=sfs, color=tc, alpha=0.88, zorder=4, fontstyle="italic")


def _arrow(ax, x0, y0, x1, y1, color="#555", lw=1.5, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0"))


def _label(ax, x, y, text, fs=9, color="#333", ha="center", va="center",
           bold=False, italic=False):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color,
            fontweight="bold" if bold else "normal",
            fontstyle="italic" if italic else "normal", zorder=5)


def _diag_save(fig, path, facecolor=_BG):
    fig.patch.set_facecolor(facecolor)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=facecolor)
    plt.close()


# ── Diagram 1: CM Architecture ────────────────────────────────────────────────

def make_cm_arch_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.5); ax.axis("off")

    _fbox(ax, 0.3, 7.0, 13.4, 1.2,
          "AsyncCampaignManager",
          "from_config(yaml, registry, asyncflow)  ·  start()  ·  wait()  ·  close()  ·  metrics()",
          fc=_NAVY, fs=18, sfs=10)

    mixin_specs = [
        ("SchedulerMixin", "Two-pass greedy scheduler\nPass 1: guarantee min_replicas\nPass 2: fill to max_replicas\nPriority / bandit ordering", _GRN,  0.3),
        ("ExecutorMixin",  "Replica lifecycle\nLaunch → monitor → complete\nGPU ID assignment\nEarly termination", _ORG, 4.85),
        ("MonitorMixin",   "Periodic health checks\nDrift detection\nStall alerting\nBP transitions", _PRP, 9.4),
    ]
    for name, desc, fc, x in mixin_specs:
        _fbox(ax, x, 4.8, 4.2, 2.0, name, desc, fc=fc, fs=13, sfs=9.5)
        _arrow(ax, x + 2.1, 7.0, x + 2.1, 6.8, color="white")

    struct_specs = [
        ("_GroupInfo",       "replicas · min/max_replicas\npriority · dependencies\nstatus · running_count", "#01579B"),
        ("ResourcePool",     "total_cpus · total_gpus\ncan_fit() / allocate()\nrelease()", "#01579B"),
        ("CampaignMetrics",  "replica_events\nscheduling_events\nbp_fractions · shard_events", "#01579B"),
        ("BaseWorkflow",     "_signal_done()\n_trigger_dependent()\nrun()  or  start()", _GRY),
    ]
    for i, (name, desc, fc) in enumerate(struct_specs):
        x = 0.3 + i * 3.42
        _fbox(ax, x, 2.7, 3.1, 1.85, name, desc, fc=fc, fs=10.5, sfs=8.5)
        if i < 3:
            _arrow(ax, x + 1.55, 4.8, x + 1.55, 4.55, color="#aaa")

    ax.text(7.0, 2.55, "Core data structures", ha="center", va="center",
            fontsize=9, style="italic", color=_LBL)

    feat_specs = [
        ("Sharder",                 "Buffer → Rank → Dispatch\nAdaptive batch sizing\nShard bandit arm"),
        ("BackpressureNegotiator",  "HOLD / THROTTLE / WIDEN\nHysteresis queue control\nDispatch multiplier"),
        ("SchedulingBandit",        "Thompson sampling\nCross-stage GPU allocation\nBeta arm per stage"),
        ("CandidateLog",            "Score + surrogate history\nScaffold class diversity\nPriority ranking"),
    ]
    for i, (name, desc) in enumerate(feat_specs):
        x = 0.3 + i * 3.42
        _fbox(ax, x, 0.4, 3.1, 1.9, name, desc, fc=_GRN, fs=10.5, sfs=8.5)

    ax.text(7.0, 0.22, "Optional features — enabled via feature flags in config YAML",
            ha="center", va="center", fontsize=9, style="italic", color=_GRN)

    for i, flag in enumerate(["sharder=true", "backpressure=true", "bandit=true", ""]):
        if flag:
            ax.text(0.3 + i * 3.42 + 1.55, 2.45, flag,
                    ha="center", va="center", fontsize=7.5, color=_GRN, style="italic",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#E8F5E9", ec=_GRN, lw=0.8))

    ax.set_title("AsyncCampaignManager — Class Architecture", fontsize=14,
                 fontweight="bold", color=_NAVY, pad=6)
    _diag_save(fig, path)


# ── Diagram 2: Sharder + Backpressure ─────────────────────────────────────────

def make_sharder_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")

    _fbox(ax, 0.2, 4.5, 2.5, 2.4,
          "Upstream\nStage  (s1)",
          "on_replica_done()\n→ score, surr_pred,\n   surr_unc computed\n→ _trigger_dependent()", fc=_BLUE, fs=12, sfs=9)

    _arrow(ax, 2.7, 5.7, 3.2, 5.7, color=_GRN, lw=2)
    ax.text(2.95, 5.95, "trigger(candidate_id,\n  score, surr_pred,\n  surr_unc, scaffold)",
            ha="center", va="bottom", fontsize=7.5, color=_GRN)

    buf_fc = "#E8F5E9"
    ax.add_patch(FancyBboxPatch((3.2, 3.5), 2.8, 4.2,
                                boxstyle="round,pad=0.12",
                                facecolor=buf_fc, edgecolor=_GRN, linewidth=2, zorder=2))
    ax.text(4.6, 7.4, "BUFFER", ha="center", va="center",
            fontsize=13, fontweight="bold", color=_GRN, zorder=5)
    ax.text(4.6, 7.05, "candidate queue", ha="center", va="center",
            fontsize=9, color=_GRN, style="italic", zorder=5)

    for yi, (score, lbl) in enumerate([
        (0.94, "0.94"), (0.88, "0.88"), (0.81, "0.81"),
        (0.75, "0.75"), (0.68, "0.68"), (0.61, "0.61"),
    ]):
        y = 6.55 - yi * 0.48
        c = plt.cm.RdYlGn(score)
        ax.add_patch(plt.Circle((3.95, y), 0.17, color=c, zorder=6))
        ax.text(4.25, y, f"score={lbl}", va="center", fontsize=7.5, color=_GRY, zorder=6)

    _fbox(ax, 6.3, 4.5, 3.5, 2.4,
          "Ranking Engine",
          "priority =\nw_score × score\n+ w_surr × surr_pred\n+ w_unc × surr_unc\n+ w_age × age",
          fc=_PRP, fs=12, sfs=8.5)
    _arrow(ax, 6.0, 5.7, 6.3, 5.7, color=_PRP, lw=2)
    ax.text(6.15, 5.95, "dispatch()", ha="center", va="bottom", fontsize=8, color=_PRP)

    _fbox(ax, 10.1, 4.5, 3.0, 2.4,
          "Downstream\nStage  (s2)",
          "receives candidates\nin priority order\n(highest score first)\nvia _pending_candidates",
          fc=_BLUE, fs=12, sfs=9)
    _arrow(ax, 9.8, 5.7, 10.1, 5.7, color=_NAVY, lw=2)
    ax.text(9.95, 5.95, "priority-ranked\nreplicas", ha="center", va="bottom",
            fontsize=7.5, color=_NAVY)

    _fbox(ax, 6.3, 2.3, 3.5, 1.9,
          "Adaptive Batch Sizing",
          "target_size = base × BP_multiplier\nstratify=soft: tail dispatch allowed\nstratify=strict: hold until full\nshard bandit learns multiplier",
          fc="#006064", fs=11, sfs=8)

    ax.add_patch(FancyBboxPatch((0.2, 0.3), 5.6, 3.8,
                                boxstyle="round,pad=0.1",
                                facecolor="#FFF9C4", edgecolor="#F57F17", linewidth=2, zorder=2))
    ax.text(3.0, 3.8, "BackpressureNegotiator", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#E65100", zorder=5)

    states = [("HOLD\n(normal)", 1.0, 2.6, "#43A047"),
              ("THROTTLE\n(queue too deep)", 3.0, 2.6, "#E53935"),
              ("WIDEN\n(queue drained)", 5.0, 2.6, "#1E88E5")]
    for lbl, x, y, c in states:
        ax.add_patch(plt.Circle((x, y), 0.55, color=c, zorder=4))
        ax.text(x, y, lbl, ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="white", zorder=5)

    for (x0, y0), (x1, y1), lbl, c in [
        ((1.55, 2.85), (2.45, 2.85), "q ≥ high_water", "#E53935"),
        ((3.55, 2.35), (4.45, 2.35), "q ≤ low_water",  "#1E88E5"),
        ((4.45, 2.85), (2.6,  2.85), "q in range → HOLD", "#43A047"),
    ]:
        _arrow(ax, x0, y0, x1, y1, color=c)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.18, lbl,
                ha="center", fontsize=7, color=c)

    for x, lbl in [(1.0, "dispatch\nnormal"), (3.0, "dispatch\n= 0"), (5.0, "dispatch\n× mult")]:
        ax.text(x, 1.8, lbl, ha="center", va="center", fontsize=7.5, color=_GRY,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", lw=0.8))

    ax.text(3.0, 0.6, "queue_depth = group.replicas − group.started_count",
            ha="center", fontsize=8, color=_GRY, style="italic")

    _arrow(ax, 5.8, 2.6, 6.3, 2.8, color="#F57F17", lw=1.5)
    ax.text(6.1, 2.85, "BP state\n→ multiplier", ha="center", fontsize=7.5, color="#E65100")

    ax.set_title("Sharder Module — Buffering, Priority Ranking, and Dispatch Control",
                 fontsize=13, fontweight="bold", color=_NAVY, pad=6)
    _diag_save(fig, path)


# ── Diagram 3: Thompson-Sampling Bandit ───────────────────────────────────────

def make_bandit_diagram(path: Path) -> None:
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor(_BG)

    fig.text(0.5, 0.97, "Scheduling Bandit — Thompson Sampling per Stage",
             ha="center", va="top", fontsize=14, fontweight="bold", color=_NAVY)

    arm_specs = [
        ("s1\nligand filter", 1, 1,  "#42a5f5", "Beta(1,1)\n(uniform prior)"),
        ("s2\nML affinity",   2, 1,  "#66bb6a", "Beta(2,1)"),
        ("s3\ndocking",       3, 1,  "#ffa726", "Beta(3,1)"),
        ("s4\nMD refine",     4, 1,  "#ef5350", "Beta(4,1)"),
        ("s5\nFEP rank",      5, 1,  "#ab47bc", "Beta(5,1)\n(warm-start: prefers s5)"),
    ]
    x_positions = np.linspace(0.06, 0.88, 5)
    ax_width, ax_height = 0.155, 0.26
    ax_y = 0.65

    xs = np.linspace(0.001, 0.999, 300)
    for i, (stage_lbl, a, b, color, prior_lbl) in enumerate(arm_specs):
        ax = fig.add_axes([x_positions[i], ax_y, ax_width, ax_height])
        log_pdf = (a - 1) * np.log(xs) + (b - 1) * np.log(1 - xs)
        pdf = np.exp(log_pdf - log_pdf.max())
        pdf = pdf / (np.trapz(pdf, xs) if hasattr(np, "trapz") else np.trapezoid(pdf, xs))
        ax.fill_between(xs, pdf, alpha=0.5, color=color)
        ax.plot(xs, pdf, color=color, linewidth=2)
        ax.set_xlim(0, 1); ax.set_ylim(0)
        ax.set_xlabel("θ (priority)", fontsize=7)
        ax.set_title(stage_lbl, fontsize=8, fontweight="bold", color=color, pad=2)
        ax.tick_params(labelsize=6)
        ax.text(0.5, ax.get_ylim()[1] * 0.75, prior_lbl,
                ha="center", fontsize=6.5, color=color, style="italic")
        sample_theta = a / (a + b)
        ax.axvline(sample_theta, color=color, linestyle="--", linewidth=1.5, alpha=0.8)
        ax.text(sample_theta, ax.get_ylim()[1] * 0.12, f"θ={sample_theta:.2f}",
                ha="center", fontsize=6, color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.8))

    main_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="none")
    main_ax.set_xlim(0, 14); main_ax.set_ylim(0, 9); main_ax.axis("off")

    for xi in x_positions:
        main_ax.annotate("", xy=(xi * 14, 5.7), xytext=(xi * 14, 5.95),
                         arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))

    _fbox(main_ax, 2.5, 4.95, 9.0, 0.65,
          "Sample θᵢ ~ Betaᵢ(αᵢ, βᵢ)  for each eligible stage",
          "Thompson sample — exploration/exploitation trade-off", fc="#37474F", fs=12, sfs=9)
    _arrow(main_ax, 7.0, 4.95, 7.0, 4.65, color="#555")

    _fbox(main_ax, 2.5, 4.0, 9.0, 0.65,
          "Rank stages by θ  →  highest θ gets next freed GPU slot",
          "deterministic tie-breaking by registration order", fc=_PRP, fs=12, sfs=9)
    _arrow(main_ax, 7.0, 4.0, 7.0, 3.7, color="#555")

    _fbox(main_ax, 2.5, 3.05, 9.0, 0.65,
          "Replica executes  →  on_replica_done  →  compute reward r ∈ [0, 1]",
          "", fc=_BLUE, fs=12)
    _arrow(main_ax, 7.0, 3.05, 7.0, 2.75, color="#555")

    reward_specs = [
        (1.8,  "THROTTLE\n(downstream queue full)", 0.2,  "#E53935"),
        (5.5,  "HOLD\n(queue healthy)",             "1 − 0.5×util", "#43A047"),
        (9.5,  "WIDEN\n(queue drained)",             0.8,  "#1E88E5"),
    ]
    for rx, lbl, r, c in reward_specs:
        main_ax.add_patch(FancyBboxPatch((rx - 1.5, 1.6), 3.0, 0.9,
                                         boxstyle="round,pad=0.08",
                                         facecolor=c, edgecolor="white", lw=1.5, zorder=3, alpha=0.9))
        main_ax.text(rx, 2.22, lbl, ha="center", va="center",
                     fontsize=9, fontweight="bold", color="white", zorder=4)
        main_ax.text(rx, 1.85, f"r = {r}", ha="center", va="center",
                     fontsize=9, color="white", style="italic", zorder=4)
        _arrow(main_ax, rx, 2.75, rx, 2.5, color=c)

    _fbox(main_ax, 2.5, 0.85, 9.0, 0.65,
          "Bayesian update:  αᵢ ← αᵢ + r    βᵢ ← βᵢ + (1 − r)",
          "positive reward → arm shifts right (higher priority in next sample)", fc=_GRN, fs=12, sfs=9)

    for rx in [1.8, 5.5, 9.5]:
        _arrow(main_ax, rx, 1.6, rx, 1.5, color="#888")
        _arrow(main_ax, rx, 1.5, 7.0, 1.5, color="#888")
    _arrow(main_ax, 7.0, 1.5, 7.0, 0.85, color="#888")

    main_ax.annotate("", xy=(0.5, 6.4), xytext=(0.5, 0.85),
                     arrowprops=dict(arrowstyle="->", color=_GRN, lw=2,
                                     connectionstyle="arc3,rad=0.0"))
    main_ax.text(0.18, 3.5, "update\nposterior", ha="center", va="center",
                 fontsize=9, color=_GRN, fontweight="bold", rotation=90)

    _diag_save(fig, path)


# ── Diagram 4: Candidate Profiles ─────────────────────────────────────────────

def make_profiles_diagram(path: Path) -> None:
    fig, (ax_heat, ax_desc) = plt.subplots(1, 2, figsize=(14, 7),
                                            gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor(_BG)
    ax_heat.set_facecolor(_BG)
    ax_desc.set_facecolor(_BG)

    profiles = ["pure_promise", "active_learning", "explore_exploit",
                "diverse_top", "round_robin"]
    weights = np.array([
        [1.0, 0.0, 0.0, 0.05, 0.0],
        [0.0, 0.0, 1.0, 0.05, 0.0],
        [0.5, 0.3, 0.4, 0.05, 0.0],
        [0.6, 0.0, 0.1, 0.05, 0.3],
        [0.0, 0.0, 0.0, 0.05, 1.0],
    ])
    signal_names = ["Score", "Surrogate", "Uncertainty", "Age", "Diversity"]
    row_colors   = ["#4caf50", "#9c27b0", "#00838f", "#f44336", "#78909c"]

    im = ax_heat.imshow(weights, cmap="YlGn", vmin=0, vmax=1.0, aspect="auto")
    ax_heat.set_xticks(range(5))
    ax_heat.set_xticklabels(signal_names, fontsize=11, fontweight="bold", color=_NAVY)
    ax_heat.set_yticks(range(5))
    ax_heat.set_yticklabels(profiles, fontsize=10.5, fontweight="bold")

    for tick_lbl, col in zip(ax_heat.get_yticklabels(), row_colors):
        tick_lbl.set_color(col)

    for i in range(5):
        for j in range(5):
            v = weights[i, j]
            txt_col = "white" if v > 0.55 else ("black" if v > 0.15 else "#aaaaaa")
            ax_heat.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=12, fontweight="bold", color=txt_col)

    ax_heat.set_title("Weight Matrix  (greener = higher weight)",
                      fontsize=12, fontweight="bold", color=_NAVY, pad=10)
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    ax_desc.axis("off")
    ax_desc.set_xlim(0, 1); ax_desc.set_ylim(0, 1)
    ax_desc.set_title("Use Cases", fontsize=12, fontweight="bold", color=_NAVY, pad=10)

    descs = [
        ("pure_promise",    "#4caf50",
         "Greedy quality routing.\nRanks purely by upstream score.\nTiny age bonus prevents starvation.\nBest when scores are reliable."),
        ("active_learning", "#9c27b0",
         "Uncertainty-first dispatch.\nMaximises surrogate model learning.\nSacrifices short-term quality.\nBest early in a campaign."),
        ("explore_exploit", "#00838f",
         "Balanced exploration.\nScore + surrogate + uncertainty.\nGood with a calibrated model.\nDefault for most campaigns."),
        ("diverse_top",     "#f44336",
         "Quality + chemical diversity.\nScaffold novelty bonus via MMR.\nPrevents chemical echo chambers.\nBest for diverse leads."),
        ("round_robin",     "#78909c",
         "Pure diversity mandate.\nRound-robin across scaffold classes.\nIgnores quality entirely.\nInitial screening / mandate."),
    ]
    for i, (name, col, desc) in enumerate(descs):
        y_top = 0.94 - i * 0.195
        ax_desc.add_patch(FancyBboxPatch((0.01, y_top - 0.15), 0.98, 0.16,
                                         boxstyle="round,pad=0.01",
                                         facecolor=col, alpha=0.12,
                                         edgecolor=col, linewidth=1.5))
        ax_desc.text(0.04, y_top, name, fontsize=10, fontweight="bold", color=col, va="top")
        ax_desc.text(0.04, y_top - 0.04, desc, fontsize=8.5, color=_GRY, va="top",
                     style="italic")

    plt.tight_layout(pad=2.5)
    _diag_save(fig, path)


# ── Diagram 5: Planner / Execution Model ──────────────────────────────────────

def make_planner_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.5); ax.axis("off")

    # ── Column 1: User inputs ────────────────────────────────────────────────
    ax.text(1.95, 8.3, "User Inputs", ha="center", fontsize=12,
            fontweight="bold", color=_NAVY)

    _fbox(ax, 0.2, 6.0, 3.5, 2.1,
          "Campaign Config  (YAML)",
          "workflows:\n  s1: replicas=10000, priority=10\n  s2: dependencies=[s1]\n       min_replicas=1\n  ...",
          fc=_BLUE, fs=11, sfs=8.5)

    _fbox(ax, 0.2, 3.5, 3.5, 2.2,
          "Workflow Classes",
          "class SimWorkflow(BaseWorkflow):\n  async def run(self, rid):\n    await do_work()\n    await self._signal_done()",
          fc=_GRY, fs=10, sfs=8)

    _fbox(ax, 0.2, 1.2, 3.5, 2.0,
          "Resource Spec",
          "engine: concurrent  # or dragon\ntotal_gpus: 4\ntotal_cpus: 128\nfeatures:\n  sharder: true",
          fc="#455A64", fs=10, sfs=8.5)

    # ── Arrows → CM ──────────────────────────────────────────────────────────
    for y_mid in [7.05, 4.6, 2.2]:
        _arrow(ax, 3.7, y_mid, 4.5, y_mid, color=_NAVY, lw=2)
    ax.text(4.1, 4.9, "from_config()", ha="center", fontsize=8.5,
            color=_NAVY, style="italic",
            bbox=dict(boxstyle="round,pad=0.2", fc=_BG, ec=_NAVY, lw=0.8))

    # ── Column 2: AsyncCampaignManager ───────────────────────────────────────
    ax.add_patch(FancyBboxPatch((4.5, 0.5), 5.2, 7.75,
                                boxstyle="round,pad=0.15",
                                facecolor="#E3F2FD", edgecolor=_NAVY,
                                linewidth=2.5, zorder=2))
    ax.text(7.1, 8.1, "AsyncCampaignManager", ha="center",
            fontsize=13, fontweight="bold", color=_NAVY)

    # Dependency graph box
    ax.add_patch(FancyBboxPatch((4.7, 5.65), 4.8, 2.4,
                                boxstyle="round,pad=0.1",
                                facecolor="#BBDEFB", edgecolor=_BLUE, linewidth=1.5, zorder=3))
    ax.text(7.1, 7.85, "Dependency Graph", ha="center",
            fontsize=9, fontweight="bold", color=_BLUE, zorder=4)

    stage_cols = ["#42a5f5", "#66bb6a", "#ffa726", "#ef5350", "#ab47bc"]
    stage_names = ["s1", "s2", "s3", "s4", "s5"]
    for i, (sn, sc) in enumerate(zip(stage_names, stage_cols)):
        nx = 5.1 + i * 0.95
        ax.add_patch(plt.Circle((nx, 6.85), 0.32, color=sc, zorder=5))
        ax.text(nx, 6.85, sn, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=6)
        if i < 4:
            _arrow(ax, nx + 0.32, 6.85, nx + 0.63, 6.85, color=_GRY, lw=1.5)
    ax.text(7.1, 6.2, "group.status: eligible → scheduling → running → done",
            ha="center", fontsize=7.5, color=_GRY, zorder=4, style="italic")

    # Scheduler box
    _fbox(ax, 4.7, 3.85, 4.8, 1.7,
          "Scheduler (per state change)",
          "1. Flush sharder buffers + refresh BP\n"
          "2. Collect eligible groups\n"
          "3. Pass 1: guarantee min_replicas\n"
          "4. Pass 2: fill to max_replicas",
          fc=_GRN, fs=10, sfs=8.5)

    # Optional features box
    _fbox(ax, 4.7, 2.15, 4.8, 1.5,
          "Optional Features (feature flags)",
          "sharder=true  ·  backpressure=true\nbandit=true  ·  monitor=true\nAll disabled by default",
          fc=_PRP, fs=10, sfs=8.5)

    # Metrics box
    _fbox(ax, 4.7, 0.65, 4.8, 1.3,
          "CampaignMetrics",
          "replica_events  ·  scheduling_events\nbp_fractions  ·  shard_events",
          fc=_GRY, fs=9.5, sfs=8)

    # ── Arrow → asyncflow ────────────────────────────────────────────────────
    _arrow(ax, 9.7, 4.5, 10.4, 4.5, color=_GRN, lw=2.5)
    ax.text(10.05, 4.85, "create_task()", ha="center", fontsize=8.5,
            color=_GRN, style="italic",
            bbox=dict(boxstyle="round,pad=0.2", fc=_BG, ec=_GRN, lw=0.8))

    # ── Column 3: asyncflow Engine ───────────────────────────────────────────
    ax.text(12.0, 8.3, "Execution Engine", ha="center", fontsize=12,
            fontweight="bold", color=_NAVY)

    _fbox(ax, 10.4, 5.5, 3.3, 2.7,
          "asyncflow\nWorkflowEngine",
          "ConcurrentBackend\n(asyncio — local)\n── or ──\nDragonBackend\n(HPC multi-node)",
          fc=_ORG, fs=12, sfs=9)

    _fbox(ax, 10.4, 3.1, 3.3, 2.2,
          "Running Replicas",
          "SimWorkflow.run(replica_0)\nSimWorkflow.run(replica_1)\n...\n(up to max_replicas concurrent)",
          fc=_GRY, fs=10, sfs=8)

    _arrow(ax, 12.05, 5.5, 12.05, 5.3, color=_GRY, lw=2)

    _fbox(ax, 10.4, 1.2, 3.3, 1.75,
          "Results",
          "on_replica_done() callbacks\n_signal_done() / _trigger_dependent()\nCampaignMetrics updated",
          fc=_BLUE, fs=10, sfs=8.5)

    _arrow(ax, 12.05, 3.1, 12.05, 2.95, color=_GRY, lw=2)

    # Feedback arrow
    ax.annotate("", xy=(7.1, 3.85), xytext=(10.4, 1.8),
                arrowprops=dict(arrowstyle="->", color=_BLUE, lw=1.8,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(9.5, 2.6, "signal / trigger\nstate update", ha="center", fontsize=8,
            color=_BLUE, style="italic")

    ax.set_title("SPHERICAL — From Config to Execution  (asyncio event-loop model)",
                 fontsize=14, fontweight="bold", color=_NAVY, pad=6)
    _diag_save(fig, path)


def generate_diagrams() -> Path:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    print("  Generating architecture diagrams...")
    make_cm_arch_diagram(DIAG_DIR / "cm_architecture.png")
    print("    cm_architecture.png")
    make_sharder_diagram(DIAG_DIR / "sharder.png")
    print("    sharder.png")
    make_bandit_diagram(DIAG_DIR / "bandit.png")
    print("    bandit.png")
    make_profiles_diagram(DIAG_DIR / "profiles.png")
    print("    profiles.png")
    make_planner_diagram(DIAG_DIR / "planner.png")
    print("    planner.png")
    return DIAG_DIR


# ── Assemble ──────────────────────────────────────────────────────────────────

def build(out_path: str) -> None:
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H

    diag_dir = generate_diagrams()

    print("Building slides...")
    slide_title(prs);                              print("  1. Title")
    slide_pipeline_overview(prs);                  print("  2. Pipeline overview")
    slide_spherical_architecture(prs, diag_dir);   print("  3. SPHERICAL architecture")
    slide_stage_profiles(prs, diag_dir);           print("  4. Stage profiles")
    slide_benchmark_design(prs);                   print("  5. Benchmark design")
    slide_cascade_funnel(prs);                     print("  6. Cascade funnel")
    slide_main_result(prs);                        print("  7. Main result (wall time)")
    slide_sharding_bp(prs, diag_dir);              print("  8. Optimisation 1 — Sharder+BP+Bandit")
    slide_bandit(prs);                             print("  9. Optimisation 2 — Scheduling bandit")
    slide_all_opt(prs);                            print(" 10. Optimisation 3 — Combined")
    slide_gantt(prs);                              print(" 11. Pipeline Gantt")
    slide_planner_execution(prs, diag_dir);        print(" 12. Planner & execution model")
    slide_methodology(prs);                        print(" 13. Methodology / caveats")
    slide_summary(prs);                            print(" 14. Summary")

    prs.save(out_path)
    print(f"\nSaved: {out_path}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="spherical_benchmark.pptx")
    args = parser.parse_args()
    build(args.out)
