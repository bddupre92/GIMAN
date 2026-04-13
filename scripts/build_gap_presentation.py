#!/usr/bin/env python3
"""Build GAP Milestone Committee Presentation for Blair Dupre."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/gap_milestone/GAP_Committee_Presentation_Blair_Dupre.pptx"
FIG = REPO / "outputs/mechanistic_twin/phase2/figures"
P1_FIG = REPO / "outputs/paper1_figures"
P3_FIG = REPO / "outputs/paper3_figures"

# Colors
NAVY = RGBColor(0x0A, 0x1F, 0x44)
TEAL = RGBColor(0x00, 0x96, 0x88)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xF5, 0xF5, 0xF5)
DARK_GRAY = RGBColor(0x33, 0x33, 0x33)
GOLD = RGBColor(0xFF, 0xB3, 0x00)
LIGHT_NAVY = RGBColor(0x1A, 0x3A, 0x6B)
GREEN = RGBColor(0x00, 0xC8, 0x53)
RED_ACCENT = RGBColor(0xE5, 0x39, 0x35)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
W = prs.slide_width
H = prs.slide_height


def add_bg(slide, color=NAVY):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text_box(slide, left, top, width, height, text, font_size=18,
                 color=WHITE, bold=False, alignment=PP_ALIGN.LEFT, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_bullet_slide(slide, items, left=0.8, top=2.0, width=5.5, font_size=16,
                     color=WHITE, spacing=Pt(8)):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(5))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = "Calibri"
        p.space_after = spacing
        p.level = 0
    return txBox


def add_image_safe(slide, path, left, top, width=None, height=None):
    p = Path(path)
    if p.exists():
        kwargs = {"left": Inches(left), "top": Inches(top)}
        if width:
            kwargs["width"] = Inches(width)
        if height:
            kwargs["height"] = Inches(height)
        slide.shapes.add_picture(str(p), **kwargs)
        return True
    return False


# ============================================================================
# SLIDE 1: Title
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
add_bg(slide, NAVY)
add_text_box(slide, 1.0, 1.5, 11, 1.5,
             "From Data-Driven to Mechanistic:",
             font_size=36, color=GOLD, bold=True, alignment=PP_ALIGN.CENTER)
add_text_box(slide, 1.0, 2.8, 11, 1.5,
             "A Computational Framework for NSD-ISS\nBiological Stage Prediction in Parkinson's Disease",
             font_size=28, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
add_text_box(slide, 1.0, 4.8, 11, 0.5,
             "Blair Dupre  |  PhD Candidate  |  School of Computing & Informatics",
             font_size=18, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)
add_text_box(slide, 1.0, 5.4, 11, 0.5,
             "University of Louisiana at Lafayette  |  GAP Milestone Committee Meeting  |  Spring 2026",
             font_size=16, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)

# ============================================================================
# SLIDE 2: The Problem
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, NAVY)
add_text_box(slide, 0.8, 0.4, 11, 0.8,
             "The Problem: Parkinson's Has No Predictive Framework",
             font_size=32, color=GOLD, bold=True)
add_bullet_slide(slide, [
    "• 1 million Americans affected, $52B annual burden (Yang et al., 2020)",
    "• NSD-ISS (Simuni et al., Lancet Neurol 2024) — first biological staging",
    "  framework using α-synuclein + DaT-SPECT imaging",
    "• But NSD-ISS is retrospective: tells WHERE patient is, not WHERE they're GOING",
    "• No computational tools to predict transitions or simulate treatments",
    "• Our systematic review (354 papers, 15 included): 0% implement",
    "  mechanistic digital twin approaches for PD",
], top=1.6, font_size=20, color=WHITE)
add_text_box(slide, 0.8, 6.2, 11, 0.8,
             "Gap: Clinicians need personalized, uncertainty-aware, mechanistic prognostic tools",
             font_size=20, color=GOLD, bold=True)

# ============================================================================
# SLIDE 3: The Vision — What I Want to Do
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, LIGHT_NAVY)
add_text_box(slide, 0.8, 0.4, 11, 0.8,
             "THE VISION: What I Want to Build",
             font_size=32, color=GOLD, bold=True)
add_text_box(slide, 0.8, 1.3, 5.5, 0.5,
             "Two Complementary Approaches:",
             font_size=22, color=WHITE, bold=True)
# Left column: Data-driven
add_text_box(slide, 0.8, 2.0, 5.5, 0.5,
             "1. Data-Driven Pipeline (Papers 1-6)",
             font_size=20, color=TEAL, bold=True)
add_bullet_slide(slide, [
    "  Stage prediction (Paper 1)",
    "  Missing data imputation (Paper 2)",
    "  Transition timing (Paper 3)",
    "  Uncertainty quantification (Paper 4)",
    "  Temporal validation (Paper 5)",
    "  Clinical decision support (Paper 6)",
], left=0.8, top=2.6, width=5.5, font_size=16, color=LIGHT_GRAY)
# Right column: Mechanistic
add_text_box(slide, 7.0, 2.0, 5.5, 0.5,
             "2. Mechanistic Twin (Paper 7)",
             font_size=20, color=GOLD, bold=True)
add_bullet_slide(slide, [
    "  α-Synuclein aggregation ODE",
    "  Dopaminergic neuron death",
    "  Per-patient Bayesian calibration",
    "  Counterfactual drug simulation",
    "  \"What if prasinezumab at month 6?\"",
    "  → Causal, not just correlational",
], left=7.0, top=2.6, width=5.5, font_size=16, color=LIGHT_GRAY)
add_text_box(slide, 0.8, 5.8, 11, 1.0,
             "Together: the first framework that can PREDICT stages, QUANTIFY uncertainty,\n"
             "SIMULATE treatments, and EXPLAIN biological mechanisms in Parkinson's disease",
             font_size=20, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

# ============================================================================
# SLIDE 4: The 5-Module ODE Architecture
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, NAVY)
add_text_box(slide, 0.8, 0.4, 11, 0.8,
             "Mechanistic Twin: 5-Module ODE Architecture",
             font_size=30, color=GOLD, bold=True)
modules = [
    ("Module 1: α-Syn Aggregation", "Monomer → Oligomer → Fibril\nCohen/Knowles kinetics", TEAL),
    ("Module 2: Neuron Death", "Oligomer toxicity → SBR decline\nCalibrated from DaT-SPECT", GREEN),
    ("Module 3: Connectome", "Network propagation of\nα-syn across brain regions", RGBColor(0x42, 0xA5, 0xF5)),
    ("Module 4: PK/PD", "Levodopa pharmacokinetics\n→ motor response", RGBColor(0xAB, 0x47, 0xBC)),
    ("Module 5: Functional", "Neuron count → NSD-ISS\nstage mapping", RGBColor(0xFF, 0x70, 0x43)),
]
for i, (title, desc, color) in enumerate(modules):
    left = 0.5 + i * 2.5
    # Box
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    Inches(left), Inches(1.6), Inches(2.3), Inches(2.5))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    # Title
    add_text_box(slide, left + 0.1, 1.7, 2.1, 0.6, title, font_size=14, color=WHITE, bold=True)
    add_text_box(slide, left + 0.1, 2.4, 2.1, 1.5, desc, font_size=12, color=WHITE)

add_text_box(slide, 0.8, 4.5, 11, 0.5,
             "Phase 1 ✓  |  Phase 2 (Modules 1-2) ✓  |  Phases 3-5: Future Work",
             font_size=18, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)
add_text_box(slide, 0.8, 5.2, 11, 1.5,
             "Key insight: Each drug class acts on a SPECIFIC reaction in the ODE.\n"
             "Prasinezumab → reduces oligomer flux (T_tox). Small molecules → reduce k_n.\n"
             "This is why phenomenological models (Fisher-Kolmogorov) were REJECTED.",
             font_size=16, color=LIGHT_GRAY)

# ============================================================================
# SLIDE 5: The Coupled ODE (Variant B)
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, NAVY)
add_text_box(slide, 0.8, 0.4, 11, 0.8,
             "The Coupled 4-State ODE (Variant B, Mass-Conserving)",
             font_size=28, color=GOLD, bold=True)
eqs = [
    "dM/dt = k_prod − k_clear^M · M − k_n · M² − k_e · M · F",
    "dO/dt = k_n · M² − k_conv · O − k_clear^O · O",
    "dF/dt = k_conv · O − k_clear^F · F       ← Variant B fix (removed k_frag·F)",
    "d(log N)/dt = −(α_tox · O + k_age)      ← log transform = positivity guaranteed",
]
for i, eq in enumerate(eqs):
    add_text_box(slide, 1.5, 1.5 + i * 0.7, 10, 0.6, eq,
                 font_size=18, color=WHITE, font_name="Consolas")

add_text_box(slide, 0.8, 4.5, 11, 0.5,
             "Under slow-fast collapse → closed-form SBR decay (~3ms per solve):",
             font_size=18, color=TEAL, bold=True)
add_text_box(slide, 1.5, 5.1, 10, 0.6,
             "SBR(t) = SBR₀ · exp(−γ · T_tox · t)     where T_tox = α_tox · k_n · M²_ss / (k_conv + k_clear^O)",
             font_size=18, color=GOLD, bold=True, font_name="Consolas")
add_text_box(slide, 0.8, 5.9, 11, 1.0,
             "Structural identifiability: PROVEN (StructuralIdentifiability.jl + SIAN.jl cross-validated)\n"
             "Practical identifiability: T_tox is the stiff direction (Gutenkunst/Transtrum sloppy models)",
             font_size=15, color=LIGHT_GRAY)

# ============================================================================
# SLIDE 6: Papers 1-6 Results Summary
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, LIGHT_NAVY)
add_text_box(slide, 0.8, 0.4, 11, 0.8,
             "COMPLETED: Data-Driven Foundation (Papers 1-6)",
             font_size=30, color=GOLD, bold=True)
rows = [
    ("Paper 1", "NSD-ISS Stage Prediction", "CatBoost AUC 0.981 (binary)", "✓"),
    ("Paper 2", "GIMIN Imputation", "22% lower RMSE vs MissForest", "✓"),
    ("Paper 3", "Transition Timing", "Graph-DT C-td 0.922, 28% less variance", "✓"),
    ("Paper 4", "Conformal Survival", "91.3% coverage, 2.6× narrower bands", "✓"),
    ("Paper 5", "Temporal Validation", "9% realistic degradation quantified", "✓"),
    ("Paper 6", "Clinical Decision Support", "Unified pipeline demonstrated", "✓"),
]
y = 1.5
for paper, title, result, status in rows:
    add_text_box(slide, 0.8, y, 1.5, 0.45, paper, font_size=16, color=TEAL, bold=True)
    add_text_box(slide, 2.3, y, 4.0, 0.45, title, font_size=16, color=WHITE)
    add_text_box(slide, 6.5, y, 5.0, 0.45, result, font_size=15, color=GOLD)
    add_text_box(slide, 12.0, y, 0.8, 0.45, status, font_size=18, color=GREEN, bold=True)
    y += 0.65

add_text_box(slide, 0.8, 5.8, 11, 1.0,
             "Data: PPMI (n=2,201) + BioFIND + PDBP + HBS external validation\n"
             "All code, checkpoints, and results archived with full reproducibility",
             font_size=16, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)

# ============================================================================
# SLIDE 7: Paper 7 — Headline Results
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, NAVY)
add_text_box(slide, 0.8, 0.4, 11, 0.8,
             "Paper 7: Mechanistic Twin — Key Results",
             font_size=30, color=GOLD, bold=True)
results = [
    ("304 patients", "Per-patient Bayesian IS posterior from serial DaT-SPECT"),
    ("3.29%/yr", "Cohort-median neuron loss (Fearnley & Lees 2-5%/yr range)"),
    ("277 CSF patients", "Joint SBR + CSF α-syn calibration breaks degeneracy"),
    ("cor → −0.113", "k_n−α_tox sloppy ridge partially broken (was −0.240)"),
    ("k_n 29% tighter", "CSF constrains nucleation rate independent of toxicity"),
    ("99.0% LOO", "Leave-future-out coverage (Bürkner et al. 2019 methodology)"),
    ("+1.24 yr delay", "Prasinezumab counterfactual consistent with PASADENA trial"),
]
y = 1.4
for number, desc in results:
    add_text_box(slide, 0.8, y, 2.5, 0.45, number, font_size=20, color=GOLD, bold=True)
    add_text_box(slide, 3.5, y, 9.0, 0.45, desc, font_size=17, color=WHITE)
    y += 0.6

add_text_box(slide, 0.8, 6.2, 11, 0.8,
             "First per-patient Bayesian posterior over α-syn nucleation + neuron death coupling\n"
             "from joint longitudinal DaT-SPECT + CSF α-synuclein with formal identifiability proof",
             font_size=16, color=TEAL, bold=True, alignment=PP_ALIGN.CENTER)

# ============================================================================
# SLIDE 8: CSF Degeneracy Breaking (with figure)
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, LIGHT_NAVY)
add_text_box(slide, 0.8, 0.3, 11, 0.6,
             "CSF α-Synuclein Breaks the Sloppy-Ridge Degeneracy",
             font_size=28, color=GOLD, bold=True)
add_image_safe(slide, FIG / "fig3_degeneracy_breaking.png", 0.3, 1.1, width=12.5)

# ============================================================================
# SLIDE 9: Counterfactual (with figure)
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, NAVY)
add_text_box(slide, 0.8, 0.3, 11, 0.6,
             "Prasinezumab Counterfactual — Validated Against PASADENA Trial",
             font_size=26, color=GOLD, bold=True)
add_image_safe(slide, FIG / "fig5_counterfactual.png", 0.3, 1.1, width=12.5)
add_text_box(slide, 0.8, 5.8, 11, 1.2,
             "PASADENA DaT-SPECT endpoint was NULL (Pagano et al. 2022 NEJM). Our twin predicts\n"
             "d ≈ 0.055 effect size — below detection at N=105/arm, 52 weeks. Correct prediction of absence.",
             font_size=15, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)

# ============================================================================
# SLIDE 10: T_tox and Neuron Loss (with figure)
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, LIGHT_NAVY)
add_text_box(slide, 0.8, 0.3, 11, 0.6,
             "Per-Patient Toxicity Flux — Biological Validation",
             font_size=28, color=GOLD, bold=True)
add_image_safe(slide, FIG / "fig2_ttox_neuron_loss.png", 0.3, 1.1, width=12.5)

# ============================================================================
# SLIDE 11: LOO Validation
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, NAVY)
add_text_box(slide, 0.8, 0.4, 11, 0.8,
             "Strict Leave-Future-Out Validation (Bürkner et al. 2019)",
             font_size=28, color=GOLD, bold=True)
# Comparison table
headers = ["Metric", "Training PPC", "LOO Forward", "Phase 1 LOO", "Ideal"]
data = [
    ["95% PI Coverage", "99.5%", "99.0%", "93.75%", "95%"],
    ["50% PI Coverage", "80.5%", "56.9%", "—", "50%"],
    ["Median |z-score|", "0.442", "0.707", "—", "1.0"],
]
y_start = 1.8
for col_i, header in enumerate(headers):
    x = 1.0 + col_i * 2.3
    add_text_box(slide, x, y_start, 2.2, 0.5, header, font_size=16, color=TEAL, bold=True)
for row_i, row in enumerate(data):
    for col_i, val in enumerate(row):
        x = 1.0 + col_i * 2.3
        color = GOLD if col_i == 2 else WHITE
        add_text_box(slide, x, y_start + 0.6 + row_i * 0.55, 2.2, 0.5,
                     val, font_size=18, color=color, bold=(col_i == 2))

add_bullet_slide(slide, [
    "• LOO barely drops from training (−0.5pp) → model generalizes, not overfitting",
    "• 50% PI dramatically better calibrated (56.9% vs 80.5% → closer to ideal 50%)",
    "• Phase 2 LOO EXCEEDS Phase 1 LOO by +5.3pp — mechanistic model forecasts better",
    "• Method: Bürkner et al. 2019 J. Stat. Comp. Sim. (90 citations) — established, not invented",
], left=0.8, top=4.2, width=11, font_size=16, color=LIGHT_GRAY)

# ============================================================================
# SLIDE 12: Literature Validation
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, LIGHT_NAVY)
add_text_box(slide, 0.8, 0.4, 11, 0.8,
             "Literature Validation — Not Scooped, Properly Anchored",
             font_size=28, color=GOLD, bold=True)
validations = [
    ("Novelty claim", "0/58 papers in Consensus MCP combine DaT-SPECT + CSF α-syn in Bayesian ODE"),
    ("Neuron loss rate", "3.29%/yr matches Fearnley & Lees 1991 canonical 2-5%/yr (prior disclosed)"),
    ("CSF stationarity", "Mollenhauer et al. 2017 (PPMI n=173): stable over 12-36 months"),
    ("r_o = 1.0", "Koehler et al. 2008: equimolar oligomer detection in total ELISA"),
    ("PASADENA null", "Pagano et al. 2022 NEJM: no DaT-SPECT effect at 52 weeks"),
    ("LOO methodology", "Bürkner et al. 2019 (90 cit), Vehtari et al. 2017 (4,425 cit)"),
    ("Competitors", "Hemedan 2026 (clinical scores), Denaro 2024 (framework only), Righetti 2025 (in vitro)"),
]
y = 1.5
for claim, evidence in validations:
    add_text_box(slide, 0.8, y, 2.8, 0.45, claim, font_size=15, color=TEAL, bold=True)
    add_text_box(slide, 3.8, y, 8.5, 0.45, evidence, font_size=14, color=WHITE)
    y += 0.6

add_text_box(slide, 0.8, 6.0, 11, 0.8,
             "Closed-loop methodology: every claim validated via Consensus MCP + PubMed\n"
             "before commitment to any manuscript or presentation (v1.1, locked 2026-04-10)",
             font_size=14, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)

# ============================================================================
# SLIDE 13: Timeline & Next Steps
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, NAVY)
add_text_box(slide, 0.8, 0.4, 11, 0.8,
             "Timeline & Next Steps",
             font_size=32, color=GOLD, bold=True)

milestones = [
    ("✅ Complete", "Papers 1-6 data-driven framework", "Feb 2026"),
    ("✅ Complete", "Paper 7 mechanistic twin (Blocks 1-5, 7)", "Apr 2026"),
    ("✅ Draft", "bioRxiv preprint (11pp, peer-reviewed)", "Apr 2026"),
    ("🔄 Next", "Block 6: Wave B recovery (156 patients)", "May 2026"),
    ("🔄 Next", "CPT:PSP journal submission", "May 2026"),
    ("📋 Future", "Phase 3: Connectome propagation (HCP validated)", "Fall 2026"),
    ("📋 Future", "Phase 4: LEDD-as-covariate PK/PD", "Fall 2026"),
    ("🎓 Target", "Dissertation defense", "Spring 2027"),
]
y = 1.5
for status, desc, date in milestones:
    c = GREEN if "✅" in status else (GOLD if "🔄" in status else LIGHT_GRAY)
    add_text_box(slide, 0.8, y, 1.8, 0.45, status, font_size=16, color=c, bold=True)
    add_text_box(slide, 2.8, y, 7.5, 0.45, desc, font_size=17, color=WHITE)
    add_text_box(slide, 10.5, y, 2.5, 0.45, date, font_size=16, color=c, bold=True)
    y += 0.6

# ============================================================================
# SLIDE 14: Committee Ask
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, LIGHT_NAVY)
add_text_box(slide, 1.0, 1.5, 11, 1.2,
             "Committee Feedback Requested",
             font_size=36, color=GOLD, bold=True, alignment=PP_ALIGN.CENTER)
add_bullet_slide(slide, [
    "1. Is the 7-paper scope (6 data-driven + 1 mechanistic) appropriate for the PhD?",
    "2. Should the mechanistic twin (Paper 7) be a separate chapter or appendix?",
    "3. Is the narrow novelty claim defensible against the competitive landscape?",
    "4. Target journals: CPT:PSP for Paper 7, IEEE JBHI for Papers 3-4?",
    "5. Timeline feasibility: defense Spring 2027?",
], left=1.0, top=3.0, width=11, font_size=20, color=WHITE, spacing=Pt(14))

# ============================================================================
# SLIDE 15: Thank You
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, NAVY)
add_text_box(slide, 1.0, 2.0, 11, 1.2,
             "Thank You",
             font_size=48, color=GOLD, bold=True, alignment=PP_ALIGN.CENTER)
add_text_box(slide, 1.0, 3.5, 11, 0.8,
             "Blair Dupre  |  dupre.blair92@gmail.com",
             font_size=22, color=WHITE, alignment=PP_ALIGN.CENTER)
add_text_box(slide, 1.0, 4.3, 11, 1.5,
             "Data: PPMI (ppmi-info.org)  |  Code: Available upon publication\n"
             "206 bibliography entries  |  Full reproducibility with provenance tracking\n"
             "Dissertation: 7 papers, 11 chapters, Appendix D (5-module architecture)",
             font_size=16, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)

# Save
prs.save(str(OUT))
print(f"Saved: {OUT}")
print(f"Slides: {len(prs.slides)}")
