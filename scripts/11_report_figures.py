"""Step 11 - figures and diagrams for the written report (reports/report_figures/*.png).

Plots only: every number is read from the pipeline's report files (reports/*.json). The two test
breakdowns at the end (by source and by true grade) come from docs/RESULTS.md section 10, which was
computed from the saved test scores of the deployed router; they are marked as such below.
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
DECK = os.getenv("DECK") == "1"          # DECK=1: blue palette + Times New Roman, for the slides
MONO = not DECK                          # report figures: black, white and grey only, Times New Roman
OUT = REP / "report_figures" / ("deck" if DECK else "")
OUT.mkdir(parents=True, exist_ok=True)

# ---- one colour system for every figure -------------------------------------------------------
INK, MUTED, GRID, PAPER = "#13203B", "#5B6478", "#E6E9F0", "#FFFFFF"
CHEAP, STRONG, AMBER, CORAL, SKY, SLATE = "#0FA58A", "#6D3FE0", "#F2A007", "#EF4F6B", "#1C8FD9", "#94A0B8"
if DECK:
    INK, PAPER = "#262626", "#FFFFFF"
    CHEAP, STRONG, AMBER, CORAL, SKY, SLATE = "#3E9BD6", "#17436F", "#E8A317", "#D45A5A", "#366C9E", "#9AA6B8"
SOFT = {CHEAP: "#DCEEFA", STRONG: "#DCE5F0", AMBER: "#FBEFD3", CORAL: "#F8E0E0", SKY: "#DDE8F3", SLATE: "#ECEFF4"}
if MONO:
    INK, MUTED, GRID, PAPER = "#000000", "#333333", "#D0D0D0", "#FFFFFF"
    CHEAP, STRONG, AMBER, CORAL, SKY, SLATE = "#B8B8B8", "#1C1C1C", "#6F6F6F", "#4B4B4B", "#9B9B9B", "#7E7E7E"
    SOFT = {CHEAP: "#F3F3F3", STRONG: "#E2E2E2", AMBER: "#EBEBEB", CORAL: "#D4D4D4", SKY: "#FFFFFF", SLATE: "#FFFFFF"}

plt.rcParams.update({
    "font.family": ["Times New Roman"],
    "font.size": 10.5, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 14, "axes.labelcolor": MUTED, "axes.labelsize": 10.5, "axes.edgecolor": GRID,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.dpi": 220, "savefig.bbox": "tight",
    "legend.frameon": False,
})

S = json.loads((REP / "summary.json").read_text())
SWEEP = json.loads((REP / "finetune_sweep.json").read_text())["runs"]
R = S["router"]


def save(fig, name, subtitle=None):
    if subtitle:
        fig.text(0.0, 1.0, subtitle, ha="left", va="bottom", color=MUTED, fontsize=9.5,
                 transform=fig.axes[0].transAxes if fig.axes else fig.transFigure)
    if MONO and len(fig.axes) == 1:
        fig.axes[0].set_title("")           # the report gives every figure a caption instead
    fig.savefig(OUT / name)
    plt.close(fig)
    if MONO:                                # guarantee no colour at all in report figures
        from PIL import Image
        Image.open(OUT / name).convert("L").save(OUT / name)
    print("saved", name)


# ---- diagram helpers ----------------------------------------------------------------------------
def box(ax, x, y, w, h, title, body="", color=SKY, title_size=10.5, body_size=8.6, filled=False):
    face = color if filled else SOFT.get(color, "#F4F6FA")
    if MONO:                                # plain textbook boxes: black outline, white or light-grey fill
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0.012", fc="#E9E9E9" if filled else "white",
                                    ec="black", lw=1.3, zorder=3))
        filled = False
    else:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025",
                                    fc=face, ec=color, lw=1.6, zorder=3))
    tc = "white" if filled else INK
    if body:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=tc, zorder=4)
        ax.text(x + w / 2, y + h * 0.30, body, ha="center", va="center", fontsize=body_size,
                color="white" if filled else MUTED, zorder=4, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=tc, zorder=4)


def arrow(ax, p, q, color=SLATE, text="", rad=0.0, lw=1.7, style="-|>", text_offset=(0, 0.018)):
    if MONO:
        color = "black"
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=6,
                                 shrinkA=0, shrinkB=0))
    if text:
        ax.text((p[0] + q[0]) / 2 + text_offset[0], (p[1] + q[1]) / 2 + text_offset[1], text,
                ha="center", va="bottom", fontsize=8.2, color=MUTED, zorder=5,
                bbox=dict(fc="white", ec="none", pad=1.2))


def canvas(w=11, h=5.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.axis("off")
    return fig, ax


# 1. serving architecture ------------------------------------------------------------------------
def fig_architecture():
    fig, ax = canvas(12, 5.0)
    ax.set_title("How RouteWise answers one prompt", loc="left")
    perf = R["performance"]
    box(ax, 0.00, 0.58, 0.13, 0.22, "User prompt", "typed in the\nweb app", SLATE)
    box(ax, 0.17, 0.58, 0.14, 0.22, "Tokenizer", "WordPiece,\nfirst 512 tokens", SKY)
    box(ax, 0.35, 0.53, 0.17, 0.32, "Distilled transformer",
        f"bge-small, 33M params\nint8 ONNX, {perf['text_model_mb']:.1f} MB\nruns on 1 CPU core", STRONG, filled=True)
    box(ax, 0.56, 0.58, 0.14, 0.22, "Stacker", "logistic regression\n= P(cheap fails)", SKY)
    box(ax, 0.74, 0.58, 0.12, 0.22, "Threshold", "set by the\nbudget lever", AMBER)
    box(ax, 0.89, 0.74, 0.11, 0.18, "Cheap LLM", "gpt-oss-20b", CHEAP, filled=True)
    box(ax, 0.89, 0.44, 0.11, 0.18, "Strong LLM", "gpt-oss-120b", STRONG, filled=True)
    for a, b in [((0.13, 0.69), (0.17, 0.69)), ((0.31, 0.69), (0.35, 0.69)), ((0.52, 0.69), (0.56, 0.69)),
                 ((0.70, 0.69), (0.74, 0.69))]:
        arrow(ax, a, b)
    arrow(ax, (0.86, 0.72), (0.89, 0.82), CHEAP, rad=-0.2)
    arrow(ax, (0.86, 0.66), (0.89, 0.54), STRONG, rad=0.2)
    ax.text(0.875, 0.88, "risk < threshold", fontsize=8, color=INK if MONO else CHEAP, ha="right")
    ax.text(0.875, 0.47, "risk ≥ threshold", fontsize=8, color=INK if MONO else STRONG, ha="right")
    # bottom band: what the user sees + timings
    ax.add_patch(FancyBboxPatch((0.0, 0.05), 1.0, 0.25, boxstyle="square,pad=0.01" if MONO else "round,pad=0.01,rounding_size=0.03",
                                fc="white" if MONO else "#F6F7FB", ec="black" if MONO else GRID, lw=1, zorder=1))
    ax.text(0.02, 0.24, "Decision card shown with every answer", fontsize=10, fontweight="bold")
    ax.text(0.02, 0.12, "risk score · threshold · chosen model · router latency · LLM latency · expected quality "
            "at the chosen budget", fontsize=9, color=MUTED)
    ax.text(0.98, 0.24, f"routing time: {perf['single_p50_ms']:.1f} ms median", ha="right", fontsize=10,
            fontweight="bold", color=INK if MONO else CHEAP)
    ax.text(0.98, 0.12, f"p95 {perf['single_p95_ms']:.0f} ms on one CPU thread", ha="right", fontsize=9, color=MUTED)
    save(fig, "fig01_architecture.png")


# 2. training pipeline -----------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = canvas(12, 6.2)
    ax.set_title("From raw data to the deployed router", loc="left")
    d = S["data"]
    box(ax, 0.00, 0.62, 0.16, 0.24, "RouteLLM data", f"{d['train_rows']:,} train\n{d['test_rows']:,} test prompts", SLATE)
    box(ax, 0.20, 0.62, 0.15, 0.24, "Pre-processing", "Unicode clean-up\nde-duplication\nleakage check", SKY)
    box(ax, 0.39, 0.62, 0.15, 0.24, "Features", f"{S['features']['n_features']:,} TF-IDF\n+ 24 prompt stats", SKY)
    box(ax, 0.58, 0.70, 0.19, 0.18, "Classical ML (CPU)", "11 classifiers · 6 regressors", CHEAP)
    box(ax, 0.58, 0.48, 0.19, 0.18, "Unsupervised", "K-Means · Ward · DBSCAN\nApriori rules", CHEAP)
    box(ax, 0.81, 0.62, 0.19, 0.24, "Light router", "Ridge regression\nfor 512 MB servers", CHEAP, filled=True)
    arrow(ax, (0.16, 0.74), (0.20, 0.74))
    arrow(ax, (0.35, 0.74), (0.39, 0.74))
    arrow(ax, (0.54, 0.76), (0.58, 0.79))
    arrow(ax, (0.54, 0.70), (0.58, 0.57))
    arrow(ax, (0.77, 0.79), (0.81, 0.74))
    # GPU lane
    ax.add_patch(FancyBboxPatch((0.0, 0.02), 1.0, 0.36, boxstyle="square,pad=0.01" if MONO else "round,pad=0.01,rounding_size=0.03",
                                fc="white" if MONO else "#F7F4FE", ec="#555555" if MONO else SOFT[STRONG], lw=1.2,
                                ls="--" if MONO else "-", zorder=1))
    ax.text(0.985, 0.33, "GPU work (NVIDIA H100)", fontsize=9.5, fontweight="bold", color=STRONG, ha="right")
    box(ax, 0.02, 0.06, 0.17, 0.22, "Fine-tune", "bge-small, two heads\n(fail + score deficit)", STRONG)
    box(ax, 0.23, 0.06, 0.17, 0.22, "Model-size study", "33M → 8B parameters\nQwen3 best", STRONG)
    box(ax, 0.44, 0.06, 0.17, 0.22, "Teacher", "Qwen3-4B, 5-fold\nout-of-fold scores", STRONG)
    box(ax, 0.65, 0.06, 0.15, 0.22, "Student", "bge-small learns\nlabels + teacher", STRONG)
    box(ax, 0.84, 0.06, 0.16, 0.22, "Full router", "int8 ONNX + stacker\nchosen on validation", STRONG, filled=True)
    for a, b in [((0.19, 0.17), (0.23, 0.17)), ((0.40, 0.17), (0.44, 0.17)), ((0.61, 0.17), (0.65, 0.17)),
                 ((0.80, 0.17), (0.84, 0.17))]:
        arrow(ax, a, b, STRONG)
    arrow(ax, (0.465, 0.62), (0.11, 0.28), SLATE, rad=0.15, text="prompts + grades", text_offset=(0.05, 0.0))
    save(fig, "fig02_training_pipeline.png")


# 3. use-case diagram -----------------------------------------------------------------------------
def actor(ax, x, y, name, color=INK):
    ax.add_patch(plt.Circle((x, y + 0.11), 0.022, fc="white", ec=color, lw=1.8, zorder=3))
    ax.plot([x, x], [y + 0.088, y + 0.02], color=color, lw=1.8)
    ax.plot([x - 0.03, x + 0.03], [y + 0.065, y + 0.065], color=color, lw=1.8)
    ax.plot([x, x - 0.025], [y + 0.02, y - 0.03], color=color, lw=1.8)
    ax.plot([x, x + 0.025], [y + 0.02, y - 0.03], color=color, lw=1.8)
    ax.text(x, y - 0.05, name, ha="center", va="top", fontsize=10, fontweight="bold")


def usecase(ax, x, y, text, color):
    ax.add_patch(matplotlib.patches.Ellipse((x, y), 0.25, 0.10, fc="white" if MONO else SOFT[color], ec="black" if MONO else color, lw=1.3 if MONO else 1.5, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=9, zorder=4)


def fig_usecase():
    fig, ax = canvas(11, 6.0)
    ax.set_title("Use-case diagram", loc="left")
    ax.add_patch(FancyBboxPatch((0.2, 0.04), 0.58, 0.88, boxstyle="square,pad=0.01" if MONO else "round,pad=0.01,rounding_size=0.03",
                                fc="white" if MONO else "#FBFBFD", ec="black" if MONO else SLATE, lw=1.3))
    ax.text(0.49, 0.89, "RouteWise system", ha="center", fontsize=11, fontweight="bold", color=MUTED)
    actor(ax, 0.08, 0.62, "User")
    actor(ax, 0.08, 0.17, "Developer")
    actor(ax, 0.92, 0.55, "LLM provider\n(Groq API)", MUTED)
    ucs_user = [(0.49, 0.80, "Ask a question", CHEAP), (0.49, 0.68, "Choose strong-model budget", AMBER),
                (0.49, 0.56, "See routing decision", SKY), (0.49, 0.44, "Explore results dashboard", SKY)]
    ucs_dev = [(0.49, 0.30, "Train and evaluate models", STRONG), (0.49, 0.16, "Build and deploy router", STRONG)]
    for x, y, t, c in ucs_user:
        usecase(ax, x, y, t, c)
        ax.plot([0.11, x - 0.125], [0.70, y], color="black" if MONO else SLATE, lw=1)
    for x, y, t, c in ucs_dev:
        usecase(ax, x, y, t, c)
        ax.plot([0.11, x - 0.125], [0.25, y], color="black" if MONO else SLATE, lw=1)
    ax.plot([0.615, 0.89], [0.80, 0.64], color="black" if MONO else SLATE, lw=1)
    ax.text(0.73, 0.60, "«include»\nroute the prompt", fontsize=8, color=MUTED, ha="center")
    save(fig, "fig03_use_case.png")


# 4. class diagram -----------------------------------------------------------------------------------
def uml_class(ax, x, y, w, name, attrs, methods, color):
    lh = 0.042
    h = 0.07 + lh * (len(attrs) + len(methods)) + 0.03
    if MONO:
        color = "black"
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="square,pad=0.004" if MONO else "round,pad=0.004,rounding_size=0.012",
                                fc="white", ec=color, lw=1.6, zorder=3))
    ax.add_patch(FancyBboxPatch((x, y - 0.065), w, 0.065, boxstyle="square,pad=0.004" if MONO else "round,pad=0.004,rounding_size=0.012",
                                fc="#E6E6E6" if MONO else color, ec=color, lw=1.6, zorder=3))
    ax.text(x + w / 2, y - 0.033, name, ha="center", va="center", color="black" if MONO else "white", fontweight="bold", fontsize=9.5, zorder=4)
    yy = y - 0.065 - lh * 0.75
    for a in attrs:
        ax.text(x + 0.008, yy, a, fontsize=7.8, color=INK, va="center", zorder=4, family="monospace")
        yy -= lh
    ax.plot([x, x + w], [yy + lh * 0.45, yy + lh * 0.45], color=GRID, lw=1, zorder=4)
    for m in methods:
        ax.text(x + 0.008, yy, m, fontsize=7.8, color=STRONG, va="center", zorder=4, family="monospace")
        yy -= lh
    return (x, y - h, w, h)


def fig_class():
    fig, ax = canvas(12, 6.6)
    ax.set_title("Main classes of the system", loc="left")
    uml_class(ax, 0.00, 0.92, 0.23, "FastAPI app (main.py)", ["rate_limits", "LLM_BASE_URL, models"],
              ["route(prompt, share)", "chat(prompt, share)", "metrics()"], SLATE)
    uml_class(ax, 0.30, 0.92, 0.25, "Router", ["featurizer", "components", "stacker", "share_thresholds"],
              ["risk(prompts)", "threshold_for(share)", "route(prompt, share)", "explain(prompt)"], STRONG)
    uml_class(ax, 0.62, 0.92, 0.25, "TextRouterModel", ["session (ONNX Runtime)", "tok (tokenizers)", "max_tokens"],
              ["logits(texts)"], STRONG)
    uml_class(ax, 0.30, 0.40, 0.25, "PromptFeaturizer", ["word_ (TF-IDF)", "char_ (TF-IDF)", "scaler_"],
              ["fit(texts)", "transform(texts)"], CHEAP)
    uml_class(ax, 0.62, 0.40, 0.25, "TwoHeadRouter (training)", ["encoder (transformer)", "head (2 outputs)"],
              ["forward(ids, mask)", "export_onnx()"], AMBER)
    uml_class(ax, 0.00, 0.40, 0.23, "TransactionDB / apriori", ["items", "matrix (bool)"],
              ["support(itemsets)", "apriori(min_support)"], SKY)
    arrow(ax, (0.23, 0.80), (0.30, 0.80), INK, text="uses")
    arrow(ax, (0.55, 0.80), (0.62, 0.80), INK)
    ax.text(0.585, 0.83, "uses", ha="center", fontsize=8.2, color=MUTED)
    arrow(ax, (0.425, 0.478), (0.425, 0.40), INK)
    ax.text(0.435, 0.44, "uses", fontsize=8.2, color=MUTED)
    arrow(ax, (0.745, 0.40), (0.745, 0.74), AMBER, text="exported to ONNX", text_offset=(0.07, -0.01),
          style="-|>")
    save(fig, "fig04_class_diagram.png")


# 5. activity diagram --------------------------------------------------------------------------------
def fig_activity():
    fig, ax = canvas(12, 4.4)
    ax.set_title("Activity diagram: one chat request", loc="left")
    ax.add_patch(plt.Circle((0.02, 0.55), 0.018, fc=INK, zorder=3))
    steps = [(0.06, "Receive prompt\nand budget", SLATE), (0.19, "Validate\n(1-4000 chars)", SKY),
             (0.32, "Rate-limit\ncheck", SKY), (0.45, "Score risk\n(router)", STRONG), (0.58, "Compare with\nthreshold", AMBER)]
    for x, t, c in steps:
        box(ax, x, 0.45, 0.10, 0.20, t, color=c, title_size=8.8)
    for i in range(len(steps) - 1):
        arrow(ax, (steps[i][0] + 0.10, 0.55), (steps[i + 1][0], 0.55))
    arrow(ax, (0.038, 0.55), (0.06, 0.55))
    box(ax, 0.73, 0.70, 0.11, 0.18, "Call cheap\nLLM", color=CHEAP, title_size=8.8)
    box(ax, 0.73, 0.22, 0.11, 0.18, "Call strong\nLLM", color=STRONG, title_size=8.8)
    arrow(ax, (0.68, 0.58), (0.73, 0.78), CHEAP, rad=-0.15)
    arrow(ax, (0.68, 0.52), (0.73, 0.31), STRONG, rad=0.15)
    ax.text(0.672, 0.80, "low risk", fontsize=8.2, color=MUTED, ha="right")
    ax.text(0.672, 0.27, "high risk", fontsize=8.2, color=MUTED, ha="right")
    box(ax, 0.88, 0.45, 0.11, 0.20, "Return answer\n+ decision", color=SLATE, title_size=8.8)
    arrow(ax, (0.84, 0.79), (0.88, 0.60), rad=-0.1)
    arrow(ax, (0.84, 0.31), (0.88, 0.50), rad=0.1)
    ax.text(0.255, 0.30, "invalid → error 422 · too many requests → error 429", fontsize=8.5, color=CORAL, ha="center")
    ax.text(0.935, 0.36, "LLM error →\ndecision only", fontsize=8, color=CORAL, ha="center")
    save(fig, "fig05_activity.png")


# 6. data: grade distribution and fail rate ----------------------------------------------------------
def fig_data():
    import pandas as pd
    tr = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet", columns=["score"])
    counts = tr.score.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    colors = [CORAL if g <= 3 else CHEAP for g in counts.index]
    bars = ax.bar(counts.index.astype(str), counts.values, color=colors, width=0.62, zorder=3)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 900, f"{v:,}\n({v / counts.sum():.1%})", ha="center", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.22)
    ax.set_xlabel("GPT-4's grade of the cheap model's answer (1 = bad, 5 = perfect)")
    ax.set_ylabel("training prompts")
    ax.set_title("Most cheap answers are fine; 13.5% fail (grade 3 or lower)")
    ax.grid(axis="x", visible=False)
    save(fig, "fig06_grade_distribution.png")


# 7. routing curve --------------------------------------------------------------------------------
def fig_routing_curve():
    c = S["curves"]
    x = np.array(c["x"]) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(x, x, np.array(c["final"]) * 100, color=STRONG, alpha=0.10, zorder=1)
    ax.plot(x, np.array(c["oracle"]) * 100, ls=(0, (2, 2)), color=INK, lw=1.6, label=f"Perfect router (APGR {S['baselines']['oracle_apgr']:.3f})")
    ax.plot(x, np.array(c["final"]) * 100, color=STRONG, lw=3, label=f"RouteWise (APGR {R['apgr']:.3f})", zorder=3)
    ax.plot([0, 100], [0, 100], color=SLATE, lw=1.5, ls="--", label=f"Random routing ({S['baselines']['random_apgr']:.3f})")
    for k, col in (("at_30pct", CHEAP), ("at_50pct", AMBER)):
        p = R[k]
        ax.scatter([p["strong_share"] * 100], [p["pgr"] * 100], s=70, color=col, zorder=5, ec="white", lw=1.5)
        ax.annotate(f"{p['strong_share']:.0%} strong calls →\n{p['pgr']:.0%} of the gap recovered\n(grade {p['quality']:.2f} / 5)",
                    (p["strong_share"] * 100, p["pgr"] * 100), xytext=(p["strong_share"] * 100 + 8, p["pgr"] * 100 - 22),
                    fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=col, lw=1.2))
    ax.set(xlim=(0, 100), ylim=(0, 102), xlabel="% of prompts sent to the strong model (cost)",
           ylabel="% of the quality gap recovered")
    ax.set_title("Routing curve on 10,000 unseen test prompts")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "fig07_routing_curve.png")


# 8. leaderboard ------------------------------------------------------------------------------------
FAMILY = {"Logistic Regression": ("linear", SKY), "Linear SVM": ("linear", SKY),
          "Multinomial Naive Bayes": ("probabilistic", AMBER), "Complement Naive Bayes": ("probabilistic", AMBER),
          "Decision Tree (Gini)": ("trees", CHEAP), "Decision Tree (Entropy / information gain)": ("trees", CHEAP),
          "Random Forest": ("trees", CHEAP), "Gradient Boosting (beyond syllabus)": ("trees", CHEAP),
          "K-Nearest Neighbours": ("distance", CORAL), "RBF-kernel SVM (20k sample)": ("distance", CORAL),
          "Neural Network (MLP)": ("neural", STRONG), "Embedding Neural Network (GPU)": ("neural", STRONG),
          "Fine-tuned Transformer (GPU)": ("transformer", INK),
          "Fine-tuned Transformer (GPU), score-deficit head": ("transformer", INK)}


def short(n):
    return (n.replace(" (beyond syllabus)", "").replace(" (Entropy / information gain)", " (entropy)")
             .replace("Fine-tuned Transformer (GPU), score-deficit head", "Distilled transformer, deficit head")
             .replace("Fine-tuned Transformer (GPU)", "Distilled transformer, fail head")
             .replace("RBF-kernel SVM (20k sample)", "RBF SVM (20k sample)"))


def fig_leaderboard():
    rows = sorted(S["classifiers"], key=lambda r: r["apgr"])
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    names = [short(r["name"]) for r in rows]
    vals = [r["apgr"] for r in rows]
    cols = [FAMILY.get(r["name"], ("other", SLATE))[1] for r in rows]
    if MONO:        # two shades only: the distilled transformer in black, every other model in grey
        cols = ["#000000" if r["name"].startswith("Fine-tuned") else "#9B9B9B" for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, [v - 0.5 for v in vals], left=0.5, color=cols, height=0.62, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.003, yi, f"{v:.3f}", va="center", fontsize=8.8)
    ax.set_yticks(y, names, fontsize=9)
    ax.axvline(0.5, color=SLATE, ls="--", lw=1)
    ax.axvline(S["baselines"]["oracle_apgr"], color=INK, ls=":", lw=1.2)
    ax.text(0.502, len(rows) - 0.4, "random", color=MUTED, fontsize=8.5)
    ax.text(S["baselines"]["oracle_apgr"] - 0.003, len(rows) - 0.4, "perfect", color=INK, fontsize=8.5, ha="right")
    ax.set_xlim(0.5, 0.91)
    ax.set_xlabel("test APGR (share of the quality gap recovered, averaged over all budgets)")
    ax.set_title("Every model, one score")
    ax.grid(axis="y", visible=False)
    handles = {}
    for fam, col in FAMILY.values():
        handles[fam] = plt.Rectangle((0, 0), 1, 1, color=col)
    if not MONO:
        ax.legend(handles.values(), handles.keys(), loc="lower left", bbox_to_anchor=(0.6, 0.02), fontsize=8.5, ncol=2)
    save(fig, "fig08_leaderboard.png")


# 9. scaling study + distillation -------------------------------------------------------------------
def best(tag_start):
    runs = [r for r in SWEEP if r["tag"].startswith(tag_start) and r.get("best_checkpoint", {}).get("val_auc_fail", 0) > 0.6]
    return max(runs, key=lambda r: r["best_checkpoint"]["val_auc_fail"]) if runs else None


def fig_scaling():
    study = [r for r in SWEEP if r["tag"].startswith("study:") and "FAILED" not in r["tag"]]
    plain = dict(best("r2: lr 5e-5, batch 64"))
    # the round-1 runs did not log their size; later bge-small runs did (same model)
    plain["params_millions"] = next(r["params_millions"] for r in SWEEP
                                    if r.get("base") == "BAAI/bge-small-en-v1.5" and r.get("params_millions"))
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.axvspan(10, 95, color=SOFT[CHEAP], zorder=0)
    ax.text(12, 0.829, "deployable on a CPU\n(int8 file < 95 MB)", fontsize=8.8, color=MUTED if MONO else CHEAP, va="top")
    pts = [(plain["params_millions"], plain["best_checkpoint"]["val_auc_fail"], "bge-small\n(plain fine-tune)", SKY)]
    for r in study:
        name = r["tag"].split(":")[1].split(",")[0].split("(")[0].strip()
        pts.append((r["params_millions"], r["best_checkpoint"]["val_auc_fail"], name, STRONG if "Qwen" in name else SKY))
    for p, auc, name, col in pts:
        ax.scatter(p, auc, s=60 + 22 * np.log10(p) ** 2, color=col, alpha=0.9, ec="white", lw=1.5, zorder=3)
        ax.annotate(name, (p, auc), xytext=(7, -4), textcoords="offset points", fontsize=8.6)
    final = S["finetune"]["results"]["Fine-tuned Transformer (GPU)"]["val_auc"]
    ax.scatter(plain["params_millions"], final, marker="*", s=360, color=AMBER, ec=INK, lw=0.8, zorder=4)
    ax.annotate(f"distilled bge-small\n(student of Qwen3-4B): {final:.3f}", (plain["params_millions"], final),
                xytext=(-8, -34), textcoords="offset points", fontsize=8.8, fontweight="bold", ha="left")
    q4 = next(p for p in pts if p[2].startswith("Qwen3-4B"))
    ax.add_patch(FancyArrowPatch((q4[0], q4[1]), (plain["params_millions"] * 1.15, final + 0.001),
                                 arrowstyle="-|>", mutation_scale=14, color=AMBER, lw=1.8,
                                 connectionstyle="arc3,rad=0.25", zorder=2))
    ax.text(250, 0.8115, "knowledge\ndistillation", color=AMBER, fontsize=9, fontweight="bold", ha="center")
    ax.set_xscale("log")
    ax.set(xlim=(15, 15000), ylim=(0.775, 0.835), xlabel="model size (millions of parameters, log scale)",
           ylabel="validation AUC")
    ax.set_title("Bigger models route better; distillation brings the gain back to a small one")
    save(fig, "fig09_scaling_and_distillation.png")


def fig_distill_steps():
    plain = best("r2: lr 5e-5, batch 64")["best_checkpoint"]["val_auc_fail"]
    steps = [("Plain fine-tune\n(bge-small)", plain, SKY),
             ("+ Qwen3-1.7B teacher", best("distill: bge-small student of Qwen3-1.7B, alpha 0.2")["best_checkpoint"]["val_auc_fail"], STRONG),
             ("+ 4 epochs,\n512 tokens", best("distill: bge-small student of Qwen3-1.7B, alpha 0.2, 4 epochs, max_tokens 512")["best_checkpoint"]["val_auc_fail"], STRONG),
             ("+ Qwen3-4B teacher", best("distill: bge-small student of Qwen3-4B")["best_checkpoint"]["val_auc_fail"], STRONG),
             ("+ mean pooling,\nlr 8e-5, alpha 0.1", S["finetune"]["results"]["Fine-tuned Transformer (GPU)"]["val_auc"], AMBER),
             ("int8 file\n(served, 33.9 MB)", S["finetune"]["export"]["variants_validation"]["int8"]["val_auc_fail"], CHEAP)]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(steps))
    vals = [s[1] for s in steps]
    ax.plot(x, vals, color=SLATE, lw=1.5, zorder=2)
    for xi, (lab, v, col) in zip(x, steps):
        ax.scatter(xi, v, s=170, color=col, ec="white", lw=2, zorder=3)
        ax.text(xi, v + 0.0012, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    teacher = best("study: Qwen3-4B")["best_checkpoint"]["val_auc_fail"]
    ax.axhline(teacher, color=STRONG, ls=":", lw=1.3)
    ax.text(len(steps) - 0.5, teacher + 0.0006, f"Qwen3-4B teacher itself: {teacher:.4f} (4,023M parameters)",
            ha="right", fontsize=8.6, color=STRONG)
    ax.set_xticks(x, [s[0] for s in steps], fontsize=8.6)
    ax.set_ylim(0.78, 0.826)
    ax.set_ylabel("validation AUC (fail head)")
    ax.set_title("How the 33M-parameter student improved, step by step")
    ax.grid(axis="x", visible=False)
    save(fig, "fig10_distillation_steps.png")


# 11. budget control + quality ----------------------------------------------------------------------
def fig_budget():
    bc = R["budget_check"]
    req = [int(k[:-1]) for k in bc]
    got = [v * 100 for v in bc.values()]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    a1.plot([0, 100], [0, 100], color=SLATE, ls="--", lw=1)
    a1.scatter(req, got, s=60, color=AMBER, ec="white", lw=1.5, zorder=3)
    a1.set(xlabel="requested strong-model share (%)", ylabel="share actually sent on test (%)")
    a1.set_title("Requested and realised budget")
    shares = [0, 20, 30, 50, 100]
    qual = [R["quality_cheap_only"]] + [R[f"at_{s}pct"]["quality"] for s in (20, 30, 50)] + [R["quality_strong_only"]]
    a2.plot(shares, qual, color=STRONG, lw=2.6, marker="o", ms=7, mfc="white", mew=2)
    for s, q in zip(shares, qual):
        a2.text(s, q + 0.025, f"{q:.2f}", ha="center", fontsize=9)
    a2.axhline(R["quality_cheap_only"], color=CHEAP, ls=":", lw=1.2)
    a2.text(100, R["quality_cheap_only"] + 0.01, "cheap model only", color=CHEAP, ha="right", fontsize=8.5)
    a2.set(xlabel="strong-model share (%)", ylabel="average grade (out of 5)", ylim=(4.35, 5.08))
    a2.set_title("Average grade at each budget")
    save(fig, "fig11_budget_and_quality.png")


# 12. confusion matrix ----------------------------------------------------------------------------
def fig_confusion():
    cm = R["classification_at_30pct"]["confusion_matrix"]
    m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    colors = np.array([[SOFT[CHEAP], SOFT[AMBER]], [SOFT[CORAL], SOFT[STRONG]]])
    labels = [["kept on cheap,\nanswer was fine", "sent to strong,\nwas not needed"],
              ["kept on cheap,\nanswer failed", "sent to strong,\nwas needed"]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(FancyBboxPatch((j + 0.04, 1 - i + 0.04), 0.92, 0.92, boxstyle="square,pad=0" if MONO else "round,pad=0,rounding_size=0.06",
                                        fc=colors[i, j], ec="white"))
            ax.text(j + 0.5, 1 - i + 0.58, f"{m[i, j]:,}", ha="center", fontsize=17, fontweight="bold")
            ax.text(j + 0.5, 1 - i + 0.27, labels[i][j], ha="center", fontsize=8.6, color=MUTED)
    ax.set(xlim=(0, 2), ylim=(0, 2))
    ax.set_xticks([0.5, 1.5], ["router: cheap", "router: strong"])
    ax.set_yticks([1.5, 0.5], ["truly fine", "truly failed"])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    c = R["classification_at_30pct"]
    ax.set_title(f"At a 30% budget: recall {c['recall_sensitivity']:.0%}, specificity {c['specificity']:.0%}")
    save(fig, "fig12_confusion_matrix.png")


# 13. clusters ------------------------------------------------------------------------------------
def fig_clusters():
    cl = sorted(S["clustering"]["clusters"], key=lambda c: c["fail_rate_train"])
    fig, ax = plt.subplots(figsize=(9, 6))
    base = S["data"]["fail_rate_train"]
    y = np.arange(len(cl))
    vals = [c["fail_rate_train"] * 100 for c in cl]
    cols = [CORAL if v / 100 > base * 1.5 else (AMBER if v / 100 > base else CHEAP) for v in vals]
    ax.barh(y, vals, color=cols, height=0.65, zorder=3)
    ax.set_yticks(y, [", ".join(c["top_words"][:3]) for c in cl], fontsize=8.6)
    for yi, v, c in zip(y, vals, cl):
        ax.text(v + 0.5, yi, f"{v:.0f}%  ({c['size_train']:,})", va="center", fontsize=8.4, color=MUTED,
                bbox=dict(fc="white", ec="none", pad=0.6), zorder=4)
    ax.axvline(base * 100, color=INK, ls="--", lw=1)
    ax.text(base * 100 + 0.4, len(cl) - 0.5, f"overall {base:.1%}", fontsize=8.6)
    ax.set_xlabel("cheap-model failure rate in the cluster (%)   (cluster size in brackets)")
    ax.set_title(f"K-Means (k = {S['clustering']['k']}): failure depends on the kind of task")
    ax.grid(axis="y", visible=False)
    save(fig, "fig13_clusters.png")


# 14. association rules ------------------------------------------------------------------------------
def fig_rules():
    rules = S["association"]["rules_to_fail"][:10][::-1]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    y = np.arange(len(rules))
    lifts = [r["lift"] for r in rules]
    ax.hlines(y, 1, lifts, color=SOFT[CORAL], lw=6, zorder=2)
    ax.scatter(lifts, y, s=[r["confidence"] * 300 for r in rules], color=CORAL, ec="white", lw=1.5, zorder=3)
    ax.set_yticks(y, [" AND ".join(r["if"]) for r in rules], fontsize=8.6)
    for yi, r in zip(y, rules):
        ax.text(r["lift"] + 0.1, yi, f"lift {r['lift']:.2f} · conf {r['confidence']:.0%}", va="center", fontsize=8.4)
    ax.axvline(1, color=SLATE, ls="--", lw=1)
    ax.set_xlim(0.9, max(lifts) + 1.2)
    ax.set_xlabel("lift (how many times more likely than average the cheap answer fails)")
    a = S["association"]
    ax.set_title(f"Apriori rules that predict a failure  (Apriori: {a['apriori']['candidates_counted']:,} candidates "
                 f"vs naive {a['naive']['candidates_counted']:,})", fontsize=11.5)
    ax.grid(axis="y", visible=False)
    save(fig, "fig14_association_rules.png")


# 15. regression -------------------------------------------------------------------------------------
def fig_regression():
    rg = S["regression"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    names = [r["name"].replace(", full features", "").replace(" (24 prompt stats)", "\n(24 stats)")
             .replace(" (log prompt length)", "\n(prompt length)") for r in rg]
    y = np.arange(len(rg))
    a1.barh(y, [r["r2"] for r in rg], color=SKY, height=0.6, zorder=3)
    a2.barh(y, [r["apgr"] - 0.5 for r in rg], left=0.5, color=CHEAP, height=0.6, zorder=3)
    for yi, r in zip(y, rg):
        a1.text(r["r2"] + 0.003, yi, f"{r['r2']:.3f}", va="center", fontsize=8.6)
        a2.text(r["apgr"] + 0.002, yi, f"{r['apgr']:.3f}", va="center", fontsize=8.6)
    a1.set_yticks(y, names, fontsize=8.6)
    a1.set_title("R² of the predicted grade")
    a2.set_title("Routing quality (test APGR)")
    a2.set_xlim(0.5, 0.71)
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
    save(fig, "fig15_regression.png")


# 16. features + activations ---------------------------------------------------------------------------
def fig_features():
    ab = S["features"]["ablation"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    names = list(ab)
    vals = [ab[n]["val_auc"] for n in names]
    cols = [AMBER if n == S["features"]["chosen"] else SKY for n in names]
    a1.bar(range(len(names)), vals, color=cols, width=0.6, zorder=3)
    a1.set_xticks(range(len(names)), [n.replace("_", " ").replace("+", "+\n").replace(" vocab ", "\nvocab\n") for n in names], fontsize=8)
    a1.set_ylim(min(vals) - 0.004, max(vals) + 0.003)
    for i, v in enumerate(vals):
        a1.text(i, v + 0.0004, f"{v:.4f}", ha="center", fontsize=8)
    a1.set_title("Feature ablation (validation AUC)")
    a1.grid(axis="x", visible=False)
    af = S["activation_functions"]
    groups = {"MLP (CPU)": {k.split(", ")[1]: v["val_auc"] for k, v in af.items() if k.startswith("MLP")},
              "Embedding network (GPU)": {k.split(", ")[1]: v["val_auc"] for k, v in af.items() if k.startswith("Embedding")}}
    acts = ["logistic", "tanh", "relu", "gelu"]
    w = 0.38
    for gi, (g, d) in enumerate(groups.items()):
        xs = [i + (gi - 0.5) * w for i, a in enumerate(acts) if a in d]
        vs = [d[a] for a in acts if a in d]
        a2.bar(xs, vs, width=w, color=[SKY, STRONG][gi], label=g, zorder=3)
        for x_, v in zip(xs, vs):
            a2.text(x_, v + 0.0006, f"{v:.3f}", ha="center", fontsize=7.6)
    a2.set_xticks(range(len(acts)), acts)
    a2.set_ylim(0.73, 0.768)
    a2.set_title("Activation functions (validation AUC)")
    a2.legend(fontsize=8, loc="upper left")
    a2.grid(axis="x", visible=False)
    save(fig, "fig16_features_activations.png")


# 17-18. test breakdowns (numbers from docs/RESULTS.md section 10, reporting only) -------------------
GRADE_RISK = [(1, 492, 0.345, 0.80), (2, 432, 0.268, 0.69), (3, 429, 0.198, 0.59), (4, 1716, 0.147, 0.37), (5, 6931, 0.105, 0.22)]
SOURCES = [("lmsys-chat-1m", 3144, 0.761, 0.788), ("sharegpt", 2027, 0.706, 0.790), ("anthropic-hh", 1594, 0.720, 0.733),
           ("flan_v2_niv2", 991, 0.726, 0.772), ("evol_instruct", 809, 0.682, 0.762), ("ultrachat", 803, 0.746, 0.782),
           ("flan_v2_cot", 231, 0.671, 0.691)]


def fig_breakdowns():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.subplots_adjust(wspace=0.5)
    g = [r[0] for r in GRADE_RISK]
    a1.bar(g, [r[3] * 100 for r in GRADE_RISK], color=[CORAL, CORAL, AMBER, SKY, CHEAP], width=0.6, zorder=3)
    for r in GRADE_RISK:
        a1.text(r[0], r[3] * 100 + 2, f"{r[3]:.0%}", ha="center", fontsize=9, fontweight="bold")
    a1.set(xlabel="true grade of the cheap answer", ylabel="% sent to the strong model (30% budget)", ylim=(0, 95))
    a1.set_title("Share sent to the strong model, by true grade")
    a1.grid(axis="x", visible=False)
    y = np.arange(len(SOURCES))[::-1]
    a2.scatter([s[2] for s in SOURCES], y, color=SKY, s=60, zorder=3, label="logistic regression")
    a2.scatter([s[3] for s in SOURCES], y, color=STRONG, s=80, zorder=3, label="final router")
    for yi, s in zip(y, SOURCES):
        a2.annotate("", (s[3], yi), (s[2], yi), arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.2))
    a2.set_yticks(y, [f"{s[0]} ({s[1]:,})" for s in SOURCES], fontsize=8.6)
    a2.set_xlabel("test AUC within the source")
    a2.set_title("Test AUC by data source")
    a2.legend(fontsize=8.5, loc="lower right")
    save(fig, "fig17_test_breakdowns.png")


if __name__ == "__main__":
    for f in (fig_architecture, fig_pipeline, fig_usecase, fig_class, fig_activity, fig_data, fig_routing_curve,
              fig_leaderboard, fig_scaling, fig_distill_steps, fig_budget, fig_confusion, fig_clusters, fig_rules,
              fig_regression, fig_features, fig_breakdowns):
        f()
