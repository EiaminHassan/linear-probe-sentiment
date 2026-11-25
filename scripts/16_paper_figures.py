"""
Paper figures and Table 1, built directly from saved results.
Output: figures/paper_fig1.pdf/.png, figures/paper_fig2.pdf/.png, results/table1.tex
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 3.4 inches fits one column of a two-column paper. Use about 6.0 for a one-column layout.
FIG_WIDTH = 3.4
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
BLUE, ORANGE, GREEN, GRAY = "#257de0", "#f47230", "#00b894", "#888888"
TICKS = ["emb"] + [str(i) for i in range(1, 13)]


def style(ax):
    ax.set_xticks(range(13))
    ax.set_xticklabels(TICKS)
    ax.set_xlabel("Residual stream point")
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")   # vector, for LaTeX
    fig.savefig(f"figures/{name}.png", dpi=300, bbox_inches="tight")
    print("saved", name)


# ---- Figure 1: accuracy by point, 5 runs ----
summary = pd.read_csv("results/resamples_summary.csv")
bow = pd.read_csv("results/resamples_bow.csv")["test_acc"]
y_test = np.load("data/labels_test.npy")
baseline = max(y_test.mean(), 1 - y_test.mean())

fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.6))
ax.axhspan(bow.mean() - bow.std(), bow.mean() + bow.std(), color=GREEN, alpha=0.15, lw=0)
ax.axhline(bow.mean(), color=GREEN, lw=1.5, ls="--", label="Bag of words")
ax.axhline(baseline, color=GRAY, lw=1, ls=":", label="Majority class")
for model, color, label in [("pretrained", BLUE, "Pretrained GPT-2"), ("random", ORANGE, "Random-init GPT-2")]:
    s = summary[summary["model"] == model]
    ax.errorbar(s["point"], s["mean"], yerr=s["std"], color=color, lw=1.5,
                marker="o", ms=3, capsize=2, label=label)
ax.set_ylabel("Test accuracy")
ax.set_ylim(0.45, 0.9)
style(ax)
handles, labels = ax.get_legend_handles_labels()
order = [labels.index(l) for l in ["Pretrained GPT-2", "Random-init GPT-2", "Bag of words", "Majority class"]]
ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False,
          ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
save(fig, "paper_fig1")

# ---- Figure 2: negation interaction ----
neg = pd.read_csv("results/negation_long.csv")
neg["interaction"] = neg["shift_neg_base"] - neg["shift_pos_base"]

fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.4))
ax.axhline(0, color=GREEN, lw=1.5, ls="--", label="Bag of words (exactly 0)")
for model, color, label in [("pretrained", BLUE, "Pretrained GPT-2"), ("random", ORANGE, "Random-init GPT-2")]:
    s = neg[neg["model"] == model].groupby("point")["interaction"].agg(["mean", "std"])
    ax.errorbar(s.index, s["mean"], yerr=s["std"], color=color, lw=1.5,
                marker="o", ms=3, capsize=2, label=label)
ax.set_ylabel("Negation interaction\n(test-set SDs)")
style(ax)
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.24))
save(fig, "paper_fig2")

# ---- Table 1: paired bootstrap comparisons, in percentage points ----
rows = {
    "pretrained 9 vs pretrained 7": "Point 9 vs point 7",
    "pretrained 9 vs pretrained 12": "Point 9 vs point 12",
    "pretrained 1 vs bag of words": "Point 1 vs bag of words",
    "pretrained 4 vs bag of words": "Point 4 vs bag of words",
    "pretrained 5 vs bag of words": "Point 5 vs bag of words",
    "pretrained 7 vs bag of words": "Point 7 vs bag of words",
    "pretrained 1 vs random 1": "Point 1, pretrained vs random-init",
    "pretrained 9 vs random 9": "Point 9, pretrained vs random-init",
}
pb = pd.read_csv("results/paired_bootstrap.csv").set_index("comparison")
lines = [r"\begin{tabular}{lrrr}", r"\toprule",
         r"Comparison & Diff. (pp) & 95\% interval (pp) & $p$ \\", r"\midrule"]
for key, name in rows.items():
    r = pb.loc[key]
    p = "$<$0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
    lines.append(f"{name} & {100 * r['diff']:+.1f} & "
                 f"[{100 * r['ci_low']:+.1f}, {100 * r['ci_high']:+.1f}] & {p} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
with open("results/table1.tex", "w") as f:
    f.write("\n".join(lines))
print("saved results/table1.tex")
print("\n".join(lines))