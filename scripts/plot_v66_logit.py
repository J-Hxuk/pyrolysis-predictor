"""v66 세 방식(baseline/fraction/logit) 비교 그래프. reports/v66_logit_fraction/ 에 저장."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "reports" / "v66_logit_fraction"

per = pd.read_csv(D / "per_doi_metrics.csv")
agg = pd.read_csv(D / "aggregate_metrics.csv")

METHODS = ["BASELINE_qe_then_clip", "FRACTION_linear", "LOGIT_FRACTION"]
SHORT = ["BASELINE", "FRACTION", "LOGIT"]

# ── 1. DOI별 MAE 분포 (박스 + 지터) ───────────────────────────────
data = [per[per.method == m]["mae"].dropna().to_numpy() for m in METHODS]
fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(data, tick_labels=SHORT, showfliers=False, widths=0.5)
rng = np.random.default_rng(0)
for i, dd in enumerate(data, start=1):
    ax.scatter(rng.normal(i, 0.05, len(dd)), dd, alpha=0.5, s=16, color="#1f77b4")
ax.set_ylabel("per-DOI MAE (mg/g)")
ax.set_title("Per-DOI MAE by target transform (188 rows / 36 DOI)\nlower box = better typical DOI; high dots = tail blowup")
fig.tight_layout()
fig.savefig(D / "doi_mae_box.png", dpi=130)
plt.close(fig)

# ── 2. 요약 지표 막대 (중앙값/q75/최악 MAE) ────────────────────────
dist = pd.read_csv(D / "doi_distribution_summary.csv").set_index("method")
x = np.arange(len(METHODS))
w = 0.25
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.bar(x - w, [dist.loc[m, "DOI_MAE_median"] for m in METHODS], w, label="median")
ax.bar(x, [dist.loc[m, "DOI_MAE_q75"] for m in METHODS], w, label="q75")
ax.bar(x + w, [dist.loc[m, "DOI_MAE_worst"] for m in METHODS], w, label="worst")
ax.set_xticks(x)
ax.set_xticklabels(SHORT)
ax.set_ylabel("per-DOI MAE (mg/g)")
ax.set_title("DOI MAE: median & q75 improve with logit, but worst-case blows up")
ax.legend()
fig.tight_layout()
fig.savefig(D / "doi_mae_summary_bars.png", dpi=130)
plt.close(fig)

# ── 3. 위반율 3방식 ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.5))
bars = ax.bar(SHORT, agg.set_index("method").loc[METHODS, "violation_rate"].to_numpy() * 100,
              color=["#d62728", "#2ca02c", "#2ca02c"])
for bar, v in zip(bars, agg.set_index("method").loc[METHODS, "violation_rate"].to_numpy() * 100):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, f"{v:.1f}%", ha="center", va="bottom")
ax.set_ylabel("Upper-cap violation rate (%)")
ax.set_title("Mass-balance violations: both fraction methods = 0%")
ax.set_ylim(0, 32)
fig.tight_layout()
fig.savefig(D / "violation_three.png", dpi=130)
plt.close(fig)

print("v66_plots_done")
