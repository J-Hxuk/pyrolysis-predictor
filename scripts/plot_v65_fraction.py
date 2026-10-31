"""v65 분율 표적 vs 기준선 비교 그래프 (연구용). reports/v65_fraction_target/ 에 저장."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "reports" / "v65_fraction_target"

oof = pd.read_csv(D / "oof_predictions.csv")
per = pd.read_csv(D / "per_doi_metrics.csv")

y = oof["target_value"].to_numpy(float)
base = oof["baseline_pred"].to_numpy(float)
frac = oof["fraction_pred"].to_numpy(float)

# ── 1. DOI별 MAE: 기준선 vs 분율 (짝지은 비교) ──────────────────────
b = per[per.method == "BASELINE"].set_index("doi")["mae"]
f = per[per.method == "FRACTION"].set_index("doi")["mae"]
common = b.index.intersection(f.index)
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(b[common], f[common], alpha=0.6, s=28, color="#1f77b4")
lim = max(b.max(), f.max()) * 1.05
ax.plot([0, lim], [0, lim], "r--", lw=1, label="equal (y=x)")
ax.set_xlabel("BASELINE per-DOI MAE (mg/g)")
ax.set_ylabel("FRACTION per-DOI MAE (mg/g)")
ax.set_title("Per-DOI MAE: fraction target vs baseline\n(below line = fraction better)")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.legend()
fig.tight_layout()
fig.savefig(D / "per_doi_mae_paired.png", dpi=130)
plt.close(fig)

# ── 2. 상한 위반 비교 막대 ─────────────────────────────────────────
agg = pd.read_csv(D / "aggregate_metrics.csv")
fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(["BASELINE\n(qe + clip)", "FRACTION\n(qe/cap)"],
              agg["violation_rate"].to_numpy() * 100,
              color=["#d62728", "#2ca02c"])
ax.set_ylabel("Upper-cap violation rate (%)")
ax.set_title("Physical mass-balance violations (pre-clip)")
for bar, v in zip(bars, agg["violation_rate"].to_numpy() * 100):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}%",
            ha="center", va="bottom")
ax.set_ylim(0, max(agg["violation_rate"].max() * 100 * 1.25, 5))
fig.tight_layout()
fig.savefig(D / "violation_comparison.png", dpi=130)
plt.close(fig)

# ── 3. 예측 vs 실제 (분율) ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y, base, alpha=0.4, s=20, color="#d62728", label="BASELINE")
ax.scatter(y, frac, alpha=0.4, s=20, color="#2ca02c", label="FRACTION")
lim = max(y.max(), base.max(), frac.max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("Actual qe (mg/g)")
ax.set_ylabel("Predicted qe (mg/g)")
ax.set_title("Predicted vs Actual — baseline vs fraction")
ax.legend()
fig.tight_layout()
fig.savefig(D / "predicted_vs_actual_both.png", dpi=130)
plt.close(fig)

print("plots_done")
