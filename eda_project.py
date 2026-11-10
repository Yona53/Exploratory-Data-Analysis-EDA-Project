"""
Exploratory Data Analysis: Wine chemistry dataset (178 wines, 3 cultivars)
Run: python eda_project.py
Outputs: 4 figures + eda_report.md (all numbers computed from the data)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUT = "/mnt/user-data/outputs"
sns.set_theme(style="whitegrid", font_scale=0.95)
pal = sns.color_palette("deep")

# ------------------------------------------------------------------
# 1. LOAD + STRUCTURE
# ------------------------------------------------------------------
w = load_wine(as_frame=True)
df = w.frame.copy()
names = {0: "Cultivar 0", 1: "Cultivar 1", 2: "Cultivar 2"}
df["cultivar"] = df["target"].map(names)
features = list(w.feature_names)

n_rows, n_cols = df.shape[0], len(features)
n_missing = int(df[features].isna().sum().sum())
n_dupes = int(df.duplicated().sum())
class_counts = df["cultivar"].value_counts().sort_index()

# ------------------------------------------------------------------
# 2. STATISTICAL SUMMARY
# ------------------------------------------------------------------
desc = df[features].describe().T
desc["skew"] = df[features].skew()
desc["cv_%"] = desc["std"] / desc["mean"] * 100       # relative spread

# outliers (Tukey 1.5 x IQR)
def iqr_outliers(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum())
desc["outliers"] = [iqr_outliers(df[f]) for f in features]

# normality (Shapiro-Wilk)
desc["shapiro_p"] = [stats.shapiro(df[f])[1] for f in features]

# ------------------------------------------------------------------
# 3. FIGURE 1: distributions
# ------------------------------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(18, 13))
for ax, f in zip(axes.flat, features):
    sns.histplot(df[f], kde=True, ax=ax, color=pal[0])
    ax.set_title(f"{f}\nskew {df[f].skew():+.2f}", fontsize=10)
    ax.set_xlabel(""); ax.set_ylabel("")
for ax in axes.flat[len(features):]:
    ax.axis("off")
fig.suptitle("Feature distributions: most are roughly bell-shaped; a few are skewed", fontsize=18, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(f"{OUT}/eda_1_distributions.png", dpi=110); plt.close()

# ------------------------------------------------------------------
# 4. CORRELATIONS
# ------------------------------------------------------------------
corr = df[features].corr()
pairs = (corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack()
         .rename("r").reset_index().rename(columns={"level_0": "a", "level_1": "b"}))
pairs["abs"] = pairs["r"].abs()
top_pairs = pairs.nlargest(6, "abs")

fig, ax = plt.subplots(1, 2, figsize=(19, 8), gridspec_kw={"width_ratios": [1.5, 1]})
mask = np.triu(np.ones_like(corr, bool))
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=True, fmt=".2f",
            annot_kws={"size": 7}, ax=ax[0], cbar_kws={"shrink": .7})
ax[0].set_title("Correlation matrix (Pearson)", fontweight="bold")
tp = top_pairs.assign(label=lambda d: d.a + "\nvs " + d.b).iloc[::-1]
ax[1].barh(tp["label"], tp["r"], color=[pal[3] if r < 0 else pal[0] for r in tp["r"]])
ax[1].set_title("Six strongest feature pairs", fontweight="bold"); ax[1].set_xlabel("Pearson r")
plt.tight_layout(); plt.savefig(f"{OUT}/eda_2_correlations.png", dpi=110); plt.close()

# ------------------------------------------------------------------
# 5. KEY INFLUENCING FACTORS: which features separate the cultivars?
#    ANOVA F-test + effect size (eta squared)
# ------------------------------------------------------------------
rows = []
for f in features:
    groups = [g[f].values for _, g in df.groupby("cultivar")]
    F, p = stats.f_oneway(*groups)
    grand = df[f].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((df[f] - grand) ** 2).sum()
    rows.append({"feature": f, "F": F, "p": p, "eta_sq": ss_between / ss_total})
sep = pd.DataFrame(rows).sort_values("eta_sq", ascending=False).reset_index(drop=True)
top_feats = sep["feature"].head(4).tolist()

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 4)
ax0 = fig.add_subplot(gs[:, 0])
s = sep.iloc[::-1]
ax0.barh(s["feature"], s["eta_sq"], color=pal[2])
ax0.set_title("How much of each feature's variance\nis explained by cultivar (eta²)", fontweight="bold")
axs = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
for ax, f in zip(axs, top_feats):
    sns.boxplot(data=df, x="cultivar", y=f, ax=ax, palette="deep", hue="cultivar", legend=False)
    sns.stripplot(data=df, x="cultivar", y=f, ax=ax, color="k", alpha=.25, size=3)
    ax.set_title(f, fontweight="bold"); ax.set_xlabel("")
ax_s = fig.add_subplot(gs[:, 3])
sns.scatterplot(data=df, x=top_feats[0], y=top_feats[1], hue="cultivar", palette="deep", ax=ax_s, s=45)
ax_s.set_title(f"Top two features together", fontweight="bold")
fig.suptitle("Key influencing factors: what distinguishes the three cultivars", fontsize=18, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(f"{OUT}/eda_3_key_factors.png", dpi=110); plt.close()

# ------------------------------------------------------------------
# 6. MULTIVARIATE STRUCTURE (PCA)
# ------------------------------------------------------------------
Z = StandardScaler().fit_transform(df[features])
pca = PCA().fit(Z)
pcs = pca.transform(Z)
evr = pca.explained_variance_ratio_
n95 = int(np.argmax(np.cumsum(evr) >= 0.95) + 1)
loadings = pd.DataFrame(pca.components_[:2].T, index=features, columns=["PC1", "PC2"])

fig, ax = plt.subplots(1, 3, figsize=(19, 6))
sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=df["cultivar"], palette="deep", ax=ax[0], s=50)
ax[0].set_xlabel(f"PC1 ({evr[0]:.0%} of variance)"); ax[0].set_ylabel(f"PC2 ({evr[1]:.0%})")
ax[0].set_title("Wines projected onto two principal components", fontweight="bold")
ax[1].plot(range(1, len(evr) + 1), np.cumsum(evr), marker="o", color=pal[0])
ax[1].axhline(.95, color="r", ls="--", alpha=.6)
ax[1].set_xlabel("Number of components"); ax[1].set_ylabel("Cumulative variance explained")
ax[1].set_title(f"{n95} components capture 95% of variance", fontweight="bold")
l = loadings["PC1"].sort_values()
ax[2].barh(l.index, l.values, color=[pal[3] if v < 0 else pal[0] for v in l.values])
ax[2].set_title("What drives PC1 (loadings)", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/eda_4_pca.png", dpi=110); plt.close()

# ------------------------------------------------------------------
# 7. REPORT (every number computed above)
# ------------------------------------------------------------------
skewed = desc[desc["skew"].abs() > 0.75].sort_values("skew", key=abs, ascending=False)
non_normal = int((desc["shapiro_p"] < 0.05).sum())
prof = df.groupby("cultivar")[top_feats].mean().round(2)
pos = pairs.nlargest(3, "r")
neg = pairs.nsmallest(3, "r")

def pair_txt(d):
    return "; ".join(f"{r.a} vs {r.b} (r = {r.r:+.2f})" for r in d.itertuples()) or "none among the top six"

rep = f"""# Exploratory Data Analysis: Wine Chemistry

## 1. Dataset overview
- **{n_rows} wines**, **{n_cols} numeric chemical measurements**, grouped into 3 grape cultivars.
- Missing values: **{n_missing}**. Duplicate rows: **{n_dupes}**.
- Cultivar sizes: {", ".join(f"{k}: {v}" for k, v in class_counts.items())} (moderately imbalanced, not extreme).

## 2. Data quality and distributions
- Features are on very different scales (e.g. proline averages {desc.loc['proline','mean']:.0f}, hue averages {desc.loc['hue','mean']:.2f}), so any distance-based or regularised method needs scaling.
- {non_normal} of {n_cols} features fail a Shapiro-Wilk normality test at 5%. Treat this loosely: with the cultivars mixed together, some non-normality is expected.
- Notably skewed features (|skew| > 0.75): {", ".join(f"{i} ({r['skew']:+.2f})" for i, r in skewed.iterrows()) or "none"}.
- IQR outliers per feature range from {int(desc['outliers'].min())} to {int(desc['outliers'].max())}; most affected: {desc['outliers'].idxmax()} ({int(desc['outliers'].max())} wines). With only {n_rows} rows I would keep them unless they are measurement errors.

## 3. Correlations
- Strongest positive pairs: {pair_txt(pos)}.
- Strongest negative pairs: {pair_txt(neg)}.
- Several phenolic measures move together, so a model using all of them has redundant inputs (multicollinearity matters for linear models, much less for trees).
- Correlation here is measured across all three cultivars pooled, so part of it reflects group differences, not a within-group relationship.

## 4. Key influencing factors (what separates the cultivars)
Ranked by eta squared (share of a feature's variance explained by cultivar), all from one-way ANOVA:

{sep.head(6).assign(F=lambda d: d.F.round(1), eta_sq=lambda d: d.eta_sq.round(3), p=lambda d: d.p.map(lambda v: f"{v:.1e}")).rename(columns={"eta_sq": "eta²"}).to_markdown(index=False)}

Mean values of the top four features by cultivar:

{prof.to_markdown()}

- **{top_feats[0]}** and **{top_feats[1]}** are the strongest separators.
- {int((sep["p"] < 0.05).sum())} of {n_cols} features differ significantly across cultivars at 5% (smallest eta² = {sep["eta_sq"].min():.2f}), but the top few do most of the work.

## 5. Multivariate structure (PCA)
- PC1 explains {evr[0]:.0%} of variance and PC2 {evr[1]:.0%}; together {evr[:2].sum():.0%}.
- It takes {n95} components to reach 95%, so the data is only moderately compressible.
- The three cultivars form largely distinct clusters in the first two components, with some overlap between Cultivars 0 and 1, which suggests a classifier should do well.
- PC1 is driven mostly by: {", ".join(loadings['PC1'].abs().nlargest(3).index)}.

## 6. Takeaways
1. The data is clean (no missing values, no duplicates), so effort goes to scaling and feature selection, not cleaning.
2. A handful of chemical measurements ({", ".join(top_feats[:3])}) carry most of the information distinguishing cultivars.
3. Correlated phenolic features are partly redundant; consider dropping or combining some for linear models.
4. Cultivars cluster well in PCA space, so supervised prediction is likely to work well (a natural next step is the modeling project).

## 7. Limitations
- Only {n_rows} samples: correlations and group differences are estimated with real uncertainty, and the outlier counts are small.
- Statistical significance with ANOVA is easy to reach on well-separated groups; use the effect size (eta²) to judge importance.
- This analysis is descriptive. It shows association, not what causes cultivar differences.
"""
open(f"{OUT}/eda_report.md", "w").write(rep)
print(rep)
