# Exploratory Data Analysis: Wine Chemistry

## 1. Dataset overview
- **178 wines**, **13 numeric chemical measurements**, grouped into 3 grape cultivars.
- Missing values: **0**. Duplicate rows: **0**.
- Cultivar sizes: Cultivar 0: 59, Cultivar 1: 71, Cultivar 2: 48 (moderately imbalanced, not extreme).

## 2. Data quality and distributions
- Features are on very different scales (e.g. proline averages 747, hue averages 0.96), so any distance-based or regularised method needs scaling.
- 12 of 13 features fail a Shapiro-Wilk normality test at 5%. Treat this loosely: with the cultivars mixed together, some non-normality is expected.
- Notably skewed features (|skew| > 0.75): magnesium (+1.10), malic_acid (+1.04), color_intensity (+0.87), proline (+0.77).
- IQR outliers per feature range from 0 to 4; most affected: alcalinity_of_ash (4 wines). With only 178 rows I would keep them unless they are measurement errors.

## 3. Correlations
- Strongest positive pairs: total_phenols vs flavanoids (r = +0.86); flavanoids vs od280/od315_of_diluted_wines (r = +0.79); total_phenols vs od280/od315_of_diluted_wines (r = +0.70).
- Strongest negative pairs: malic_acid vs hue (r = -0.56); flavanoids vs nonflavanoid_phenols (r = -0.54); color_intensity vs hue (r = -0.52).
- Several phenolic measures move together, so a model using all of them has redundant inputs (multicollinearity matters for linear models, much less for trees).
- Correlation here is measured across all three cultivars pooled, so part of it reflects group differences, not a within-group relationship.

## 4. Key influencing factors (what separates the cultivars)
Ranked by eta squared (share of a feature's variance explained by cultivar), all from one-way ANOVA:

| feature                      |     F |       p |   eta² |
|:-----------------------------|------:|--------:|-------:|
| flavanoids                   | 233.9 | 3.6e-50 |  0.728 |
| proline                      | 207.9 | 5.8e-47 |  0.704 |
| od280/od315_of_diluted_wines | 190   | 1.4e-44 |  0.685 |
| alcohol                      | 135.1 | 3.3e-36 |  0.607 |
| color_intensity              | 120.7 | 1.2e-33 |  0.58  |
| hue                          | 101.3 | 5.9e-30 |  0.537 |

Mean values of the top four features by cultivar:

| cultivar   |   flavanoids |   proline |   od280/od315_of_diluted_wines |   alcohol |
|:-----------|-------------:|----------:|-------------------------------:|----------:|
| Cultivar 0 |         2.98 |   1115.71 |                           3.16 |     13.74 |
| Cultivar 1 |         2.08 |    519.51 |                           2.79 |     12.28 |
| Cultivar 2 |         0.78 |    629.9  |                           1.68 |     13.15 |

- **flavanoids** and **proline** are the strongest separators.
- 13 of 13 features differ significantly across cultivars at 5% (smallest eta² = 0.12), but the top few do most of the work.

## 5. Multivariate structure (PCA)
- PC1 explains 36% of variance and PC2 19%; together 55%.
- It takes 10 components to reach 95%, so the data is only moderately compressible.
- The three cultivars form largely distinct clusters in the first two components, with some overlap between Cultivars 0 and 1, which suggests a classifier should do well.
- PC1 is driven mostly by: flavanoids, total_phenols, od280/od315_of_diluted_wines.

## 6. Takeaways
1. The data is clean (no missing values, no duplicates), so effort goes to scaling and feature selection, not cleaning.
2. A handful of chemical measurements (flavanoids, proline, od280/od315_of_diluted_wines) carry most of the information distinguishing cultivars.
3. Correlated phenolic features are partly redundant; consider dropping or combining some for linear models.
4. Cultivars cluster well in PCA space, so supervised prediction is likely to work well (a natural next step is the modeling project).

## 7. Limitations
- Only 178 samples: correlations and group differences are estimated with real uncertainty, and the outlier counts are small.
- Statistical significance with ANOVA is easy to reach on well-separated groups; use the effect size (eta²) to judge importance.
- This analysis is descriptive. It shows association, not what causes cultivar differences.
