# Holistic PR Analysis Meta-Report
Generated on: 2025-04-20 20:59:53
Based on analysis output from: `./thesis_holistic_analysis_v7/`

---

## 1. Introduction & Scope
- **Total Repositories Scanned:** 82
- **Total PRs Analyzed (within timeframe):** 212,687
- **Analysis Timeframe:** 2022-01-01 to 2025-04-20
- **Repositories Included in Statistical Comparisons (>= 10 PRs):** 77
- **Identified Repository Clusters:** 3
- **Competency Scores Analyzed:** Architecture Impact, Code Quality, Collaboration, Maintainability, Technical Leadership, User Impact

---

## 2. Overall Competency Score Distributions
The following table summarizes the distribution of each competency score across *all* analyzed PRs within the timeframe.
| Competency           |   Mean |   Std Dev |   Min |   25% (Q1) |   Median (Q2) |   75% (Q3) |   Max |
|:---------------------|-------:|----------:|------:|-----------:|--------------:|-----------:|------:|
| Architecture Impact  |  0.296 |     0.265 | 0.000 |      0.089 |         0.186 |      0.445 | 0.998 |
| Code Quality         |  0.419 |     0.318 | 0.001 |      0.123 |         0.327 |      0.731 | 0.999 |
| Collaboration        |  0.450 |     0.318 | 0.001 |      0.148 |         0.388 |      0.761 | 0.999 |
| Maintainability      |  0.289 |     0.275 | 0.001 |      0.077 |         0.169 |      0.442 | 0.999 |
| Technical Leadership |  0.244 |     0.236 | 0.001 |      0.072 |         0.148 |      0.337 | 0.998 |
| User Impact          |  0.306 |     0.276 | 0.001 |      0.083 |         0.191 |      0.483 | 0.998 |

**Highlights:**
- Highest average score: **Collaboration** (`0.450`).
- Lowest average score: **Technical Leadership** (`0.244`).
- Most score variability: **Code Quality** (Std Dev: `0.318`).
- Least score variability: **Technical Leadership** (Std Dev: `0.236`).

*Refer to plots in `plots/descriptive/` for visual distributions.*

---

## 3. Repository Differences (Kruskal-Wallis)
Kruskal-Wallis tests were performed to check for significant differences in median competency scores across the 77 repositories meeting the minimum PR threshold.
Significance level (alpha) = 0.05, with FDR Benjamini-Hochberg correction.

**Significant differences *between repositories* found for:**
- **Architecture Impact** (p_corr = 0)
- **Code Quality** (p_corr = 0)
- **Collaboration** (p_corr = 0)
- **Maintainability** (p_corr = 0)
- **Technical Leadership** (p_corr = 0)
- **User Impact** (p_corr = 0)

*Refer to boxplots in `plots/comparison/` for visual comparisons.*

---

## 4. Competency Correlations
### Repository-Level (Based on Mean Scores) Correlations

**Top 3 Positive Correlations:**
- Architecture Impact & User Impact: 0.392 (Weak)
- Architecture Impact & Technical Leadership: 0.349 (Weak)
- Technical Leadership & User Impact: 0.242 (Weak)

**Top 3 Negative Correlations:**
- Architecture Impact & Collaboration: -0.221 (Weak)
- Collaboration & Technical Leadership: -0.159 (Weak)
- Collaboration & Maintainability: -0.150 (Weak)


### PR-Level (Across All Individual PRs) Correlations

**Top 3 Positive Correlations:**
- Architecture Impact & User Impact: 0.415 (Moderate)
- Architecture Impact & Technical Leadership: 0.383 (Weak)
- Technical Leadership & User Impact: 0.346 (Weak)

**Top 3 Negative Correlations:**
- *None found.*

**Interpretation Note:** Comparing repository-level and PR-level correlations can be insightful. Repo-level correlations show relationships between the *average* tendencies of repositories, while PR-level correlations reflect relationships within individual units of work across the entire dataset.

*Refer to heatmaps in `plots/correlation/` for repo-level visualization.*

---

## 5. Repository Archetypes (PCA & Clustering)
Principal Component Analysis (PCA) and K-Means clustering (k=3) were used to identify potential repository archetypes based on mean competency scores.
Refer to the analysis log (`holistic_analysis_log_v5.log`) or PCA plots for explained variance ratios.

**Cluster Characteristics:**
### Cluster 0.0
- **Size:** 40 repositories
- **PR Volume:** Mean ~1,470, Median ~793 PRs per repo
- **Dominant PR Size Categories:**
  - 301 - 1,006 PRs: 35.0%
  - 19 - 301 PRs: 27.5%
  - 1,006 - 3,013 PRs: 27.5%
  - 3,013 - 24,721 PRs: 10.0%
- **Competency Profile (Mean Scores vs Overall):**

---

### Cluster 1.0
- **Size:** 27 repositories
- **PR Volume:** Mean ~3,749, Median ~1,814 PRs per repo
- **Dominant PR Size Categories:**
  - 3,013 - 24,721 PRs: 33.3%
  - 19 - 301 PRs: 29.6%
  - 1,006 - 3,013 PRs: 22.2%
  - 301 - 1,006 PRs: 14.8%
- **Competency Profile (Mean Scores vs Overall):**

---

### Cluster 2.0
- **Size:** 10 repositories
- **PR Volume:** Mean ~5,265, Median ~3,720 PRs per repo
- **Dominant PR Size Categories:**
  - 3,013 - 24,721 PRs: 60.0%
  - 1,006 - 3,013 PRs: 20.0%
  - 19 - 301 PRs: 10.0%
  - 301 - 1,006 PRs: 10.0%
- **Competency Profile (Mean Scores vs Overall):**

---


*Refer to PCA/Cluster plots (static and interactive) in `plots/pca_cluster/` for visualization.*

---

## 6. Manual Ranking Files
- Found Rene's ranking file (`(Rene) Ranking_PRs - Sheet1.csv`) with 35 entries.
- Found Chris's ranking file (`(Chris) Ranking_PRs - Sheet1.csv`) with 35 entries.

*Note: Direct comparison between manual ranks and automated scores requires matching PR identifiers and potentially standardizing manual scores. This was not performed in this automated report.*

---

## 7. Time Series Analysis
Time series decomposition (trend, seasonality, residuals) was performed on the monthly average scores for each competency.
This helps identify underlying patterns or shifts in PR quality aspects over the analyzed timeframe.

*Refer to individual component plots in `plots/timeseries/` for details.*

---

## 8. Key Takeaways & Conclusion
This automated analysis provides a multi-faceted view of PR quality characteristics across the studied repositories.

**Summary Points (Automated Extraction):**
- Significant differences **between repositories** were observed for: Architecture Impact, Code Quality, Collaboration, Maintainability, Technical Leadership, User Impact.
- Clustering identified **3 distinct repository archetypes** based on their average competency profiles and PR volume (see Section 5 for details).
- Repo-level analysis suggests potential relationships, e.g., Architecture Impact & User Impact: 0.392 (Weak).
- PR-level analysis showed different or similar patterns, e.g., Architecture Impact & User Impact: 0.415 (Moderate).
- Overall, **Collaboration** received the highest average score, while **Technical Leadership** received the lowest.

Further investigation could delve into the qualitative differences between clusters, the impact of specific repository practices, or the validation of automated scores against manual assessments.

---
