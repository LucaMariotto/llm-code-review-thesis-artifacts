# Survey Analysis Meta Results Summary (V7 - Extended Hypotheses)
Generated: 2025-04-20 22:39:42
Analysis Version: v7
Significance Level (Alpha): 0.05

Based on **38 unique participants** and **228 individual comparison responses**.

*Note: This report summarizes key findings. For full details and exploratory analyses, consult the CSV files and plots in the 'output_data' and 'visualizations' subdirectories.*

## 1. Specific Hypothesis Test Results (Binomial Test)
This section tests specific directional hypotheses about preference (e.g., 'Is IO preferred over O?').
The test compares the proportion preferring the 'Expected Winner' vs. the 'Expected Loser' among participants who chose one or the other (N Winner + N Loser).
P-values test if the preference for the Winner is significantly greater than 50% (one-sided test, alpha = 0.05).
An extended set (H7-H12) tests the *opposite* direction to help interpret non-significant results in H1-H6.

| Hypothesis | Comparison                         | N Winner | N Loser | N Neither | N Other | % Winner (of W+L) | P-value | Sig. |
| :--------- | :--------------------------------- | :------- | :------ | :-------- | :------ | :---------------- | :------ | :--- |
| **H1: O > D** | Original vs Degraded | 15 | 16 | 7 | 0 | 48.4% | 0.640 | **(ns)** |
| **H2: IO > O** | Original vs Improved_Original | 20 | 11 | 7 | 0 | 64.5% | 0.075 | **(ns)** |
| **H3: ID > D** | Degraded vs Improved_Degraded | 23 | 11 | 4 | 0 | 67.6% | 0.029 | ***** |
| **H4: IO > ID** | Improved_Degraded vs Improved_Original | 20 | 11 | 7 | 0 | 64.5% | 0.075 | **(ns)** |
| **H5: ID > O** | Original vs Improved_Degraded | 19 | 8 | 11 | 0 | 70.4% | 0.026 | ***** |
| **H6: IO > D** | Degraded vs Improved_Original | 22 | 13 | 3 | 0 | 62.9% | 0.088 | **(ns)** |
| **H7: D > O** | Original vs Degraded | 16 | 15 | 7 | 0 | 51.6% | 0.500 | **(ns)** |
| **H8: O > IO** | Original vs Improved_Original | 11 | 20 | 7 | 0 | 35.5% | 0.965 | **(ns)** |
| **H9: D > ID** | Degraded vs Improved_Degraded | 11 | 23 | 4 | 0 | 32.4% | 0.988 | **(ns)** |
| **H10: ID > IO** | Improved_Degraded vs Improved_Original | 11 | 20 | 7 | 0 | 35.5% | 0.965 | **(ns)** |
| **H11: O > ID** | Original vs Improved_Degraded | 8 | 19 | 11 | 0 | 29.6% | 0.990 | **(ns)** |
| **H12: D > IO** | Degraded vs Improved_Original | 13 | 22 | 3 | 0 | 37.1% | 0.955 | **(ns)** |

**Interpretation Guidance:**
- **Significant Result (*, **, ***):** There is sufficient statistical evidence (at alpha=0.05) to support the stated directional hypothesis (e.g., H1: O > D).
- **Non-Significant Result (ns):** There is *not* sufficient statistical evidence (at alpha=0.05) to conclude the stated directional hypothesis is true. This *does not* prove the opposite is true or that there is absolutely no difference (absence of evidence is not evidence of absence).
  - Look at the **% Winner**: If it's clearly above 50% (e.g., >60%) and p-value is low (e.g., < 0.15), consider this a **potential trend** favoring the winner, requiring discussion and support from qualitative data.
  - Check the **opposing hypothesis** (e.g., H7 for H1): If *both* H1 (A>B) and H7 (B>A) are non-significant, this strengthens the conclusion that **no statistically significant difference in preference was detected** between A and B.
- **Low Counts:** Tests with low 'N Winner + N Loser' may lack statistical power; interpret non-significant results with extra caution, as a real difference might exist but be undetectable.

**Interpretation of Specific Hypothesis Pairs:**
- **(H1: O > D vs H7: D > O):** Test results: [H1: O > D (48.4% pref, p=0.640 (ns))]; [H7: D > O (51.6% pref, p=0.500 (ns))]. **Conclusion: No statistically significant difference in preference detected between the two versions.**
- **(H2: IO > O vs H8: O > IO):** Test results: [H2: IO > O (64.5% pref, p=0.075 (ns))]; [H8: O > IO (35.5% pref, p=0.965 (ns))]. **Conclusion: No statistically significant difference in preference detected between the two versions.** (However, note the trend favoring H2 with 64.5% preference, p=0.075. Qualitative insights needed).
- **(H3: ID > D vs H9: D > ID):** Test results: [H3: ID > D (67.6% pref, p=0.029 *)]; [H9: D > ID (32.4% pref, p=0.988 (ns))]. **Conclusion: Evidence supports H3: ID > D.**
- **(H4: IO > ID vs H10: ID > IO):** Test results: [H4: IO > ID (64.5% pref, p=0.075 (ns))]; [H10: ID > IO (35.5% pref, p=0.965 (ns))]. **Conclusion: No statistically significant difference in preference detected between the two versions.** (However, note the trend favoring H4 with 64.5% preference, p=0.075. Qualitative insights needed).
- **(H5: ID > O vs H11: O > ID):** Test results: [H5: ID > O (70.4% pref, p=0.026 *)]; [H11: O > ID (29.6% pref, p=0.990 (ns))]. **Conclusion: Evidence supports H5: ID > O.**
- **(H6: IO > D vs H12: D > IO):** Test results: [H6: IO > D (62.9% pref, p=0.088 (ns))]; [H12: D > IO (37.1% pref, p=0.955 (ns))]. **Conclusion: No statistically significant difference in preference detected between the two versions.** (However, note the trend favoring H6 with 62.9% preference, p=0.088. Qualitative insights needed).

*See `specific_hypothesis_tests_*.csv` for detailed counts and statistics.*

## 2. Overall Alignment with Primary Hypotheses (Descriptive Overview)
This shows the overall percentage breakdown of responses based on whether they **aligned** with the primary hypothesis (H1-H6), **contradicted** it, were **neutral** ('Neither'), or **other/unknown**.

| Comparison (Hypothesis)              | % Aligned (Success) (N) | % Contradicted (Failure) (N) | % Neutral (N) | % Other/Unknown (N) |
| :----------------------------------- | :---------------------- | :--------------------------- | :------------ | :------------------ |
| **Original vs Degraded**<br>(H1: O > D) | 39.5% (15) | 42.1% (16) | 18.4% (7) | 0.0% (0) |
| **Original vs Improved_Degraded**<br>(H5: ID > O) | 50.0% (19) | 21.1% (8) | 28.9% (11) | 0.0% (0) |
| **Degraded vs Improved_Degraded**<br>(H3: ID > D) | 60.5% (23) | 28.9% (11) | 10.5% (4) | 0.0% (0) |
| **Improved_Degraded vs Improved_Original**<br>(H4: IO > ID) | 52.6% (20) | 28.9% (11) | 18.4% (7) | 0.0% (0) |
| **Original vs Improved_Original**<br>(H2: IO > O) | 52.6% (20) | 28.9% (11) | 18.4% (7) | 0.0% (0) |
| **Degraded vs Improved_Original**<br>(H6: IO > D) | 57.9% (22) | 34.2% (13) | 7.9% (3) | 0.0% (0) |

*This provides a high-level view. For inferential tests on preference direction, see Section 1.*
*See `overall_alignment_descriptive_analysis.csv` and `overall_alignment_comparison_*.png` plots for details.*

## 3. Influence of Experience & LLM Satisfaction (Exploratory Analysis)
(Focus on significant results, p < 0.05, from Spearman and Kruskal-Wallis tests)
*Caution: These are exploratory correlations. Interpret significant findings carefully, considering the number of tests performed.*

**Spearman Correlations (Preference Score vs. Variable):**
- *No significant Spearman correlations found at p < 0.05.*
  *(See `preference_correlations_spearman.csv` for full results)*

**Kruskal-Wallis Tests (Variable Distribution across Preference Groups):**
- *No significant differences found in variable distributions across preference groups using Kruskal-Wallis (p < 0.05).*
  *(See `preference_correlations_kruskal_wallis.csv` and associated boxplots for details)*

*Overall, the exploratory analysis did not reveal strong significant associations between Experience/LLM Satisfaction and preferences in the tested comparisons (at p < 0.05).*

## 4. Qualitative Insights (Essential Next Step)
While the quantitative analysis shows *what* preferences exist (or don't significantly), the qualitative remarks explain *why*. This is crucial for understanding the nuances, especially for non-significant results or observed trends.
**Action Required:** Perform thematic analysis on the extracted remarks in `output_data/qualitative_remarks_context.csv`.

**Key questions to investigate in the remarks:**
- What specific aspects made users prefer the Original, Degraded, ID, or IO versions?
- **Clarity & Conciseness:** Was the language easy to understand? Was there too much/too little detail?
- **Tone:** Was the tone perceived positively (friendly, professional) or negatively (robotic, verbose, informal)?
- **Structure:** Did elements like summaries, checklists, bullet points help or hinder understanding?
- **Accuracy & Usefulness:** Did participants comment on the perceived correctness or helpfulness of the generated content?
- **Specific Examples:** Are there recurring comments about specific phrases, sections, or generated elements (e.g., 'word noise', 'human touch', 'good structure')?
- **Reasons for 'Neither':** What trade-offs or lack of difference led participants to choose 'Neither'? This is very important if significance wasn't found.
- **Reasons for Trends (e.g., p<0.15):** For comparisons showing trends (like H2: IO > O), what do the comments say about *why* people leaned that way?
- **Explaining Lack of Difference:** For pairs where *neither* direction was significant (e.g., H1/H7), what reasons do comments give for similarity or ambivalence?
- **Contradictions:** Why did some participants contradict the expected hypothesis (e.g., prefer Degraded over Original, or prefer O over IO)?
- **Demographic Patterns:** Do comments suggest different priorities based on experience, role, or LLM satisfaction? (Refer to `alignment_by_*` plots for quantitative hints).