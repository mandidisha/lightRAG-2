from scipy.stats import ttest_rel, wilcoxon, chi2_contingency
import numpy as np


def interpret_p_text(p, metric_name, a, b):
    """
    Return a clause explaining the p-value significance level in narrative form.
    """
    if p is None:
        return f"could not be computed for {metric_name} (insufficient data)."
    if p < 0.01:
        sig = "highly significant (p < 0.01)"
    elif p < 0.05:
        sig = "significant (p < 0.05)"
    elif p < 0.1:
        sig = "marginally significant (p < 0.1)"
    else:
        sig = "not significant (p ≥ 0.1)"
    return f"returned p = {p:.4f}, which is {sig}."


def narrative_f1_test(f1_a, f1_b, label_a, label_b):
    """
    Perform paired t-test and Wilcoxon test on F1 scores and print narrative.
    """
    t_stat, p_t = ttest_rel(f1_a, f1_b)
    try:
        _, p_w = wilcoxon(f1_a, f1_b)
    except ValueError:
        p_w = None

    print(f"\n▶ Comparing F1 between {label_a} and {label_b}:")
    print("  - Paired t-test", interpret_p_text(p_t, "F1", label_a, label_b))
    if p_w is not None:
        print("  - Wilcoxon test", interpret_p_text(p_w, "F1", label_a, label_b))


def narrative_em_test(em_a, em_b, label_a, label_b):
    """
    Perform paired t-test and Wilcoxon test on Exact Match scores and print narrative.
    """
    t_stat, p_t = ttest_rel(em_a, em_b)
    try:
        _, p_w = wilcoxon(em_a, em_b)
    except ValueError:
        p_w = None

    print(f"\n▶ Comparing Exact Match (EM) between {label_a} and {label_b}:")
    print("  - Paired t-test", interpret_p_text(p_t, "EM", label_a, label_b))
    if p_w is not None:
        print("  - Wilcoxon test", interpret_p_text(p_w, "EM", label_a, label_b))


def narrative_halluc_test(halluc_a, halluc_b, label_a, label_b):
    """
    Perform chi-square test on hallucination rates and print narrative.
    """
    count_a = [sum(halluc_a), len(halluc_a) - sum(halluc_a)]
    count_b = [sum(halluc_b), len(halluc_b) - sum(halluc_b)]

    print(f"\n▶ Comparing hallucination rates between {label_a} and {label_b}:")
    if count_a[0] == 0 and count_b[0] == 0:
        print("  - No hallucinations observed in either system; chi-square not applicable.")
        return

    table = np.array([count_a, count_b])
    chi2, p_val, _, _ = chi2_contingency(table)
    print("  - Chi-square test", interpret_p_text(p_val, "hallucinations", label_a, label_b))


def print_all_tests(f1s, ems, hallucinations):
    """
    For each pair among KG, Dense, Hybrid:
      1) Compare F1 with t-test & Wilcoxon
      2) Compare EM with t-test & Wilcoxon
      3) Compare hallucination rates with chi-square
    Output narrative sentences for each comparison.
    """
    print("\n=== Statistical Significance Narrative ===")
    pairs = [("KG", "Dense"), ("KG", "Hybrid"), ("Dense", "Hybrid")]

    for a, b in pairs:
        narrative_f1_test(f1s[a], f1s[b], a, b)
        narrative_em_test(ems[a], ems[b], a, b)
        narrative_halluc_test(hallucinations[a], hallucinations[b], a, b)
