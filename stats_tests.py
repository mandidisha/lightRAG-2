from scipy.stats import ttest_rel, wilcoxon, chi2_contingency
import numpy as np

def interpret_p(p):
    if p is None:
        return "→ Not computed (insufficient data) ❓"
    if p < 0.01:
        return "→ Highly significant (p < 0.01) ✅"
    elif p < 0.05:
        return "→ Significant (p < 0.05) ✅"
    elif p < 0.1:
        return "→ Marginal (p < 0.1) ⚠️"
    else:
        return "→ Not significant (p ≥ 0.1) ❌"

def compare_f1_scores(f1_a, f1_b, label_a="Model A", label_b="Model B"):
    t_stat, p_ttest = ttest_rel(f1_a, f1_b)

    try:
        w_stat, p_wilcoxon = wilcoxon(f1_a, f1_b)
    except ValueError:
        p_wilcoxon = None

    return {
        "paired_ttest_p": round(p_ttest, 4),
        "wilcoxon_p": round(p_wilcoxon, 4) if p_wilcoxon is not None else None,
        "comparison": f"{label_a} vs {label_b}"
    }

def compare_em_scores(em_a, em_b, label_a="Model A", label_b="Model B"):
    return compare_f1_scores(em_a, em_b, label_a, label_b)

def chi_square_hallucination(halluc_a, halluc_b, label_a="Model A", label_b="Model B"):
    count_a = [sum(halluc_a), len(halluc_a) - sum(halluc_a)]
    count_b = [sum(halluc_b), len(halluc_b) - sum(halluc_b)]

    if count_a[0] == 0 and count_b[0] == 0:
        return {
            "chi2_p": None,
            "note": "No hallucinations in either model",
            "comparison": f"{label_a} vs {label_b}"
        }

    try:
        table = np.array([count_a, count_b])
        chi2, p_val, _, _ = chi2_contingency(table)
        return {
            "chi2_p": round(p_val, 4),
            "comparison": f"{label_a} vs {label_b}"
        }
    except ValueError as e:
        return {
            "chi2_p": None,
            "note": str(e),
            "comparison": f"{label_a} vs {label_b}"
        }

def print_all_tests(f1s, ems, hallucinations):
    print("\n=== Statistical Significance Tests ===")
    pairs = [("KG", "Dense"), ("KG", "Hybrid"), ("Dense", "Hybrid")]

    for a, b in pairs:
        print(f"\n📊 {a} vs {b}")

        f1_result = compare_f1_scores(f1s[a], f1s[b], a, b)
        print(f"F1 Paired t-test p={f1_result['paired_ttest_p']} {interpret_p(f1_result['paired_ttest_p'])}")
        print(f"F1 Wilcoxon test p={f1_result['wilcoxon_p']} {interpret_p(f1_result['wilcoxon_p'])}")

        em_result = compare_em_scores(ems[a], ems[b], a, b)
        print(f"EM Paired t-test p={em_result['paired_ttest_p']} {interpret_p(em_result['paired_ttest_p'])}")
        print(f"EM Wilcoxon test p={em_result['wilcoxon_p']} {interpret_p(em_result['wilcoxon_p'])}")

        chi_result = chi_square_hallucination(hallucinations[a], hallucinations[b], a, b)
        print(f"Hallucination Chi2 p={chi_result['chi2_p']} {interpret_p(chi_result['chi2_p'])}")
