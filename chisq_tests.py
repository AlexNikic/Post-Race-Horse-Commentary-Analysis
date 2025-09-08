"""
This script performs chisq tests for independence on the topic features (at lag 0 and lag 1) against the "is_winner" column.
It also calculates the Pearson Correlations and saves the results.
"""
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr
import numpy as np

def perform_chisq_and_correlation(model_type, num_topics):
    print(f"Loading dataset with {model_type} {num_topics} topics...")
    ds = pd.read_csv(f"data/preprocessed_raceform_with_{model_type}_{num_topics}_topic_features.csv", low_memory=False)
    mask = ds["train_val_test_split"] == "train"

    results = []
    for i in range(num_topics):
        row = {"topic": f"topic_{i}"}
        for lag in [0, 1]:
            feature_col = f"topic_{i}" if lag == 0 else f"topic_{i}_lag_1"
            pval_col_name = f"pval_lag{lag}_train"
            corr_col_name = f"corr_lag{lag}_train"

            # Chi-squared test
            contingency_table = pd.crosstab(ds.loc[mask, feature_col], ds.loc[mask, "is_winner"])
            
            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                corrected_p = min(num_topics * p, 1.0) # Bonferroni correction
            else:
                corrected_p = np.nan
            
            row[pval_col_name] = round(corrected_p, 4)

            # Pearson Correlation
            try:
                corr, _ = pearsonr(ds.loc[mask, feature_col], ds.loc[mask, "is_winner"])
            except Exception:
                corr = np.nan
            row[corr_col_name] = round(corr, 4) if pd.notnull(corr) else np.nan
        
        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"evaluation_metrics/chisq_results_{model_type}_{num_topics}.csv", index=False)


if __name__ == "__main__":
    model_type = "lda"
    for num_topics in [10, 20, 30, 40, 50]:
        perform_chisq_and_correlation(model_type, num_topics)
        print(f"Chisq and correlation results saved for {model_type} with {num_topics} topics.")