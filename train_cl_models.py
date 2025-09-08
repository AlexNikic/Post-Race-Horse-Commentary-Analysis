import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit
import numpy as np
from joblib import load
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# NOTE: In an ideal world, I would like to be able to save the CL models, but statsmodels does not support this yet.

def add_prob_and_logit(col_prefix, model):
    ds[f"{col_prefix}_win_prob"] = model.predict_proba(ds[feature_cols])[:, 1]
    # Normalize per race
    ds[f"{col_prefix}_win_prob"] = (
        ds.groupby("race_id")[f"{col_prefix}_win_prob"].transform(lambda x: x / x.sum())
    ).clip(1e-12, 1-1e-12)
    ds[f"logit_{col_prefix}_win_prob"] = np.log(ds[f"{col_prefix}_win_prob"] / (1 - ds[f"{col_prefix}_win_prob"])).clip(1e-12, 1-1e-12)

def predict_conditional_logit(result, ds, feature_cols):
    beta = result.params.values
    X = ds[feature_cols]

    util = np.dot(X, beta)
    ds["exp_util"] = np.exp(util)
    
    denom = ds.groupby("race_id")["exp_util"].transform("sum")
    probs = ds["exp_util"] / denom
    
    return probs

def plot_strategy(strategy, ds, path):
    # Ensure dataset is in chronological order
    ds = ds.sort_values(by=['date', 'race_id']).reset_index(drop=True)
    ds["date"] = pd.to_datetime(ds["date"])

    fig, ax = plt.subplots()

    for strat in strategy:
        model_name = strat["model"]
        ds[f"pnl_{model_name}"] = 0.0

        wagers = strat["wagers"].values  # convert to numpy array
        is_winner = ds["is_winner"].values  # numpy array

        win_mask = (wagers > 0) & (is_winner == 1)
        lose_mask = (wagers > 0) & (is_winner == 0)

        ds.loc[win_mask, f"pnl_{model_name}"] = wagers[win_mask] * (ds.loc[win_mask, "sp"].values - 1)
        ds.loc[lose_mask, f"pnl_{model_name}"] = wagers[lose_mask] * (-1)

        ds[f"cum_pnl_{model_name}"] = ds[f"pnl_{model_name}"].cumsum()

        ax.plot(ds["date"], ds[f"cum_pnl_{model_name}"], label=model_name, linestyle='-')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))

    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL")
    plt.title("Strategy PnL Over Time")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(path) 

if __name__ == "__main__":
    for model_type in ["embeddings"]: #["lda", "embeddings"]:
        number_of_topics = [10, 20, 30, 40, 50] if model_type == "lda" else [128]
        for num_topics in number_of_topics:
            print(f"Processing {num_topics} topics...")
                    
            # Load dataset
            if model_type == "lda":
                path = f"data/preprocessed_raceform_with_{model_type}_{num_topics}_topic_features.csv"
            elif model_type == "embeddings":
                path = f"data/preprocessed_raceform_with_embeddings.csv"
            
            ds = pd.read_csv(path, low_memory=False)

            # Masks
            train_mask = ds["train_val_test_split"] == "train"
            val_mask = ds["train_val_test_split"] == "val"
            test_mask = ds["train_val_test_split"] == "test"
            y_train = ds.loc[train_mask, "is_winner"].values
            groups = ds.loc[train_mask, "race_id"].values

            print("Fitting baseline CL model...")
            X_train = ds.loc[train_mask, ["logit_sp_win_prob"]]
            baseline_model = ConditionalLogit(y_train, X_train, groups=groups)
            baseline_result = baseline_model.fit(disp=False)

            coefs_df = pd.DataFrame(
                columns=["model", "beta_hat", "beta_z_stat", "beta_se", "beta_ci_lower", "beta_ci_upper", "beta_p_val",
                        "gamma_hat", "gamma_z_stat", "gamma_se", "gamma_ci_lower", "gamma_ci_upper", "gamma_p_val"])
            coefs_df = pd.concat([coefs_df, pd.DataFrame({
                "model": ["baseline"],
                "beta_hat": [baseline_result.params["logit_sp_win_prob"]],
                "beta_z_stat": [baseline_result.tvalues["logit_sp_win_prob"]],
                "beta_se": [baseline_result.bse["logit_sp_win_prob"]],
                "beta_ci_lower": [baseline_result.conf_int().loc["logit_sp_win_prob", 0]],
                "beta_ci_upper": [baseline_result.conf_int().loc["logit_sp_win_prob", 1]],
                "beta_p_val": [baseline_result.pvalues["logit_sp_win_prob"]],
                "gamma_hat": [np.nan],
                "gamma_z_stat": [np.nan],
                "gamma_se": [np.nan],
                "gamma_ci_lower": [np.nan],
                "gamma_ci_upper": [np.nan],
                "gamma_p_val": [np.nan]
            })], ignore_index=True
            )

            coefs_df.to_csv(f"evaluation_metrics/cl_model_coefficients_baseline_{model_type}_{num_topics}_topics.csv", index=False)
            
            # Load significant topic features
            if model_type == "lda":
                chisq_results_path = f"evaluation_metrics/chisq_results_{model_type}_{num_topics}.csv"
                chisq_results = pd.read_csv(chisq_results_path)
                significant_features = chisq_results[chisq_results['pval_lag1_train'] < 0.05]['topic'].tolist()
                feature_cols = [f"{feature}_lag_1" for feature in significant_features if f"{feature}_lag_1" in ds.columns]
            elif model_type == "embeddings":
                feature_cols = [f"topic_{i}_lag_1" for i in range(128)]

            # Load language models
            print("Loading language predictive models...")
            if model_type == "lda":
                nb_model = load(f"language_predictive_models/naive_bayes_{model_type}_{num_topics}.joblib")
                calibrated_nb_model = load(f"language_predictive_models/calibrated_naive_bayes_{model_type}_{num_topics}.joblib")
                xgb_model = load(f"language_predictive_models/xgboost_{model_type}_{num_topics}.joblib")
                calibrated_xgb_model = load(f"language_predictive_models/calibrated_xgboost_{model_type}_{num_topics}.joblib")
            elif model_type == "embeddings":
                nb_model = load(f"language_predictive_models/naive_bayes_{model_type}.joblib")
                calibrated_nb_model = load(f"language_predictive_models/calibrated_naive_bayes_{model_type}.joblib")
                xgb_model = load(f"language_predictive_models/xgboost_{model_type}.joblib")
                calibrated_xgb_model = load(f"language_predictive_models/calibrated_xgboost_{model_type}.joblib")

            print("Predicting with language models...")
            logscore_df = pd.DataFrame(columns=["model", "t_stat", "se", "ci_lower", "ci_upper", "p_val_one_sided", "mean_log_score_diff"])
            for model_name, model in [("nb", nb_model), ("calibrated_nb", calibrated_nb_model),
                                        ("xgb", xgb_model), ("calibrated_xgb", calibrated_xgb_model)]:
                
                add_prob_and_logit(model_name, model)

                X_train = ds.loc[train_mask, ["logit_sp_win_prob", f"logit_{model_name}_win_prob"]]
                X_val = ds.loc[val_mask, ["logit_sp_win_prob", f"logit_{model_name}_win_prob"]]
                y_val = ds.loc[val_mask, "is_winner"].values
                cl_model = ConditionalLogit(y_train, X_train, groups=groups)
                cl_result = cl_model.fit(disp=False)

                coefs_df = pd.concat([coefs_df, pd.DataFrame({
                    "model": [model_name],
                    "beta_hat": [cl_result.params["logit_sp_win_prob"]],
                    "beta_z_stat": [cl_result.tvalues["logit_sp_win_prob"]],
                    "beta_se": [cl_result.bse["logit_sp_win_prob"]],
                    "beta_ci_lower": [cl_result.conf_int().loc["logit_sp_win_prob", 0]],
                    "beta_ci_upper": [cl_result.conf_int().loc["logit_sp_win_prob", 1]],
                    "beta_p_val": [cl_result.pvalues["logit_sp_win_prob"]],
                    "gamma_hat": [cl_result.params[f"logit_{model_name}_win_prob"]],
                    "gamma_z_stat": [cl_result.tvalues[f"logit_{model_name}_win_prob"]],
                    "gamma_se": [cl_result.bse[f"logit_{model_name}_win_prob"]],
                    "gamma_ci_lower": [cl_result.conf_int().loc[f"logit_{model_name}_win_prob", 0]],
                    "gamma_ci_upper": [cl_result.conf_int().loc[f"logit_{model_name}_win_prob", 1]],
                    "gamma_p_val": [cl_result.pvalues[f"logit_{model_name}_win_prob"]]
                })], ignore_index=True
                )

                # Predicted probabilities
                baseline_model_y_pred = predict_conditional_logit(baseline_result, ds, ["logit_sp_win_prob"])
                new_model_y_pred = predict_conditional_logit(cl_result, ds, ["logit_sp_win_prob", f"logit_{model_name}_win_prob"])
                y_true = ds["is_winner"].values

                # Per-sample Brier score
                baseline_model_brier = (baseline_model_y_pred - y_true) ** 2
                new_model_brier = (new_model_y_pred - y_true) ** 2

                # Take only validation set losses
                baseline_model_brier_val = baseline_model_brier[val_mask]
                new_model_brier_val = new_model_brier[val_mask]

                # Paired t-test (two-sided)
                t_stat, p_value_two_sided = ttest_rel(baseline_model_brier_val, new_model_brier_val)

                # One-sided p-value for improvement (baseline worse than new model)
                mean_diff = np.mean(baseline_model_brier_val - new_model_brier_val)
                if mean_diff > 0:
                    p_value_one_sided = p_value_two_sided / 2
                else:
                    p_value_one_sided = 1 - p_value_two_sided / 2

                # Standard error
                se = np.std(baseline_model_brier_val - new_model_brier_val, ddof=1) / np.sqrt(len(new_model_brier_val))

                # 95% CI
                ci_lower = mean_diff - 1.96 * se
                ci_upper = mean_diff + 1.96 * se

                # Append to dataframe
                logscore_df = pd.concat([logscore_df, pd.DataFrame({
                    "model": [model_name],
                    "t_stat": [t_stat],
                    "se": [se],
                    "ci_lower": [ci_lower],
                    "ci_upper": [ci_upper],
                    "p_val_one_sided": [p_value_one_sided],
                    "mean_brier_diff": [mean_diff]
                })], ignore_index=True)


                # betting strategy (on test set)
                # for each race, we bet £1 on the horse with the highest predicted win probability if EV > bet_threshold
                bet_threshold = 0.2  # minimum expected value to place a bet
                ds["prob_winner"] = baseline_model_y_pred
                ds["expected_value"] = ds["prob_winner"] * (ds["sp"] - 1) - (1 - ds["prob_winner"])
                max_probs = ds.groupby("race_id")["prob_winner"].transform("max")
                ds["bet"] = ((ds["prob_winner"] == max_probs) & (ds["expected_value"] > bet_threshold)).astype(int)
                strategy = [{"model": f"baseline", "wagers": ds.loc[test_mask, "bet"]}]
                ds["prob_winner"] = new_model_y_pred
                ds["expected_value"] = ds["prob_winner"] * (ds["sp"] - 1) - (1 - ds["prob_winner"])
                max_probs = ds.groupby("race_id")["prob_winner"].transform("max")
                ds["bet"] = ((ds["prob_winner"] == max_probs) & (ds["expected_value"] > bet_threshold)).astype(int)
                strategy.append({"model": f"{model_name}", "wagers": ds.loc[test_mask, "bet"]})
                print(f"Plotting strategy for {model_name} with {num_topics} topics...")
                plot_strategy(strategy, ds.loc[test_mask].copy(), path=f"evaluation_metrics/cl_model_strategy_pnl_{model_type}_{model_name}_{num_topics}_topics.png")


            logscore_df.to_csv(f"evaluation_metrics/cl_model_logscore_t_tests_{model_type}_{num_topics}_topics.csv", index=False)
            coefs_df.to_csv(f"evaluation_metrics/cl_model_coefficients_{model_type}_{num_topics}_topics.csv", index=False)