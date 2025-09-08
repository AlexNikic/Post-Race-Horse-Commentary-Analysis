"""
This script trains models using language features to predict race winners.
It selects significant features based on Chi-Square tests, trains XGBoost, Naive Bayes, and optionally SVMs,
performs hyperparameter tuning using a predefined validation set, calibrates models, evaluates them,
and performs t-tests against uniform predictions with Bonferroni correction.
"""

import numpy as np
import pandas as pd
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from scipy.stats import ttest_rel
import os
import pickle

# ------------------ Model training functions ------------------

def train_svm(X_train, y_train, params=None):
    model = svm.SVC(probability=True, kernel='rbf', **(params or {}))
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, params=None):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **(params or {}))
    model.fit(X_train, y_train)
    return model

def train_naive_bayes(X_train, y_train, params=None):
    model = GaussianNB(**(params or {}))
    model.fit(X_train, y_train)
    return model

def calibrate_model(model, X_val, y_val):
    cal_model = CalibratedClassifierCV(estimator=model, method="isotonic", cv="prefit")
    cal_model.fit(X_val, y_val)
    return cal_model

# ------------------ Evaluation functions ------------------

def t_test_vs_uniform(model, X_val, y_val):
    y_pred = model.predict_proba(X_val)   # shape (n_samples, n_classes)
    n_samples, n_classes = y_pred.shape

    # One-hot encode y_val
    y_true_onehot = np.zeros_like(y_pred)
    y_true_onehot[np.arange(n_samples), y_val] = 1

    # Per-sample Brier score for the model
    model_brier = np.sum((y_pred - y_true_onehot) ** 2, axis=1)

    # Per-sample Brier score for uniform predictions
    uniform_probs = np.full_like(y_pred, 1.0 / n_classes)
    uniform_brier = np.sum((uniform_probs - y_true_onehot) ** 2, axis=1)

    # Paired t-test (two-sided)
    t_stat, p_value_two_sided = ttest_rel(uniform_brier, model_brier)

    # One-sided p-value for improvement (model better than uniform)
    mean_diff = np.mean(uniform_brier - model_brier)
    if mean_diff > 0:
        p_value_one_sided = p_value_two_sided / 2
    else:
        p_value_one_sided = 1 - p_value_two_sided / 2

    return t_stat, p_value_one_sided, mean_diff

# ------------------ Hyperparameter tuning ------------------

def tune_model(model_class, param_grid, X_train, y_train, X_val, y_val):
    # Combine train + val
    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = pd.concat([y_train, y_val], axis=0)
    # Define PredefinedSplit: -1 = train, 0 = val
    test_fold = np.concatenate([-1*np.ones(len(X_train)), np.zeros(len(X_val))])
    ps = PredefinedSplit(test_fold)

    grid = GridSearchCV(
        estimator=model_class,
        param_grid=param_grid,
        cv=ps,
        scoring='neg_log_loss',
        n_jobs=1
    )
    grid.fit(X_combined, y_combined)
    return grid.best_estimator_, grid.best_params_

# ------------------ Main script ------------------

if __name__ == "__main__":
    model_type = "lda"
    for num_topics in [10, 20, 30, 40, 50]:
        # Load dataset
        print("Loading dataset...")
        path = f"data/preprocessed_raceform_with_{model_type}_{num_topics}_topic_features.csv"
        ds = pd.read_csv(path, low_memory=False)

        # Load significant features
        chisq_results_path = f"evaluation_metrics/chisq_results_{model_type}_{num_topics}.csv"
        chisq_results = pd.read_csv(chisq_results_path)
        significant_features = chisq_results[chisq_results['pval_lag1_train'] < 0.05]['topic'].tolist()

        # Feature preparation
        feature_cols = [f"{feature}_lag_1" for feature in significant_features if f"{feature}_lag_1" in ds.columns]
        train_mask = ds["train_val_test_split"] == "train"
        val_mask = ds["train_val_test_split"] == "val"
        X_train = ds.loc[train_mask, feature_cols]
        y_train = ds.loc[train_mask, "is_winner"]
        X_val = ds.loc[val_mask, feature_cols]
        y_val = ds.loc[val_mask, "is_winner"]

        # ------------------ Hyperparameter tuning ------------------

        # XGBoost
        print(f"Tuning XGBoost for {model_type} with {num_topics} topics...")
        xgb_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1]
        }
        xgboost_model, xgb_best_params = tune_model(
            XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            xgb_param_grid, X_train, y_train, X_val, y_val
        )
        print("Best XGBoost params:", xgb_best_params)

        # Naive Bayes
        print(f"Tuning Naive Bayes for {model_type} with {num_topics} topics...")
        naive_bayes_model, nb_best_params = tune_model(
            BernoulliNB(), {}, X_train, y_train, X_val, y_val
        )
        print("Best Naive Bayes params:", nb_best_params)


        # ------------------ Calibrate models ------------------
        calibrated_xgboost = calibrate_model(xgboost_model, X_val, y_val)
        calibrated_naive_bayes = calibrate_model(naive_bayes_model, X_val, y_val)

        # ------------------ T-tests against uniform predictions ------------------
        t_test_results = []
        n_tests = 4  # XGBoost + Cal XGBoost + NB + Cal NB

        for name, model in [("XGBoost", xgboost_model), 
                            ("Calibrated XGBoost", calibrated_xgboost),
                            ("Naive Bayes", naive_bayes_model),
                            ("Calibrated Naive Bayes", calibrated_naive_bayes)]:
            t_stat, p_value, mean_diff = t_test_vs_uniform(model, X_val, y_val)
            p_value_bonf = min(p_value * n_tests, 1.0)  # Bonferroni correction

            t_test_results.append({
                "model": name,
                "t_statistic": t_stat,
                "p_value": p_value,
                "p_value_bonferroni": p_value_bonf,
                "mean_brier_diff": mean_diff
            })

            print(f"{name}: t-statistic = {t_stat:.4f}, original p-value = {p_value:.4f}, "
                  f"Bonferroni p-value = {p_value_bonf:.4f}, mean improvement = {mean_diff:.4f}")

        # Save t-test results
        t_test_results_df = pd.DataFrame(t_test_results)
        t_test_results_df.to_csv(f"evaluation_metrics/t_test_vs_uniform_{model_type}_{num_topics}.csv", index=False)
        print(f"T-test results with Bonferroni correction saved for {model_type} with {num_topics} topics.")

        # Save models
        save_folder = "language_predictive_models"
        os.makedirs(save_folder, exist_ok=True)
        model_save_paths = {
            "XGBoost": os.path.join(save_folder, f"xgboost_{model_type}_{num_topics}.joblib"),
            "Calibrated XGBoost": os.path.join(save_folder, f"calibrated_xgboost_{model_type}_{num_topics}.joblib"),
            "Naive Bayes": os.path.join(save_folder, f"naive_bayes_{model_type}_{num_topics}.joblib"),
            "Calibrated Naive Bayes": os.path.join(save_folder, f"calibrated_naive_bayes_{model_type}_{num_topics}.joblib")
        }
        with open(model_save_paths["XGBoost"], "wb") as f:
            pickle.dump(xgboost_model, f)

        with open(model_save_paths["Calibrated XGBoost"], "wb") as f:
            pickle.dump(calibrated_xgboost, f)

        with open(model_save_paths["Naive Bayes"], "wb") as f:
            pickle.dump(naive_bayes_model, f)

        with open(model_save_paths["Calibrated Naive Bayes"], "wb") as f:
            pickle.dump(calibrated_naive_bayes, f)

        print(f"Saved best models for {model_type} with {num_topics} topics to '{save_folder}'")
        

    # --- Train models for embeddings ---
    model_type = "embeddings"

    # Load dataset
    print("Loading dataset...")
    path = f"data/preprocessed_raceform_with_embeddings.csv"
    ds = pd.read_csv(path, low_memory=False)

    # Feature preparation
    feature_cols = [f"topic_{i}_lag_1" for i in range(128)]
    train_mask = ds["train_val_test_split"] == "train"
    val_mask = ds["train_val_test_split"] == "val"
    X_train = ds.loc[train_mask, feature_cols]
    y_train = ds.loc[train_mask, "is_winner"]
    X_val = ds.loc[val_mask, feature_cols]
    y_val = ds.loc[val_mask, "is_winner"]

    # ------------------ Hyperparameter tuning ------------------

    # XGBoost
    print(f"Tuning XGBoost for {model_type}")
    xgb_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1]
    }
    xgboost_model, xgb_best_params = tune_model(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        xgb_param_grid, X_train, y_train, X_val, y_val
    )
    print("Best XGBoost params:", xgb_best_params)

    # Naive Bayes
    print(f"Tuning Naive Bayes for {model_type}...")
    # Naive Bayes
    print(f"Tuning Naive Bayes for {model_type} with {num_topics} topics...")
    naive_bayes_model, nb_best_params = tune_model(
        GaussianNB(), {}, X_train, y_train, X_val, y_val
    )
    print("Best Naive Bayes params:", nb_best_params)

    print("Best Naive Bayes params:", nb_best_params)

    # ------------------ Calibrate models ------------------
    calibrated_xgboost = calibrate_model(xgboost_model, X_val, y_val)
    calibrated_naive_bayes = calibrate_model(naive_bayes_model, X_val, y_val)

    # ------------------ T-tests against uniform predictions ------------------
    t_test_results = []
    n_tests = 4  # XGBoost + Cal XGBoost + NB + Cal NB

    for name, model in [("XGBoost", xgboost_model), 
                        ("Calibrated XGBoost", calibrated_xgboost),
                        ("Naive Bayes", naive_bayes_model),
                        ("Calibrated Naive Bayes", calibrated_naive_bayes)]:
        t_stat, p_value, mean_diff = t_test_vs_uniform(model, X_val, y_val)
        p_value_bonf = min(p_value * n_tests, 1.0)  # Bonferroni correction

        t_test_results.append({
            "model": name,
            "t_statistic": t_stat,
            "p_value": p_value,
            "p_value_bonferroni": p_value_bonf,
            "mean_brier_diff": mean_diff
        })

        print(f"{name}: t-statistic = {t_stat:.4f}, original p-value = {p_value:.4f}, "
                f"Bonferroni p-value = {p_value_bonf:.4f}, mean improvement = {mean_diff:.4f}")

    # Save t-test results
    t_test_results_df = pd.DataFrame(t_test_results)
    t_test_results_df.to_csv(f"evaluation_metrics/t_test_vs_uniform_{model_type}.csv", index=False)
    print(f"T-test results with Bonferroni correction saved for {model_type}.")

    # Save models
    save_folder = "language_predictive_models"
    os.makedirs(save_folder, exist_ok=True)
    model_save_paths = {
        "XGBoost": os.path.join(save_folder, f"xgboost_{model_type}.joblib"),
        "Calibrated XGBoost": os.path.join(save_folder, f"calibrated_xgboost_{model_type}.joblib"),
        "Naive Bayes": os.path.join(save_folder, f"naive_bayes_{model_type}.joblib"),
        "Calibrated Naive Bayes": os.path.join(save_folder, f"calibrated_naive_bayes_{model_type}.joblib")
    }
    with open(model_save_paths["XGBoost"], "wb") as f:
        pickle.dump(xgboost_model, f)

    with open(model_save_paths["Calibrated XGBoost"], "wb") as f:
        pickle.dump(calibrated_xgboost, f)

    with open(model_save_paths["Naive Bayes"], "wb") as f:
        pickle.dump(naive_bayes_model, f)

    with open(model_save_paths["Calibrated Naive Bayes"], "wb") as f:
        pickle.dump(calibrated_naive_bayes, f)

    print(f"Saved best models for {model_type} topics to '{save_folder}'")