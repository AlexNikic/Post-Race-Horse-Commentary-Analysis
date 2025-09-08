"""
TODO: write this
"""

import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from datetime import datetime

# TODO: we need to consider how to deal with phrases such as "2f out", right now they are being removed by the model
# but instead we probably need to figure out a way to add these as features too
# TODO: justify/find the best threshold; it seems like lowering this gives better results

def topics_to_vector(topic_dist, num_topics):
    vec = np.zeros(num_topics)
    for topic_id, prob in topic_dist:
        vec[topic_id] = prob
    return vec

def create_comment_features(ds, comment_model_path, dictionary_path, topic_probability_threshold):
    comment_model = LdaModel.load(comment_model_path)
    comment_dictionary = corpora.Dictionary.load(dictionary_path)
    num_topics = comment_model.num_topics

    print(f"{datetime.now()} --- Assigning Topics for Comments ---")

    valid_mask = ds["preprocessed_comment"].notna()
    comment_texts = [doc.split() for doc in ds.loc[valid_mask, "preprocessed_comment"]]
    comment_corpus = [comment_dictionary.doc2bow(text) for text in comment_texts]

    # Get topic distributions
    comment_topics = [
        comment_model.get_document_topics(bow, minimum_probability=topic_probability_threshold)
        for bow in comment_corpus
    ]
    topic_vectors = [
        [1 if prob > 0 else 0 for prob in topics_to_vector(dist, num_topics)]
        for dist in comment_topics
    ]

    # Initialise all topic columns with 0
    for i in range(num_topics):
        ds[f"topic_{i}"] = 0

    # Assign topic vectors back to valid rows
    topic_vector_df = pd.DataFrame(
        topic_vectors,
        columns=[f"topic_{i}" for i in range(num_topics)],
        index=ds.loc[valid_mask].index
    )

    for col in topic_vector_df.columns:
        ds.loc[valid_mask, col] = topic_vector_df[col]

    return ds


def create_rollavg_features(ds, features_regex, window_size):
    """
    This function creates rolling average features for the post-race comments,
    shifted by one row as to not include the current race.
    """
    ds = ds.sort_values(by=['date', 'race_id']).reset_index(drop=True)

    topic_columns = ds.columns[ds.columns.str.contains(features_regex)].tolist()
    
    for col in topic_columns:
        rollavg_col_name = f"{col}_rollavg_{window_size}"
        ds[rollavg_col_name] = (
            ds.groupby("horse_id")[col]
            .rolling(window=window_size, min_periods=1, closed="left")
            .mean()
            .fillna(0)
            .reset_index(level=0, drop=True)
        )

    return ds.copy() # return .copy() to defragment dataframe

def create_lag_features(ds, features_regex, max_lag):
    """
    This function creates lag features for columns matching the regex,
    shifted by 1 to max_lag rows, grouped by horse_id to avoid data leakage.
    """
    ds = ds.sort_values(by=['date', 'race_id']).reset_index(drop=True)

    lag_columns = ds.columns[ds.columns.str.contains(features_regex)].tolist()

    for col in lag_columns:
        for lag in range(1, max_lag + 1):
            lag_col_name = f"{col}_lag_{lag}"
            ds[lag_col_name] = (
                ds.groupby("horse_id")[col]
                .shift(lag)
                .fillna(0)
            )

    return ds.copy() # return copy to avoid fragmentation

def create_bet_type_interaction_features(ds, features_regex):
    ds["sp_bet_type"] = ds["sp_win_prob"].apply(
        lambda x: "longshot" if x <= 0.05 else "sureshot" if x >= 0.5 else "neither"
    )
    bettype_dummies = pd.get_dummies(ds["sp_bet_type"], drop_first=False)
    feature_columns = ["sp_win_prob"] + ds.columns[ds.columns.str.contains(features_regex)].tolist()

    for bet_col in bettype_dummies.columns:
        for feature_col in feature_columns:
            ds[f"{feature_col}_{bet_col}"] = ds[feature_col] * bettype_dummies[bet_col]

    return ds

def create_z_scores_by_race(ds, features_regex):
    columns = ds.columns[ds.columns.str.contains(features_regex)].tolist()

    for col in columns:
        z_col = col + "_z"
        ds[z_col] = ds.groupby("race_id")[col].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) != 0 else 0
        )

    return ds

if __name__ == "__main__":
    df = pd.read_csv("data/preprocessed_raceform.csv", low_memory=False)
    for model_type in ["lda"]:
        for num_topics in [10, 20, 30, 40, 50]:
            ds = df.copy()

            threshold = 1/num_topics
            ds = create_comment_features(ds, 
                                        f"{model_type}_models/comment_{model_type}_{num_topics}_topics.model", 
                                        f"{model_type}_models/comment_dictionary_train.dict", 
                                        topic_probability_threshold=threshold)
            #ds = create_bet_type_interaction_features(ds, r"^topic_\d+$")
            #ds = create_rollavg_features(ds, r"^topic_\d", window_size=5)
            ds = create_lag_features(ds, r"^topic_\d", max_lag=1)

            ds.to_csv(f"data/preprocessed_raceform_with_{model_type}_{num_topics}_topic_features.csv", index=False)

    # For embeddings
    ds = pd.read_csv("data/preprocessed_raceform_with_embeddings.csv", low_memory=False)
    ds = create_lag_features(ds, r"^topic_\d+", max_lag=1)
    ds.to_csv(f"data/preprocessed_raceform_with_embeddings.csv", index=False)
    print("Feature creation complete.")