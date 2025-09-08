from datetime import datetime
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import word_tokenize, pos_tag
import pandas as pd
import numpy as np
import re


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
stop_words.add("op")
stop_words.add("tchd")
stop_words.add("final")
stop_words.add("furlong")
stop_words.remove("no")
stop_words.remove("up")
stop_words.remove("off")

def fractional_to_decimal(fractional_odds):
    """
    Converts fractional odds string to decimal odds.
    Handles standard fractions, 'Evs', 'Evens', 'EvsF', etc.
    Returns NaN for invalid or missing input.
    This function is specific to the kaggle dataset.
    """
    # Handle missing values
    if pd.isna(fractional_odds):
        return np.nan

    # Normalize to string
    odds_str = str(fractional_odds).strip().lower()

    # Handle evens variations
    if odds_str.startswith('evs') or odds_str.startswith('evens'):
        return 2.0

    # Regex to extract fractional odds like 5/2
    match = re.search(r'(\d+)\s*/\s*(\d+)', odds_str)
    if match:
        num = int(match.group(1))
        den = int(match.group(2))
        return round((num / den) + 1, 2)

    # If no match, return NaN
    return np.nan

def get_wordnet_pos(tag):
    """Map POS tag to WordNet format for lemmatization."""
    tag = tag[0].upper()
    return {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }.get(tag, wordnet.NOUN)

def preprocess_text(text):
    """
    Tokenizes text, removes stopwords and non-alphabetic tokens.
    Then lemmatizes and lowercases.
    """
    if not isinstance(text, str):
        return ""

    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    cleaned_tokens = []
    for word, tag in pos_tags:
        word_lower = word.lower()
        if word_lower.isalpha() and word_lower not in stop_words:
            lemma = lemmatizer.lemmatize(word_lower, get_wordnet_pos(tag))
            cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)

def preprocess_data(load_path, save_path):
    print(f"{datetime.now()} ---Loading Data---")
    ds = pd.read_csv(load_path, low_memory=False)

    # Create horse ids
    ds["horse_id"] = pd.factorize(ds["horse"])[0] + 1

    # Is it very important to sort the data chronologically to avoid data leakage
    ds = ds.sort_values(by=['date', 'race_id']).reset_index(drop=True)

    # Remove rows which do not have starting prices
    ds["sp"] = ds["sp"].apply(fractional_to_decimal)
    ds = ds.groupby("race_id").filter(
        lambda group: group["sp"].notna().all()
    )

    # Pre-process comment data for later use in the topic models
    print(f"{datetime.now()} ---Pre-Processing Comments---")
    ds["preprocessed_comment"] = ds["comment"].apply(preprocess_text)

     # Calculate market implied win probabilities using power margin removal
    ds["sp_win_prob"] = ds.groupby("race_id")["sp"].transform(lambda x: (1 / x) / (1 / x).sum()).clip(1e-12, 1 - 1e-12)
    ds["logit_sp_win_prob"] = np.log(ds["sp_win_prob"] / (1 - ds["sp_win_prob"]))

    # Create target variable for winners of each race
    ds['is_winner'] = (ds['pos'] == "1").astype(int)

    # Make sure date is in the correct format
    ds['date'] = pd.to_datetime(ds['date'], errors='coerce')
    
    # Train/Validation/Test split
    # Train: 2015-01-01 to 2018-12-31 (4 years)
    # Validation: 2019-01-01 to 2021-12-31 (2 years)
    # Test: 2022-01-01 to 2025-06-30 (3.5 years)
    def assign_split(row):
        if row['date'] < pd.Timestamp('2019-01-01'):
            return 'train'
        elif pd.Timestamp('2019-01-01') <= row['date'] < pd.Timestamp('2022-01-01'):
            return 'val'
        elif pd.Timestamp('2022-01-01') <= row['date']:
            return 'test'
    ds['train_val_test_split'] = ds.apply(assign_split, axis=1)

    # Save the cleaned dataset
    print(f"{datetime.now()} ---Saving Cleaned Data---")
    ds.to_csv(save_path, index=False)

    # Check how many races are in each split
    print(f"Train: {ds[ds["train_val_test_split"] == "train"].groupby('race_id').ngroups} races")
    print(f"Val: {ds[ds["train_val_test_split"] == "val"].groupby('race_id').ngroups} races")
    print(f"Test: {ds[ds["train_val_test_split"] == "test"].groupby('race_id').ngroups} races")

if __name__ == "__main__":
    preprocess_data(
        load_path="data/raceform.csv",
        save_path="data/preprocessed_raceform.csv"
    )