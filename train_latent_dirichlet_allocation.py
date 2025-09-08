"""
This script trains Latent Dirichlet Allocation (LDA) models on the comments columns.
It also produces evaluation metrics.
"""

import gensim
from gensim import corpora
from gensim.models import TfidfModel
from gensim.models.phrases import Phrases, Phraser
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

def tokenize_with_separator(docs, separator=" - "):
    tokenized_docs = []
    for doc in docs:
        clauses = doc.split(separator)
        tokenized_clauses = [clause.strip().split() for clause in clauses]
        tokenized_docs.append(tokenized_clauses)
    return tokenized_docs

def flatten_clauses(tokenized_docs_clauses, separator_token=" - "):
    processed_docs = []
    for doc_clauses in tokenized_docs_clauses:
        tokens = []
        for i, clause in enumerate(doc_clauses):
            tokens.extend(clause)
            if i < len(doc_clauses) - 1:
                tokens.append(separator_token)
        processed_docs.append(tokens)
    return processed_docs

def apply_bigrams_clausewise(tokenized_docs_clauses, bigram_phraser, separator_token="-"):
    processed_docs = []
    for doc_clauses in tokenized_docs_clauses:
        bigrammed_clauses = [bigram_phraser[clause] for clause in doc_clauses]
        tokens = []
        for i, clause in enumerate(bigrammed_clauses):
            tokens.extend(clause)
            if i < len(bigrammed_clauses) - 1:
                tokens.append(separator_token)
        processed_docs.append(tokens)
    return processed_docs


# Function to train LDA model
def train_lda_model(num_topics, corpus, dictionary, model_path):
    print(f"{datetime.now()} ---Training LDA Model with {num_topics} topics on {model_path}---")
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha="auto",
        eta="auto",
        per_word_topics=True
    )

    print(f"{datetime.now()} ---Saving Model---")
    lda_model.save(model_path)
    return lda_model


def evaluate_model(model, 
                   corpus_train,
                   corpus_val,
                   dictionary_train,
                   texts_train,
                   texts_val,
                   topn=15):
    print(f"{datetime.now()} ---Evaluating Model---")


    # Coherence
    coherence_model_train = gensim.models.CoherenceModel(
        model=model,
        texts=texts_train,
        dictionary=dictionary_train,
        coherence='c_v',
        topn=topn
    )
    coherence_model_val = gensim.models.CoherenceModel(
        model=model,
        texts=texts_val,
        dictionary=dictionary_train,
        coherence='c_v',
        topn=topn
    )
    coherence_score_train = coherence_model_train.get_coherence()
    coherence_score_val = coherence_model_val.get_coherence()


    # Perplexity
    perplexity_train = 2 ** (-model.log_perplexity(corpus_train))
    perplexity_val = 2 ** (-model.log_perplexity(corpus_val))


    # Topic Unique Metric (TU) (Nan et al., 2019)
    topic_words = [
        set([word for word, _ in model.show_topic(tid, topn=topn)])
        for tid in range(model.num_topics)
    ]
    
    T = len(topic_words)
    tu_scores = []

    for i in range(T):
        overlaps = [
            len(topic_words[i] & topic_words[j]) / topn
            for j in range(T) if j != i
        ]
        avg_overlap = sum(overlaps) / (T - 1)
        tu = 1 - avg_overlap
        tu_scores.append(tu)

    tu_score = sum(tu_scores) / T


    # Topic Similarity (average cosine similarity between topic vectors)
    # Get the topic-word distribution matrix
    topic_word_probs = []
    for topic_id in range(model.num_topics):
        # Get full distribution over dictionary words (length = vocab size)
        dist = model.get_topic_terms(topicid=topic_id, topn=len(dictionary_train))
        # Create vector of probabilities (initialized to zero)
        vec = np.zeros(len(dictionary_train))
        for word_id, prob in dist:
            vec[word_id] = prob
        topic_word_probs.append(vec)

    topic_word_probs = np.array(topic_word_probs)
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(topic_word_probs)
    # Exclude diagonal (self similarity)
    n = model.num_topics
    sum_sim = np.sum(similarity_matrix) - n  # subtract diagonal ones
    count_pairs = n * (n - 1)
    avg_topic_similarity = sum_sim / count_pairs if count_pairs > 0 else 0

    return {
        "coherence_train": coherence_score_train,
        "coherence_val": coherence_score_val,
        "perplexity_train": perplexity_train,
        "perplexity_val": perplexity_val,
        "tu_score": tu_score,
        "avg_topic_similarity": avg_topic_similarity
    }


if __name__ == "__main__":
    if not os.path.exists("lda_models"):
        os.makedirs("lda_models")

    ds = pd.read_csv("data/preprocessed_raceform.csv", low_memory=False)

    train_docs_raw = ds.loc[ds["train_val_test_split"] == "train", "preprocessed_comment"].dropna().tolist()
    comment_texts_train_clauses = tokenize_with_separator(train_docs_raw)
    all_train_clauses = [clause for doc in comment_texts_train_clauses for clause in doc]
    bigram_model = Phrases(all_train_clauses, min_count=100, threshold=10)
    bigram_phraser = Phraser(bigram_model)
    comment_texts_train = apply_bigrams_clausewise(comment_texts_train_clauses, bigram_phraser)
    comment_dictionary_train = corpora.Dictionary(comment_texts_train)
    comment_dictionary_train.save("lda_models/comment_dictionary_train.dict")
    comment_corpus_train = [comment_dictionary_train.doc2bow(text) for text in comment_texts_train]

    comment_texts_val = [
        doc.split() for doc in ds.loc[ds["train_val_test_split"] == "val", "preprocessed_comment"].dropna()
    ]
    comment_corpus_val = [comment_dictionary_train.doc2bow(text) for text in comment_texts_val]

    # Apply TF-IDF filtering to the training corpus
    tfidf_model = TfidfModel(comment_corpus_train)
    corpus_tfidf = tfidf_model[comment_corpus_train]
    threshold = 0.05
    comment_corpus_train = [
        [(id, count) for (id, count), (_, score) in zip(bow, tfidf) if score >= threshold]
        for bow, tfidf in zip(comment_corpus_train, corpus_tfidf)
    ]

    comment_results = []

    for num_topics in [10, 20, 30, 40, 50]:
        comment_model = train_lda_model(
            num_topics, 
            comment_corpus_train, 
            comment_dictionary_train, 
            f"lda_models/comment_lda_{num_topics}_topics.model"
        )

        # Evaluate models
        comment_metrics = evaluate_model(
            model=comment_model, 
            corpus_train=comment_corpus_train,
            corpus_val=comment_corpus_val,
            dictionary_train=comment_dictionary_train,
            texts_train=comment_texts_train,
            texts_val=comment_texts_val,
            topn=15
        )

        print(f"{datetime.now()} --- Evaluation Metrics for {num_topics} Topics ---")
        
        print("\nComment Model:")
        for metric, value in comment_metrics.items():
            print(f"{metric.capitalize()}: {value:.3f}")

        print(f"{datetime.now()} --- Training and Evaluation for {num_topics} Topics Complete ---")
        comment_results.append({"num_topics": num_topics, **comment_metrics})

    # Save evaluation results into two separate CSV files
    comment_df = pd.DataFrame(comment_results)
    if not os.path.exists("evaluation_metrics"):
        os.makedirs("evaluation_metrics")

    comment_df.to_csv("evaluation_metrics/lda_comment_evaluation_metrics.csv", index=False)
    print(f"{datetime.now()} --- Evaluation Results Saved ---")