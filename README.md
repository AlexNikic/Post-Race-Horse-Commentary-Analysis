# Horse Racing Post-Commentary Analysis using Topic Modelling Approaches

The aim of this repository is to use using Topic Modelling to extract meaningful features from the post-race commentary. We will then use these to engineer features which describe each horse's past performance. Then we will use these features in prediction models, to see how good they are.

## Requirements

[TODO: update this conda environment]
This project requires a specific Conda environment to ensure compatibility between the libraries. Please create the environment using the provided `environment.yml` file:

```bash
conda myenv create -f environment.yml
conda activate myenv
```

If you have any problems running the code with this environment please let me know and I can try and fix it 😊.

## 🧾 NLTK Setup

This project uses NLTK for text preprocessing. 

If you wish to run the preprocessing or topic model training scripts, you may need to download some NLTK data before running any code. If you are running notebooks using the already processed data, then you should be fine without it.

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
```

## 📁 Repository Structure

```
/
├── data/                                  # Folder to store our datasets
├── lda_models/                            # Folder to store our LDA models
├── preprocessing.py                       # Script to preprocess the data
├── train_latent_dirichlet_allocation.py   # Script to train the LDA models
├── feature_creator.py                     # File that to store useful functions for feature creation
├── results.ipynb                          # Notebook for results
├── environment.yml                        # Conda environment
└── README.md
```

## preprocessing.py

The first thing we need to do is ensure the `raceform.csv` is in the data folder. This is the dataset from kaggle, which can be found at `https://www.kaggle.com/datasets/deltaromeo/horse-racing-results-ukireland-2015-2025`.
Then we can run `preprocessing.py` to clean the data and add the needed columns for later analysis. It is good to run this and save the cleaned data before later analysis because pre-processing can take a while.

## train_latent_dirichlet_allocation.py

This script is used to train Latent Dirichlet Allocation (LDA) models on the training split of the post-comment data. LDA is an unsupervised topic modelling technique which infers the underlying latent topics in a collection of documents by discovering patterns of word co-occurrence and representing each document as a mixture of these topics. It will allow us to reduce the dimensionality of our comment by grouping them into key themes which we can then use as features in downstream models.

Since you need to specify the number of topics to use as a hyperparameter, the script trains a number of models using different numbers of topics. You can specify in the code how many models you would like to train.
```python
for num_topics in [10, 20, 50, 100]:
        comment_model = train_lda_model(
            num_topics, 
            comment_corpus_train, 
            comment_dictionary_train, 
            f"topic_models/comment_lda_{num_topics}_topics.model"
        )

        # ...
```
We also have a function named `evaluate_metrics` which we can use to compute commonly-used metrics for topic models, including coherence, perplexity, topic diversity, and topic similarity.
```python
comment_metrics = evaluate_model(
    model=comment_model, 
    corpus_train=comment_corpus_train,
    corpus_val=comment_corpus_val,
    dictionary_train=comment_dictionary_train,
    texts_train=comment_texts_train,
    texts_val=comment_texts_val,
    topn=15
)

# ...
```

The script should save a file named in the `evaluation_metrics` folder, named `lda_comment_evaluation_metrics.csv`:

```csv
| num_topics | coherence_train | coherence_val | perplexity_train | perplexity_val | tu_score | avg_topic_similarity |
|------------|------------------|----------------|-------------------|----------------|----------|-----------------------|
| 10         | 0.4958           | 0.4507         | 24.5868           | 28.9411        | 0.8637   | 0.1025                |

```

## feature_creator.py

Now that we have trained the topic models, we need to use them to create useful features for our downstream models. This file contains a number of functions we can use to do this. I have set it up so that if you run the script it will generate these features for the LDA model with 10 topics, and it will save the new data frame to `preprocessed_raceform_with_topic_features.csv`.
[TODO: explain how this process works in more detail]

## results.ipynb

This notebook file shows the most promising results I have found so far.