# Horse Racing Post-Commentary Analysis using Topic Modelling Approaches

The aim of this repository is to use using Topic Modelling to extract meaningful features from the post-race commentary. We will then use these to engineer features which describe each horse's past performance. Then we will use these features in prediction models, to see how good they are.

## Requirements

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
├── betting_graphs/                        # Folder to store betting graphs
├── data/                                  # Folder to store our datasets
├── evaluation_metrics/                    # Folder to store evaluation metrics of the models
├── figures/                               # Folder to store any graphics for figures
├── language_predictive_models/            # Folder to store predictive models
├── lda_models/                            # Folder to store our LDA models
├── notebooks/                             # Folder for .ipynb files
├── wordclouds/                            # Folder to store wordclouds

├── preprocessing.py                       # Script to preprocess the data
├── train_latent_dirichlet_allocation.py   # Script to train the LDA models
├── feature_creator.py                     # File that to store useful functions for feature creation
├── chisq_tests.py                         # Script to perform feature selection for LDA models
├── get_embeddings.py                      # Script to generate transformer embeddings of comments
├── train_language_predictive_models.py    # Script to train predictive models using the language features
├── train_cl_models.py                     # Script to train the Conditional Logit models and save evaluation metrics

├── environment.yml                        # Conda environment
├── README.md

├── The_Dissertation.pdf                   # Final report of the project
└── The_Presentation.pdf                   # Presentation giving a summary of the project
```

## data/

This folder contains the dataset used in this project. It is also where the enhanced datasets are saved after running the scripts. Since the datasets are rather large (~600MB each) I cannot upload them to GitHub, but if you would like me to upload them somewhere to save you from the running the scripts please feel free to let me know and I can try my best to get something set up. If you would like to run the scripts yourself, please run them in the order as listed below.

## preprocessing.py

The first thing we need to do is ensure the `raceform.csv` is in the data folder. This is the dataset from kaggle, which can be found at `https://www.kaggle.com/datasets/deltaromeo/horse-racing-results-ukireland-2015-2025`.
Then we can run `preprocessing.py` to clean the data and add the needed columns for later analysis. It is good to run this and save the cleaned data before later analysis because pre-processing can take a while. This will save a the result to the path `data/preprocessed_raceform.csv`.

## train_latent_dirichlet_allocation.py

This script is used to train Latent Dirichlet Allocation (LDA) models on the training split of the post-comment data. LDA is an unsupervised topic modelling technique which infers the underlying latent topics in a collection of documents by discovering patterns of word co-occurrence and representing each document as a mixture of these topics. It will allow us to reduce the dimensionality of our comment by grouping them into key themes which we can then use as features in downstream models. The models are saved in the `lda_models` folder.

Since you need to specify the number of topics to use as a hyperparameter, the script trains a number of models using different numbers of topics. You can specify in the code how many models you would like to train.
```python
for num_topics in [10, 20, 30, 40, 50]:
        comment_model = train_lda_model(
            num_topics, 
            comment_corpus_train, 
            comment_dictionary_train, 
            f"lda_models/comment_lda_{num_topics}_topics.model"
        )

        # ...
```

## get_embeddings.py

This script generates embeddings of the comments using a transformer model from HuggingFace. Currently it is using the BERT-tiny model:
```python
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
```
It is possible to use other models, however I will have to update the code elsewhere in the project as right now it is hard-coded to use embedding vectors of length 128 😜. This script will save a data frame to `data/preprocessed_raceform_with_embeddings.csv`.

## feature_creator.py

Now that we have trained the topic models, we need to use them to create useful features for our downstream models. This file contains a number of functions we can use to do this. Currently it will generate "lag_1" features, which are essentially the comments from each horse's previous race. I have set it up so that if you run the script it will generate these features for the LDA models with 10, 20, 30, 40, and 50 topics, and it will save the new data frames to `data/preprocessed_raceform_with_lda_10_topic_features.csv`, ... `data/preprocessed_raceform_with_lda_50_topic_features.csv`.

## chisq_tests.py

This script is used to perform feature selection on the features from the LDA models. It uses Chi-Squared Tests for Independence between the topic features and the binary variable "is_winner" - whether the horse wins the race or not. It saves its results in the `evaluation_metrics` folder.

## train_language_predictive_models.py

This script trains different classifiers to predict the outcome of the race. For each LDA model and embeddings model, it trains a Naive-Bayes classifier and an XGBoost classifier, and additionally performs Isotonic Regression on these for probability calibration. These models are saved in the `language_predictive_models` folder, and certain evaluation metrics are also saved in the `evaluation_metrics` folder.

## train_cl_models.py   

Finally this script trains the Conditional Logit (CL) models that ensemble the output probabilities from the classifiers with the starting prices of the horses. Unfortunately it does not save these models as the version of the `statsmodels` library does not yet support this function. Once the CL models have been fitted, it computes a lot of evaluation metrics and betting graphs, which are saved to the respective folders.