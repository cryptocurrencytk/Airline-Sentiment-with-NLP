
# Airline-Sentiment-with-NLP Readme

This document summarizes a sentiment analysis project conducted on American Airlines customer reviews spanning from December 2013 to April 2024. The project applied multiple natural language processing techniques to classify customer reviews as positive or negative recommendations.

## Project Overview
- **Data Source**: www.airlinequality.com
- **Time Period**: December 2013 - April 2024
- **Goal**: Analyze customer sentiments to help American Airlines make data-driven decisions for enhancing customer experience
- **Project Files**:
 - `aa_transformer_(bert).py` - Implementation of the BERT transformer model
 - `v5_downsample.py` - Code for the downsampling approach
 - `v5_Original.py` - Original implementation without rebalancing
 - `v5_oversample.py` - Code for the oversampling approach

## Data Collection and Preprocessing
- **Web Scraping with Selenium**:
 - Automated extraction of review data from airlinequality.com
 - Retrieved key fields including comments and recommendations
 - Saved data to Excel format for further processing
- **Data Issues and Adjustments**:
 - **Data Imbalance**: The dataset showed significant imbalance between positive and negative recommendations
 - **Rebalancing Approaches**:
   - Over-sampling: Increased the minority class to match the majority class (implemented in `v5_oversample.py`)
   - Under-sampling: Reduced the majority class to match the minority class (implemented in `v5_downsample.py`)
 - **Text Preprocessing**:
   - Removed HTML tags
   - Transformed abbreviated negations
   - Tokenization (removed non-letters and non-numbers)
   - Removed stop words
   - Applied lemmatization
   - Joined words back together
 - **Evaluation Adjustment**: Due to imbalance, the team used AUC and F1-score instead of relying solely on accuracy

## Methodology
The team employed three different approaches to sentiment analysis:

### 1. Traditional Method: TF-IDF
- Simple, interpretable approach suitable for smaller datasets
- Used 80-20 train-test split with unigram features
- Applied Logistic Regression for classification
- Performance: 97.86% AUC, 76.55% F1-score

### 2. Word Embedding Method
- Used pre-trained word embedding models (Word2Vec, GloVe, fastText)
- Applied embedding averaging to represent reviews
- Addressed data imbalance through over-sampling and under-sampling
- Tested multiple classification models:
 - Logistic Regression
 - LASSO and RIDGE Regression
 - Random Forest
 - Boosted Trees
- Performance: Random Forest and Boosted Trees with over-sampling achieved 99.97% AUC

### 3. Transformer Method: BERT
- Utilized pre-trained BERT model (implemented in `aa_transformer_(bert).py`)
- Applied fine-tuning for the classification task
- Performance: 98.81% AUC, 84.01% F1-score, 95.71% Accuracy

## Key Findings
- The dataset showed imbalanced distribution of recommendations
- AUC and F1-score were used as primary evaluation metrics due to class imbalance
- Embedding-based methods with ensemble models (Random Forest, Boosted Trees) performed best
- BERT achieved strong results for text classification without extensive feature engineering
- Rebalancing the dataset through over-sampling improved model performance significantly

## Potential Improvements
- Implement advanced embedding techniques (ELMo, Seq2Seq)
- Apply hyperparameter tuning (Grid Search) for classification models
- Explore additional models (KNN, Naïve Bayes, Neural Networks)
- For transformer models: data augmentation, hyperparameter tuning, ensemble learning
- Additional data rebalancing techniques like SMOTE (Synthetic Minority Over-sampling Technique)

The project demonstrates the progression of NLP techniques from simple TF-IDF to advanced transformer models, with each approach showing strengths in different areas of sentiment analysis.
