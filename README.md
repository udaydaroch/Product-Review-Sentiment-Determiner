# Review Sentiment Determiner

## Overview

The **Review Sentiment Determiner** project aims to predict the sentiment or rating of product reviews using machine learning techniques. It utilizes natural language processing (NLP) methods to analyze textual reviews and predict their corresponding ratings. This can help businesses and consumers better understand and categorize customer feedback automatically.

## Data Source

The dataset used in this project is sourced from the UCSD Amazon product reviews dataset available at [UCSD Data Repository](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Software.jsonl.gz). The dataset contains reviews from various software products, including user IDs, review texts, and ratings.

## Project Workflow

1. **Data Download and Extraction**:
   - The raw data is downloaded from the provided URL using Python's `requests` library.
   - The data is then processed to extract relevant fields such as user IDs, review texts, and ratings using JSON parsing and text manipulation techniques.

2. **Data Preprocessing**:
   - Text preprocessing is performed to clean the review texts:
     - Special HTML characters like `<br />` are removed.
     - Text is converted to lowercase and non-alphabetic characters are removed.
     - Stop words (commonly occurring words like "the", "is", etc.) are filtered out using NLTK's stop words list.

3. **Feature Engineering**:
   - Text data is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization:
     - TF-IDF assigns weights to each word in the reviews based on how frequently they appear in a review relative to how frequently they appear across all reviews.
     - This converts text data into numerical vectors suitable for machine learning algorithms.

4. **Model Training and Evaluation**:
   - A logistic regression model is trained using the vectorized text data and corresponding ratings:
     - Logistic regression is chosen due to its simplicity and effectiveness in binary and multi-class classification tasks.
   - The model is evaluated using metrics such as accuracy and classification report:
     - Accuracy measures the overall correctness of the model's predictions.
     - The classification report provides precision, recall, and F1-score metrics for each rating class.

5. **Model Persistence**:
   - Trained model and vectorizer are saved using `joblib` for future use without needing to retrain.

6. **Example Usage**:
   - After training and saving the model, new reviews can be inputted to predict their ratings:
     - The saved model and vectorizer are loaded.
     - New reviews are transformed using the vectorizer and predicted ratings are obtained using the model.

## Tools Used

- **Python Libraries**: `pandas`, `dask`, `scikit-learn`, `nltk`
- **Machine Learning Models**: Logistic Regression
- **Text Vectorization**: TF-IDF Vectorizer
- **Data Processing**: JSON parsing, text cleaning, stop word removal
- **Model Persistence**: `joblib`

## Getting Started

To run this project locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd Review-Sentiment-Determiner
2. Download the dataset:
   the dataset will be downloaded automatically by running the python script
3. Run the script:
   python sentiment_analysis.py
