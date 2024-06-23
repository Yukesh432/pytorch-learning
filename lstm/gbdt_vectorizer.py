import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, log_loss
import mlflow
import mlflow.sklearn
import os
import nltk
from config import settings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
nltk.download('punkt')
nltk.download('stopwords')

"""
Ref:: https://github.com/piskvorky/gensim-data
Embedding models: 
- fasttext-wiki-news-subwords-300
- word2vec-google-news-300
- glove-twitter-100
- glove-twitter-200
- glove-twitter-25
- glove-twitter-50
"""

NUM_DATA = 1000
BATCH_SIZE = 16
# EMBEDDING_MODEL= 'glove-twitter-25'
# EMBEDDING_DIM= 300
# MAX_SEQ_LEN= 1000
MODEL= GradientBoostingClassifier
VECTORIZER= CountVectorizer

STOP_WORDS = set(stopwords.words('english'))

def load_dataset(csv_filepath):
    return pd.read_csv(csv_filepath, nrows=NUM_DATA)

def preprocess_text(text):
    text = re.sub(r'\d', '0', text)  # replace every digit with 0
    if 'www.' in text or 'http:' in text or 'https:' in text or '.com' in text:
        text = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", " ", text)  # remove links and urls
    text = re.sub(r'[^A-Za-z]', ' ', text)  # anything which is not a character replace with white space char
    
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in STOP_WORDS]  # remove stopwords
    text = ' '.join(text)
    
    return text

def preprocess_data(data):
    df = data.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time', 'Summary'], axis=1)
    df = df[df['Score'] != 3]

    df['Score'] = df['Score'].apply(lambda i: 'positive' if i > 4 else 'negative')
    df['Text'] = df['Text'].apply(lambda x: preprocess_text(x))

    df.columns = ['labels', 'text']
    idx_positive = df[df['labels'] == 'positive'].index
    nbr_to_drop = len(df) - len(idx_positive)
    drop_indices = np.random.choice(idx_positive, nbr_to_drop, replace=False)
    df = df.drop(drop_indices)

    text_as_list = df['text'].tolist()
    labels_as_list = df['labels'].tolist()

    vectorizer = VECTORIZER()
    encoded_text = vectorizer.fit_transform(text_as_list)
    
    # features = encoded_text.shape[1]

    encoded_text_as_list = encoded_text.toarray().tolist()
    
    labels = [1 if labels_as_list[i] == 'positive' else 0 for i in range(len(encoded_text_as_list))]

    X = np.array(encoded_text_as_list)
    y = np.array(labels)

    # Flatten the embeddings for use with Naive Bayes
    # X = X.reshape(X.shape[0], -1)

   
    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(len(X_train))
    print(len(X_test))

    return X_train, X_test, y_train, y_test

def log_params(params):
    mlflow.log_params(params)

if __name__ == "__main__":
    
    os.makedirs('figures', exist_ok=True)
    os.makedirs('loss_curves', exist_ok=True)

    mlflow.set_experiment("gradient-boosting-tfidf")

    df = load_dataset(settings.DATA_PATH)  # Replace with actual path
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [200, 300, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        # 'min_samples_split': [2, 5, 10],
        # 'min_samples_leaf': [1, 2, 4],
        # 'subsample': [0.8, 1.0],
        # 'max_features': ['auto', 'sqrt', 'log2']
    }

    run_name = f"model_{MODEL.__name__}_GB_ND{NUM_DATA}_BS{BATCH_SIZE}"

    with mlflow.start_run(run_name=run_name) as run:
        log_params({
            "batch_size": BATCH_SIZE,
            "param_grid": param_grid,
            "vectorizer": VECTORIZER.__name__,
        })

        # Initialize GridSearchCV
        grid_search = GridSearchCV(MODEL(), param_grid, cv=5, scoring='accuracy', n_jobs=-1, return_train_score=True)
        grid_search.fit(X_train, y_train)

        # Log the best hyperparameters
        best_params = grid_search.best_params_
        mlflow.log_params(best_params)

        # Best model from GridSearchCV
        best_model = grid_search.best_estimator_

        train_preds = best_model.predict(X_train)
        test_preds = best_model.predict(X_test)
        train_probs = best_model.predict_proba(X_train)
        test_probs = best_model.predict_proba(X_test)

        train_loss = log_loss(y_train, train_probs)
        test_loss = log_loss(y_test, test_probs)
        train_accuracy = accuracy_score(y_train, train_preds)
        test_accuracy = accuracy_score(y_test, test_preds)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })

        results = grid_search.cv_results_
        param_names = list(param_grid.keys())

        for param_name in param_names:
            param_values = results[f'param_{param_name}']
            train_scores = results['mean_train_score']
            test_scores = results['mean_test_score']

            # Log each parameter set's metrics
            for i, param_value in enumerate(param_values):
                param_str = f"{param_name}_{param_value}".replace('.', '_')  # Replace dots with underscores to follow naming conventions
                mlflow.log_metrics({
                    f"train_accuracy_{param_str}": train_scores[i],
                    f"test_accuracy_{param_str}": test_scores[i]
                })

            # Plotting train and test scores
            plt.figure(figsize=(10, 5))
            plt.plot(param_values, train_scores, label='Train Accuracy', marker='o')
            plt.plot(param_values, test_scores, label='Test Accuracy', marker='o')
            plt.xlabel(param_name)
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title(f'Train and Test Accuracy vs {param_name}')
            plot_filename = f"./figures/accuracy_vs_{param_name}_{run.info.run_id}.png"
            plt.savefig(plot_filename)
            mlflow.log_artifact(plot_filename)

        cm = confusion_matrix(y_test, test_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        
        cm_filename = f"./figures/confusion_matrix_{run.info.run_id}.png"
        plt.savefig(cm_filename)
        mlflow.log_artifact(cm_filename)

        mlflow.sklearn.log_model(best_model, "model")