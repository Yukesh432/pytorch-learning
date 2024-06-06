import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, log_loss
import mlflow
import mlflow.sklearn
import os
import nltk
from config import settings
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB, GaussianNB
import gensim.downloader as api
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from nltk.stem import PorterStemmer

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
EMBEDDING_MODEL= 'fasttext-wiki-news-subwords-300'
# EMBEDDING_DIM= 300
MAX_SEQ_LEN= 1000
MODEL= 'BernoulliNB'

STOP_WORDS = set(stopwords.words('english'))

def load_dataset(csv_filepath):
    return pd.read_csv(csv_filepath, nrows=NUM_DATA)

def preprocess_text(text):
    text = text.lower()
    text = text.replace(",", "000000").replace("000", "m").replace("000", "k").replace("", "").replace("", "")
    text = text.replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")
    text = text.replace("n't", " not").replace("what's", "what is").replace("it's", "it is")
    text = text.replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")
    text = text.replace("he's", "he is").replace("she's", "she is").replace("'s", " own")
    text = text.replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")
    text = text.replace("€", " euro ").replace("'ll", " will")
    text = re.sub(r'([0-9]+)000000', r'\1m', text)
    text = re.sub(r'([0-9]+)000', r'\1k', text)

    porter = PorterStemmer()
    pattern = re.compile('\W')

    if isinstance(text, str):
        text = re.sub(pattern, ' ', text)

    if isinstance(text, str):
        text = porter.stem(text)
        example1 = BeautifulSoup(text, "html.parser")
        text = example1.get_text()

    words = word_tokenize(text)
    words = [word for word in words if word not in STOP_WORDS]
    return ' '.join(words)

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

    model = api.load(EMBEDDING_MODEL)

    def encode_text(text, model, max_seq_len=MAX_SEQ_LEN):
        words = text.split()
        words = words[:max_seq_len]
        embedding = np.zeros((max_seq_len, model.vector_size))
        for i, word in enumerate(words):
            if word in model:
                embedding[i] = model[word]
        return embedding


    encoded_text_as_list = [encode_text(text, model) for text in text_as_list]

    
    labels = [1 if labels_as_list[i] == 'positive' else 0 for i in range(len(encoded_text_as_list))]

    X = np.array(encoded_text_as_list)
    y = np.array(labels)

    # Flatten the embeddings for use with Naive Bayes
    X = X.reshape(X.shape[0], -1)

   
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(len(X_train))
    print(len(X_test))

    return X_train, X_test, y_train, y_test

def log_params(params):
    mlflow.log_params(params)

if __name__ == "__main__":
    
    os.makedirs('figures', exist_ok=True)
    os.makedirs('loss_curves', exist_ok=True)

    mlflow.set_experiment("naive-bayes-w2v")

    df = load_dataset(settings.DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # classifier = MultinomialNB()
    classifier= MODEL()
    classifier.fit(X_train, y_train)

    train_preds = classifier.predict(X_train)
    test_preds = classifier.predict(X_test)
    train_probs = classifier.predict_proba(X_train)
    test_probs = classifier.predict_proba(X_test)

    train_loss = log_loss(y_train, train_probs)
    test_loss = log_loss(y_test, test_probs)
    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)

    run_name = f"model_{MODEL}_NB_ND{NUM_DATA}_BS{BATCH_SIZE}_EMB_{EMBEDDING_MODEL}_MSQL{MAX_SEQ_LEN}"

    with mlflow.start_run(run_name=run_name) as run:
        params = {
            "batch_size": BATCH_SIZE
        }
        log_params(params)
        run_id = run.info.run_id

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })

        plt.figure(figsize=(10, 5))
        plt.plot([train_loss], label='Train Loss')
        plt.plot([test_loss], label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Test Loss')
        lcurve_filename= f"./lcurve/loss_curve_{run_id}.png"
        plt.savefig(lcurve_filename)
        mlflow.log_artifact(lcurve_filename)

        cm = confusion_matrix(y_test, test_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        
        cm_filename = f"./figs/confusion_matrix_{run_id}.png"
        plt.savefig(cm_filename)
        mlflow.log_artifact(cm_filename)

        mlflow.sklearn.log_model(classifier, "model")
