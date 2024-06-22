import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, log_loss
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import mlflow
import os
import nltk
from config import settings

nltk.download('punkt')
nltk.download('stopwords')

NUM_DATA = 10000
LR_RATE = 0.001
BATCH_SIZE = 16
REGULARIZATION = 'l2'  # 'l1' or 'l2'
REGULARIZATION_WEIGHT = 0.01
MAX_ITER = 150
SOLVER= 'sag'
VECTORIZER= CountVectorizer

STOP_WORDS = stopwords.words('english')

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
    np.random.seed(42)
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
    # vectorizer= TfidfVectorizer()
    encoded_text = vectorizer.fit_transform(text_as_list)
    
    X = encoded_text.toarray()
    y = np.array([1 if label == 'positive' else 0 for label in labels_as_list])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test, vectorizer

def log_params(params):
    mlflow.log_params(params)

if __name__ == "__main__":
    
    os.makedirs('figs', exist_ok=True)
    os.makedirs('lcurve', exist_ok=True)

    mlflow.set_experiment("logreg-countvectorizer")

    df = load_dataset(settings.DATA_PATH)
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(df)

    classifier = LogisticRegression(
        # random_state=42, 
        penalty=REGULARIZATION,
        C=1/REGULARIZATION_WEIGHT,
        solver=SOLVER,
        max_iter=MAX_ITER
    )

    classifier.fit(X_train, y_train)

    train_preds = classifier.predict(X_train)
    test_preds = classifier.predict(X_test)
    train_probs = classifier.predict_proba(X_train)
    test_probs = classifier.predict_proba(X_test)

    train_loss = log_loss(y_train, train_probs)
    test_loss = log_loss(y_test, test_probs)
    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)

    run_name = f"ND{NUM_DATA}_LR{LR_RATE}_BS{BATCH_SIZE}_REG{REGULARIZATION}_RW{REGULARIZATION_WEIGHT}_MAXITER{MAX_ITER}"

    with mlflow.start_run(run_name=run_name) as run:
        params = {
            "learning_rate": LR_RATE,
            "batch_size": BATCH_SIZE,
            "regularization": REGULARIZATION,
            "regularization_weight": REGULARIZATION_WEIGHT,
            "max_iter": MAX_ITER,
            "solver": SOLVER,
            "vectorizer": VECTORIZER.__name__
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
        lcurve_filename = f"./lcurve/loss_curve_{run_id}.png"
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
