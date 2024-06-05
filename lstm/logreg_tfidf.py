import torch
import torch.nn as nn
import torch.utils
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# from torchnlp.encoders.text import SpacyEncoder, pad_tensor
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.pytorch
import os
import nltk
from config import settings
nltk.download('punkt')
nltk.download('stopwords')


NUM_DATA= 10000
EPOCHS= 5000
LR_RATE= 0.005
DROPOUT= 0.7
NUM_LAYERS= 1
HIDDEN_SIZE= 6
BATCH_SIZE= 16




class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear= nn.Linear(input_dim, output_dim)


    def forward(self, x):
        out= self.linear(x)
        return out

    
def load_dataset(csv_filepath):
    return pd.read_csv(csv_filepath, nrows=NUM_DATA)


def preprocess_data(data):
    df= data.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time', 'Summary'], axis=1)
    df= df[df['Score']!= 3]

    df['Score']= df['Score'].apply(lambda i: 'positive' if i> 4 else 'negative')
    df['Text'] = df['Text'].apply(lambda x: preprocess_text(x))

    df.columns= ['labels', 'text']
    idx_positive= df[df['labels']=='positive'].index
    nbr_to_drop= len(df)- len(idx_positive)
    drop_indices= np.random.choice(idx_positive, nbr_to_drop, replace=False)
    df= df.drop(drop_indices)

    text_as_list= df['text'].tolist()
    labels_as_list= df['labels'].tolist()

    vectorizer= TfidfVectorizer(max_features=100, sublinear_tf=True)
    encoded_text= vectorizer.fit_transform(text_as_list)
    
    tfidf_features= encoded_text.shape[1]

    encoded_text_as_list= encoded_text.toarray().tolist()
    
    labels= [1 if labels_as_list[i] == 'positive' else 0 for i in range(len(encoded_text_as_list))]

    X= torch.Tensor(encoded_text_as_list)
    y= torch.tensor(labels)
    print(y.shape)
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state=42)

    ds_train= torch.utils.data.TensorDataset(X_train, y_train)
    ds_test= torch.utils.data.TensorDataset(X_test, y_test)

    train_loader= torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader= torch.utils.data.DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader, tfidf_features

def preprocess_text(text):

    text = text.lower()

    text = BeautifulSoup(text, "html.parser").get_text()

    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    #word tokenizer
    words = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def log_params(params):
    mlflow.log_params(params)

if __name__ == "__main__":
    
    os.makedirs('figs', exist_ok=True)
    os.makedirs('lcurve', exist_ok=True)

    mlflow.set_experiment("logreg-tfidf")
    device = torch.device('cpu')

    df = load_dataset(settings.DATA_PATH)
    train_loader, test_loader, tfidf_features = preprocess_data(df)

    
    # # classifier = LstmNetwork(input_size= tfidf_features).to(device)
    classifier= LogisticRegressionModel(input_dim= tfidf_features, output_dim=1).to(device)
    optimizer = optim.SGD(classifier.parameters(), lr=LR_RATE)
    criterion = nn.BCEWithLogitsLoss()

    

    train_losses = []
    test_losses = []
    test_accuracies = []
    run_name = f"ND{NUM_DATA}_EP{EPOCHS}_LR{LR_RATE}_DO{DROPOUT}_NL{NUM_LAYERS}_HS{HIDDEN_SIZE}_BS{BATCH_SIZE}"

    with mlflow.start_run(run_name=run_name) as run:
        params= {
            "learning_rate": LR_RATE,
            "num_epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "dropout": DROPOUT
        }
        log_params(params)
        run_id= run.info.run_id
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS}")
            total_train_loss = 0
            total_correct_train = 0
            
            classifier.train()
            for i, (datapoints, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
                optimizer.zero_grad()
                # datapoints = datapoints.unsqueeze(1)   
                preds = classifier(datapoints.to(device)).squeeze(1)
                loss = criterion(preds, labels.to(device).float())
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                total_correct_train += ((torch.sigmoid(preds) > 0.5) == labels.to(device)).float().sum().item()

            train_loss = total_train_loss / len(train_loader)
            train_accuracy = total_correct_train / len(train_loader.dataset)
            train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            classifier.eval()
            total_correct_test = 0
            total_test_loss = 0

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for i, (datapoints_, labels_) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing"):
                    # datapoints_ = datapoints_.unsqueeze(1) 
                
                    preds = classifier(datapoints_.to(device)).squeeze(1)
                    loss = criterion(preds, labels_.to(device).float())
                    total_test_loss += loss.item()
                    total_correct_test += ((torch.sigmoid(preds) > 0.5) == labels_.to(device)).float().sum().item()

                    all_preds.extend((torch.sigmoid(preds) > 0.5).cpu().numpy())
                    all_labels.extend(labels_.cpu().numpy())

            test_loss = total_test_loss / len(test_loader)
            test_accuracy = total_correct_test / len(test_loader.dataset)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy
            }, step= epoch)

        # Plotting training and test loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Test Loss')
        lcurve_filename= f"./lcurve/loss_curve_{run_id}.png"
        plt.savefig(lcurve_filename)
        # plt.show()
        mlflow.log_artifact(lcurve_filename)

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        
        cm_filename = f"./figs/confusion_matrix_{run_id}.png"
        plt.savefig(cm_filename)
        # plt.show()
        mlflow.log_artifact(cm_filename)

        mlflow.pytorch.log_model(classifier, "model")