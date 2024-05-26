import math
import torch
import torch.nn as nn
import torch.utils
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config import settings
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.pytorch
import gensim.downloader as api
import os
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
nltk.download('punkt')
nltk.download('stopwords')
"""
Equation of LSTM cells:

a. f_t= sigmoid(Uf*x_t+ Vf*h_(t-1)+ b_f)
b. C_t'= f_t* C_(t-1)
c. i_t= sigmoid(Ui*x_t+ Vi*h_(t-1) + b_i)
d. g_t= tanh(Ug*x_t+ Vg*h_(t-1)+ b_g)
e. C_t= (i_t * g_t)+ f_t* C_(t-1)
f. h_t= o_t * tanh(C_t) 
"""

NUM_DATA= 100
EPOCHS= 9
LR_RATE= 0.01
DROPOUT= 0.3
NUM_LAYERS= 6
HIDDEN_SIZE= 2
EMBEDDING_MODEL= 'glove-wiki-gigaword-100'
EMBEDDING_DIM= 100
BATCH_SIZE= 16
MAX_SEQ_LEN= 100


class CustomLSTM(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_layers:int= NUM_LAYERS, dropout_prob:float=DROPOUT):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            self.lstm_layers.append(LSTMCell(input_dim, hidden_size))

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, init_states=None):
        """Here X.shape is in the form of (batch_size, sequence_size, input_size)"""
        batch_size, seq_size, _ = x.size()
        hidden_seq = []

        if init_states is None:
            init_states = [(torch.zeros(batch_size, self.hidden_size).to(x.device),
                            torch.zeros(batch_size, self.hidden_size).to(x.device)) for _ in range(self.num_layers)]

        h_t, c_t = zip(*init_states)
        h_t, c_t= list(h_t), list(c_t)

        for t in range(seq_size):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h_t[layer], c_t[layer] = self.lstm_layers[layer](x_t, (h_t[layer], c_t[layer]))
                x_t = self.dropout(h_t[layer])

            hidden_seq.append(h_t[-1].unsqueeze(0))

        # reshape hidden sequence
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight initialization for input gate (i_t)
        self.U_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # Forget gate
        self.U_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # C_t
        self.U_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        # o_t
        self.U_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, states):
        h_t, c_t = states

        i_t = torch.sigmoid(x @ self.U_i + h_t @ self.V_i + self.b_i)
        f_t = torch.sigmoid(x @ self.U_f + h_t @ self.V_f + self.b_f)
        g_t = torch.tanh(x @ self.U_c + h_t @ self.V_c + self.b_c)
        o_t = torch.sigmoid(x @ self.U_o + h_t @ self.V_o + self.b_o)
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t



class LstmNetwork(nn.Module):
    def __init__(self, num_hidden= HIDDEN_SIZE ,num_layers:int=NUM_LAYERS, dropout_prob:float=DROPOUT):
        super().__init__()
        self.lstm = CustomLSTM(EMBEDDING_DIM, num_hidden, num_layers, dropout_prob)  
        self.dropout = nn.Dropout(dropout_prob)  
        self.fc1 = nn.Linear(num_hidden, 1)  
    
    def forward(self, x):
        x_, (h_n, c_n) = self.lstm(x)
        x_ = (x_[:, -1, :])
        x_ = self.dropout(x_)  
        x_ = self.fc1(x_)
        return x_

def load_dataset(csv_filepath):
    return pd.read_csv(csv_filepath, nrows=NUM_DATA)


def preprocess_data(data):
    df= data.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time', 'Summary'], axis=1)
    df= df[df['Score']!= 3]

    df['Score']= df['Score'].apply(lambda i: 'positive' if i> 4 else 'negative')
    df['Text']= df['Text'].apply(lambda x: preprocess_text(x))

    df.columns= ['labels', 'text']
    idx_positive= df[df['labels']=='positive'].index
    nbr_to_drop= len(df)- len(idx_positive)
    drop_indices= np.random.choice(idx_positive, nbr_to_drop, replace=False)
    df= df.drop(drop_indices)

    text_as_list= df['text'].tolist()
    labels_as_list= df['labels'].tolist()

    # vectorizer= TfidfVectorizer(max_features=100, sublinear_tf=True)
    # encoded_text= vectorizer.fit_transform(text_as_list)

    # encoded_text_as_list= encoded_text.toarray().tolist()
    
        # Load the pre-trained GloVe model
    # model = api.load("glove-twitter-25")
    model = api.load(EMBEDDING_MODEL)
    # model = api.load("glove-wiki-gigaword-200")

    def encode_text(text, model, max_seq_len=MAX_SEQ_LEN):
        words = text.split()
        words = words[:max_seq_len]
        embedding = np.zeros((max_seq_len, model.vector_size))
        for i, word in enumerate(words):
            if word in model:
                embedding[i] = model[word]
        return embedding

    encoded_text_as_list = [encode_text(text, model) for text in text_as_list]

    labels= [1 if labels_as_list[i] == 'positive' else 0 for i in range(len(encoded_text_as_list))]

    X= torch.Tensor(encoded_text_as_list)
    y= torch.tensor(labels)

    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state=42)

    ds_train= torch.utils.data.TensorDataset(X_train, y_train)
    ds_test= torch.utils.data.TensorDataset(X_test, y_test)

    train_loader= torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader= torch.utils.data.DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)

    # for batch_idx, batch in enumerate(train_loader):
    #     print("Batch:", batch_idx + 1)
    return train_loader, test_loader

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenize text
    # words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in text if word not in stop_words]
    return ' '.join(words)

def log_params(params):
    mlflow.log_params(params)

def run_epoch(classifier, data_loader, criterion, optimizer=None, training=True):
    epoch_loss = 0
    correct_preds = 0
    all_preds = []
    all_labels = []

    if training:
        classifier.train()
    else:
        classifier.eval()

    for datapoints, labels in tqdm(data_loader, desc="Training" if training else "Testing", colour='#85929E'):
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            preds = classifier(datapoints.to(device)).squeeze(1)
            loss = criterion(preds, labels.to(device).float())
            if training:
                loss.backward()
                optimizer.step()
        epoch_loss += loss.item()
        correct_preds += ((torch.sigmoid(preds) > 0.5) == labels.to(device)).float().sum().item()
        all_preds.extend((torch.sigmoid(preds) > 0.5).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss /= len(data_loader)
    accuracy = correct_preds / len(data_loader.dataset)
    
    return epoch_loss, accuracy, all_preds, all_labels

if __name__ == "__main__":


    os.makedirs('figs', exist_ok=True)
    mlflow.set_experiment("lstm_experiment")

    device = torch.device('cpu')
    classifier = LstmNetwork().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=LR_RATE)
    criterion = nn.BCEWithLogitsLoss()

    df = load_dataset(settings.DATA_PATH)
    train_loader, test_loader = preprocess_data(df)

    train_losses = []
    test_losses = []
    test_accuracies = []

    run_name = f"ND{NUM_DATA}_EP{EPOCHS}_LR{LR_RATE}_DO{DROPOUT}_NL{NUM_LAYERS}_HS{HIDDEN_SIZE}_{EMBEDDING_MODEL.split('-')[0]}{EMBEDDING_DIM}_BS{BATCH_SIZE}_SEQ{MAX_SEQ_LEN}"
    
    with mlflow.start_run(run_name=run_name) as run:
        params = {
            "learning_rate": LR_RATE,
            "word2vec_model": EMBEDDING_MODEL,
            "num_epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "dropout": DROPOUT,
            "embedding_dim": EMBEDDING_DIM
        }
        log_params(params)
        run_id = run.info.run_id

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS}")
            train_loss, train_accuracy, _, _ = run_epoch(classifier, train_loader, criterion, optimizer, training=True)
            train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            test_loss, test_accuracy, all_preds, all_labels = run_epoch(classifier, test_loader, criterion, training=False)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy
            }, step=epoch)

        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        cm_filename = f"./figs/confusion_matrix_{run_id}.png"
        plt.savefig(cm_filename)
        mlflow.log_artifact(cm_filename)

        mlflow.pytorch.log_model(classifier, "model")

        # # Plotting training and test loss curves
        # plt.figure(figsize=(10, 5))
        # plt.plot(train_losses, label='Train Loss')
        # plt.plot(test_losses, label='Test Loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.title('Training and Test Loss')
        # plot_filename = f"loss_curve_{run_id}.png"
        # plt.savefig(plot_filename)
        # # plt.show()
        # mlflow.log_artifact(plot_filename)