import math
import torch
import torch.nn as nn
import torch.utils
import pandas as pd
import numpy as np
# from torchnlp.encoders.text import SpacyEncoder, pad_tensor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



"""
Equation of LSTM cells:

a. f_t= sigmoid(Uf*x_t+ Vf*h_(t-1)+ b_f)
b. C_t'= f_t* C_(t-1)
c. i_t= sigmoid(Ui*x_t+ Vi*h_(t-1) + b_i)
d. g_t= tanh(Ug*x_t+ Vg*h_(t-1)+ b_g)
e. C_t= (i_t * g_t)+ f_t* C_(t-1)
f. h_t= o_t * tanh(C_t) 
"""


class CustomLSTM(nn.Module):
    def __init__(self, input_size:int, hidden_size:int):
        super().__init__()
        self.input_size= input_size
        self.hidden_size= hidden_size

        # Weight intitialization for input gate(i_t)
        self.U_i= nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_i= nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_i= nn.Parameter(torch.Tensor(hidden_size))

        # Forget gate
        self.U_f= nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_f= nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_f= nn.Parameter(torch.Tensor(hidden_size))

        # C_t
        self.U_c= nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_c= nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_c= nn.Parameter(torch.Tensor(hidden_size))

        # o_t
        self.U_o= nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_o= nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_o= nn.Parameter(torch.Tensor(hidden_size))
        
        self.init_weights()

    
    def init_weights(self):
        stdv= 1.0/math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    # Feed forward operation
    def forward(self, x, init_states= None):
        """Here X.shape is in the form of (batch_size, sequence_size, input_size)"""
        batch_size, seq_size, _= x.size()
        hidden_seq= []

        if init_states is None:
            h_t, c_t= (torch.zeros(batch_size, self.hidden_size).to(x.device),
                       torch.zeros(batch_size, self.hidden_size).to(x.device))
        else:
            h_t, c_t= init_states
        
        for t in range(seq_size):
            x_t= x[:, t, :]

            i_t= torch.sigmoid(x_t @ self.U_i+ h_t@ self.V_i + self.b_i)
            f_t= torch.sigmoid(x_t @ self.U_f + h_t@ self.V_f + self.b_f)
            g_t= torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
            o_t= torch.sigmoid(x_t @ self.U_o + h_t@self.V_o + self.b_o)
            c_t= f_t * c_t + i_t *g_t
            h_t= o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        # reshape hidden sequence
        hidden_seq= torch.cat(hidden_seq, dim=0)
        hidden_seq= hidden_seq.transpose(0,1).contiguous()
        return hidden_seq, (h_t, c_t)
    

class LstmNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.embedding= nn.Embedding(500, 32)
        self.lstm= CustomLSTM(32, 32)
        self.fc1= nn.Linear(32, 2)
    
    def forward(self,x):
        x_= self.embedding(x)
        x_, (h_n, c_n )= self.lstm(x_)
        x_= (x_[:, -1, :])
        x_= self.fc1(x_)
        return x_



def load_dataset(csv_filepath):
    return pd.read_csv(csv_filepath, nrows=5000)


def preprocess_data(data):
    df= load_dataset('data/reviews/Reviews.csv')
    df= df.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time', 'Summary'], axis=1)
    df= df[df['Score']!= 3]

    df['Score']= df['Score'].apply(lambda i: 'positive' if i> 4 else 'negative')
    df['Text']= df['Text'].apply(lambda x: x.lower())

    df.columns= ['labels', 'text']
    idx_positive= df[df['labels']=='positive'].index
    nbr_to_drop= len(df)- len(idx_positive)
    drop_indices= np.random.choice(idx_positive, nbr_to_drop, replace=False)
    df= df.drop(drop_indices)

    text_as_list= df['text'].tolist()
    labels_as_list= df['labels'].tolist()

    vectorizer= TfidfVectorizer(max_features=50)
    encoded_text= vectorizer.fit_transform(text_as_list)

    encoded_text_as_list= encoded_text.toarray().tolist()
    
    
    labels= [1 if labels_as_list[i] == 'positive' else 0 for i in range(len(encoded_text_as_list))]

    X= torch.Tensor(encoded_text_as_list)
    y= torch.tensor(labels)

    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state=42)

    ds_train= torch.utils.data.TensorDataset(X_train, y_train)
    ds_test= torch.utils.data.TensorDataset(X_test, y_test)

    train_loader= torch.utils.data.DataLoader(ds_train, batch_size=6, shuffle=True)
    test_loader= torch.utils.data.DataLoader(ds_test, batch_size=6, shuffle=True)
    print(len(X_test))

    # for batch_idx, batch in enumerate(train_loader):
    #     print("Batch:", batch_idx + 1)
        


    return train_loader, test_loader





if __name__ == "__main__":
    device = torch.device('cpu')
    classifier = LstmNetwork().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    df = load_dataset('data/reviews/Reviews.csv')
    train_loader, test_loader = preprocess_data(df)

    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(1000):
        print(f"Epoch {epoch+1}/100")
        total_train_loss = 0
        total_correct_train = 0
        
        classifier.train()
        for i, (datapoints, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            optimizer.zero_grad()
            preds = classifier(datapoints.to(device).long())
            loss = criterion(preds, labels.to(device))
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_correct_train += (preds.argmax(dim=1) == labels.to(device)).float().sum().item()
            
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
                preds = classifier(datapoints_.to(device).long())
                loss = criterion(preds, labels_.to(device))
                total_test_loss += loss.item()
                total_correct_test += (preds.argmax(dim=1) == labels_.to(device)).float().sum().item()
                
                all_preds.extend(preds.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels_.cpu().numpy())
                
        test_loss = total_test_loss / len(test_loader)
        test_accuracy = total_correct_test / len(test_loader.dataset)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Plotting training and test loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()