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
    return pd.read_csv(csv_filepath, nrows=1000)


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





if __name__== "__main__":
    epoch_bar = tqdm(range(10),
                 desc="Training",
                 position=0,
                 total=2)
    acc=0

    device= torch.device('cpu')
    classifier= LstmNetwork().to(device)
    optimizer= optim.Adam(classifier.parameters(), lr= 0.001)
    criterion= nn.CrossEntropyLoss()
        
    df= load_dataset('data/reviews/Reviews.csv')
    train_loader, test_loader= preprocess_data(df)
    
    for epoch in tqdm(range(10), desc="training", position=0, total=2):
        batch_bar= tqdm(enumerate(train_loader), desc="Epoch: {}".format(str(epoch)),
                        position=1, total=len(train_loader))
        
        for i, (datapoint, labels) in batch_bar:
            optimizer.zero_grad()

            preds= classifier(datapoint.long().to(device))
            loss= criterion(preds, labels).to(device)
            loss.backward()
            optimizer.step()
            if (i + 1) % 50 == 0:
                acc = 0
                
                with torch.no_grad():
                    for  i, (datapoints_, labels_) in enumerate(test_loader):
                        preds = classifier(datapoints_.to(device))
                        acc += (preds.argmax(dim=1) == labels_.to(device)).float().sum().cpu().item()
                # acc /= len(X_test)
                acc /= 161

            batch_bar.set_postfix(loss=loss.cpu().item(),
                                accuracy="{:.2f}".format(acc),
                                epoch=epoch)
            batch_bar.update()

            
        epoch_bar.set_postfix(loss=loss.cpu().item(),
                            accuracy="{:.2f}".format(acc),
                            epoch=epoch)
        epoch_bar.update()
