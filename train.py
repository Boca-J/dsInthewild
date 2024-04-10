from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as metrics
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import seaborn as sns
import matplotlib.pyplot as plt

from data import *

class ournet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x

def train_builtin(model,train_features,train_labels,test_features,test_labels):
    model.fit(train_features, train_labels)
    predict = model.predict(test_features)
    f1score = metrics.f1_score(test_labels, predict)
    auroc = metrics.roc_auc_score(test_labels, predict)
    accuracy = metrics.accuracy_score(test_labels, predict)
    print(f1score, auroc, accuracy)
    return f1score,model

def train(df_ls):
    f1_scores=[]
    models=[]
    for df in df_ls:
        f1=[]
        m=[]
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df=train_df.reset_index()
        test_df=test_df.reset_index()
        train_features=train_df[['ALQ121','DBQ700', 'DBD910','SMQ040']].values
        train_labels=train_df['DIQ010'].values
        test_features=test_df[['ALQ121','DBQ700', 'DBD910','SMQ040']].values
        test_labels = test_df['DIQ010'].values

        print(df['DIQ010'].mean(),'are label 1')
        weight0=len(df['DIQ010'])/(2*len(df[df['DIQ010']==0]))
        weight1 = len(df['DIQ010']) / (2 * len(df[df['DIQ010'] == 1]))
        train_weights=[weight0 if label==0 else weight1 for label in train_labels]
        test_weights = [weight0 if label == 0 else weight1 for label in test_labels]

        print('knn')
        knn=KNeighborsClassifier()
        a,b=train_builtin(knn,train_features,train_labels,test_features,test_labels)
        f1.append(a)
        m.append(b)

        print('random forest')
        randomforest=RandomForestClassifier()
        a, b = train_builtin(randomforest, train_features, train_labels, test_features, test_labels)
        f1.append(a)
        m.append(b)

        print('adaboost')
        ada=AdaBoostClassifier()
        a, b = train_builtin(ada, train_features, train_labels, test_features, test_labels)
        f1.append(a)
        m.append(b)

        print('logistic regression')
        regression=LogisticRegression(class_weight='balanced')
        a, b = train_builtin(regression, train_features, train_labels, test_features, test_labels)
        f1.append(a)
        m.append(b)

        print('svc rbf')
        svc_rbf=SVC(class_weight='balanced',kernel='rbf')
        a, b = train_builtin(svc_rbf, train_features, train_labels, test_features, test_labels)
        f1.append(a)
        m.append(b)

        print('svc poly')
        svc_poly = SVC(class_weight='balanced', kernel='poly')
        a, b = train_builtin(svc_poly, train_features, train_labels, test_features, test_labels)
        f1.append(a)
        m.append(b)

        print('mlp')
        mlp=MLPClassifier()
        a, b = train_builtin(mlp, train_features, train_labels, test_features, test_labels)
        f1.append(a)
        m.append(b)

        train_dataset=Customdataset(train_df)
        test_dataset=Customdataset(test_df)
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

        print('ournet')
        model = ournet(4, 100, 200, 1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight1))
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(10):
            model.train()
            total_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                labels=labels.reshape((-1,1))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(total_loss/100)

        model.eval()
        predict=torch.sigmoid(model(torch.tensor(test_features).float()).detach())
        predict=(predict>=0.5).float()
        f1score = metrics.f1_score(test_labels, predict)
        auroc = metrics.roc_auc_score(test_labels, predict)
        accuracy = metrics.accuracy_score(test_labels, predict)
        print(f1score, auroc, accuracy)
        f1.append(f1score)
        m.append(model)
        f1_scores.append(f1)
        models.append(m)
    return models,f1_scores


def plot_graphs(df_ls):
    if not os.path.exists('plots'):
        os.makedirs('plots')

    for i,df in enumerate(df_ls):
        for column in ['ALQ121','DBQ700', 'DBD910','SMQ040']:
            sns.boxplot(data=df,x='RIDRETH3',y=column,showfliers=False)
            plt.title(f'boxplot of {column}_{i}')
            plt.xlabel('Race')
            plt.ylabel(column)
            output_file = os.path.join('plots', f'{column}_{i}_box.png')
            plt.savefig(output_file)
            plt.close()

if __name__ == '__main__':
    filenames = [['P_DEMO.XPT', 'P_DIQ.XPT', 'P_ALQ.XPT', 'P_DBQ.XPT', 'P_SMQ.XPT']]
    columns_to_append = [['SEQN', 'RIDRETH3'], ['SEQN', 'DIQ010'], ['SEQN', 'ALQ121'], ['SEQN', 'DBQ700', 'DBD910'],
                         ['SEQN', 'SMQ040']]
    columns_replace = {'RIDRETH3': ([7, '.'], [np.nan, np.nan]),
                       'DIQ010': ([2,3, 7, 9, '.'], [0,1, np.nan, np.nan, np.nan]),
                       'ALQ121': ([77, 99, '.'], [np.nan, np.nan, 0]),
                       'DBQ700': ([7, 9, '.'], [np.nan, np.nan, np.nan]),
                       'DBD910': ([6666, 7777, 9999, '.'], [90, np.nan, np.nan, np.nan]),
                       'SMQ040': ([7, 9, '.'], [np.nan, np.nan, 3])}
    df = data_collection_merge(filenames, columns_to_append)
    df = data_cleaning(df, columns_replace)
    df = df.dropna()
    combined_models,combined_f1_scores=train([df])
    print(combined_models,combined_f1_scores)

    separated_df=[]
    grouped=df.groupby('RIDRETH3')
    for _,group_df in grouped:
        separated_df.append(group_df)
    separated_models,separated_f1_scores=train(separated_df)
    print(separated_models,separated_f1_scores)

    separated_df = []
    grouped = df.groupby('DIQ010')
    for _, group_df in grouped:
        separated_df.append(group_df)
    plot_graphs(separated_df)