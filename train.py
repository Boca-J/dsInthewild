
import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, PredefinedSplit
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score,roc_auc_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from data import *

import torch
import torch.nn as nn

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For CUDA to ensure reproducibility on GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # For deterministic algorithm, might reduce performance
    torch.backends.cudnn.benchmark = False  # Disable optimization to ensure reproducibility


def plot_graphs(df_ls):
    if not os.path.exists('plots'):
        os.makedirs('plots')

    for i,df in enumerate(df_ls):
        for column in ['ALQ130','DBD900', 'DBD910','SMD650','PAD660','PAD675','WHQ040','SLD012','OCQ180']:
            sns.boxplot(data=df,x='RIDRETH3',y=column,showfliers=False)
            plt.title(f'boxplot of {column}_{i}')
            plt.xlabel('Race')
            plt.ylabel(column)
            output_file = os.path.join('plots', f'{column}_{i}_box.png')
            plt.savefig(output_file)
            plt.close()


def process_df(train_df, test_df):
    train_df=train_df.reset_index()
    test_df=test_df.reset_index()
    train_features1=train_df[['ALQ130','DBD900', 'DBD910','SMD650','PAD660','PAD675','WHQ040','SLD012','OCQ180']].values
    train_labels1=train_df['KIQ022'].values
    test_features=test_df[['ALQ130','DBD900', 'DBD910','SMD650','PAD660','PAD675','WHQ040','SLD012','OCQ180']].values
    test_labels = test_df['KIQ022'].values




    # The data is 15% vs 85% need to balance the data
    smote = SMOTE(random_state=42)
    train_features, train_labels = smote.fit_resample(train_features1, train_labels1)
    return train_features, train_labels, test_features, test_labels


class ournet(nn.Module):
    def __init__(self, input_dim, output_dim, depth=2, width=1000, hidden=None, eps=0.01, bias=True):
        super(ournet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eps = eps
        self.bias = bias

        if hidden is None:
            hidden = [width for _ in range(depth - 1)]

        self.hidden = hidden
        self.hidden_layers = nn.ModuleList()  # Use ModuleList to properly register modules

        # Building the layers
        previous_dim = self.input_dim
        for current_dim in self.hidden:
            layer = nn.Linear(previous_dim, current_dim, bias=self.bias)
            # layer.weight = nn.Parameter(torch.randn(layer.weight.shape) * eps, requires_grad=True)
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                layer.bias = nn.Parameter(torch.zeros_like(layer.bias), requires_grad=True)
            self.hidden_layers.append(layer)
            previous_dim = current_dim

        # Output layer
        final_layer = nn.Linear(self.hidden[-1], self.output_dim, bias=self.bias)
        nn.init.kaiming_normal_(final_layer.weight, nonlinearity='linear')
        self.hidden_layers.append(final_layer)

    def forward(self, x):
        for layer in self.hidden_layers[:-1]:
            x = nn.ReLU()(layer(x))
        return self.hidden_layers[-1](x)


def train_builtin(model,train_features,train_labels,test_features,test_labels):
    model.fit(train_features, train_labels)
    predict = model.predict(test_features)
    f1score = metrics.f1_score(test_labels, predict)
    auroc = metrics.roc_auc_score(test_labels, predict)
    accuracy = metrics.accuracy_score(test_labels, predict)
    print(f"f1_socre: {f1score}, auroc: {auroc}, accuracy: {accuracy}")
    return f1score,model

def find_param(model,param_grid, df):


    # grid_search = GridSearchCV(model, param_grid, cv=3,scoring='f1')  # cv is the number of folds, scoring can be adjusted
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)


    X = df[['ALQ130','DBD900', 'DBD910','SMD650','PAD660','PAD675','WHQ040','SLD012','OCQ180']].values
    y = df['KIQ022']
    # Fit the model to the data
    # grid_search.fit(X, y)

    random_search.fit(X,y)

    # Best parameters found
    print("Best parameters:", random_search.best_params_)

def find_param_for_net(train_df, test_df, weight1, depths, widths, epochs=100, lr=0.01):
        print("Train label distribution:", np.unique(train_df['KIQ022'], return_counts=True))
        print("Test label distribution:", np.unique(test_df['KIQ022'], return_counts=True))
        train_dataset = Customdataset(train_df)
        test_dataset = Customdataset(test_df)
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)
        test_features = test_df[
            ['ALQ130', 'DBD900', 'DBD910', 'SMD650', 'PAD660', 'PAD675',
             'WHQ040', 'SLD012', 'OCQ180']].values
        test_labels = test_df['KIQ022'].values

        best_f1_score = 0
        best_config = None
        best_model = None
        for depth in depths:
            for width in widths:
                print(f'Training model with depth {depth} and width {width}')
                model = ournet(9, 1, depth,
                               width)  # Assuming input_dim=9, output_dim=1 for binary classification
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight1))
                optimizer = optim.Adam(model.parameters(), lr=lr)

                # Training loop
                for epoch in range(epochs):
                    model.train()
                    total_loss = 0
                    for inputs, labels in train_loader:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        labels = labels.reshape((-1, 1))
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()

                    # if epoch % 50 == 0 or epoch == epochs - 1:  # Optional: limit how often you log
                    #     print(f"Epoch {epoch + 1}/{epochs} | Depth {depth}, Width {width} | Loss: {total_loss:.4f}")

                # Evaluation loop
                model.eval()
                predict = torch.sigmoid(
                    model(torch.tensor(test_features).float()).detach())

                predict = (predict >= 0.5).float()
                # print("Positive predictions:", (predict == 1).sum().item(), "/", len(predict))
                # print("Unique labels in test:", np.unique(test_labels))

                # 3, 200




                f1score = metrics.f1_score(test_labels, predict)

                print(f'Depth: {depth}, Width: {width}, F1 Score: {f1score}')


                if f1score > best_f1_score:
                    best_f1_score = f1score
                    best_config = (depth, width)
                    best_model = model

        return best_config, best_f1_score, best_model


def train_big_model(train_df,test_df):



    f1=[]
    m=[]
    model_names = []
 

    train_features, train_labels, test_features, test_labels = process_df(train_df, test_df)
    columns = ['ALQ130','DBD900', 'DBD910','SMD650','PAD660','PAD675','WHQ040','SLD012','OCQ180']
    train_df = pd.DataFrame(train_features, columns=columns)
    train_df['KIQ022'] = train_labels

    test_df = pd.DataFrame(test_features, columns=columns)
    test_df['KIQ022'] = test_labels

    # # print(df['KIQ022'].mean(),'are label 1')
    weight0=len(df['KIQ022'])/(2*len(df[df['KIQ022']==0]))
    weight1 = len(df['KIQ022']) / (2 * len(df[df['KIQ022'] == 1]))
    train_weights=[weight0 if label==0 else weight1 for label in train_labels]
    test_weights = [weight0 if label == 0 else weight1 for label in test_labels]

    # print('baseline: guessing 0')
    predict = np.zeros(test_labels.shape)
    f1score = metrics.f1_score(test_labels, predict)
    auroc = metrics.roc_auc_score(test_labels, predict)
    accuracy = metrics.accuracy_score(test_labels, predict)


    print(f1score, auroc, accuracy)

    print('knn')
    knn=KNeighborsClassifier(metric="minkowski", weights="uniform", n_neighbors=16)
    
    # param_grid = {
    #     'n_neighbors': range(2, 21),  # Testing values from 1 to 20
    #     'weights': ['uniform', 'distance'],
    #     'metric': ['euclidean', 'manhattan', 'minkowski']
    # }
    # find_param(knn,param_grid,train_df)
    # exit()
    #
    a,b=train_builtin(knn,train_features,train_labels,test_features,test_labels)
    f1.append(a)
    m.append(b)
    model_names.append('KNN')

    
    print(a, b, accuracy)


    print('random forest')
    # randomforest=RandomForestClassifier()

    randomforest = RandomForestClassifier(bootstrap = True,
                         criterion =  'entropy',
                         max_depth = 10,
                         max_features = 'log2',
                         min_samples_leaf = 4,
                         min_samples_split = 2,
                         n_estimators = 100,
                         random_state = 47)
    
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     # More trees can be better, but take longer to compute
    #     'max_depth': [10, 20, 30, None],
    #     # None means max depth not constrained
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'bootstrap': [True, False]
    # }
    # find_param(randomforest, param_grid,train_df)
    
    a, b = train_builtin(randomforest, train_features, train_labels, test_features, test_labels)
    f1.append(a)
    m.append(b)
    model_names.append('Random Forest')


    print('adaboost')
    ada = AdaBoostClassifier(algorithm='SAMME', learning_rate=0.01, n_estimators=50,estimator=DecisionTreeClassifier(max_depth=1))
    # ada = AdaBoostClassifier(algorithm='SAMME')
    a, b = train_builtin(ada, train_features, train_labels, test_features, test_labels)
    f1.append(a)
    m.append(b)
    model_names.append('AdaBoost')



    # param_grid = {
    #     'estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],
    #     'n_estimators': [50, 100, 200],  # Number of models to iteratively train
    #     'learning_rate': [0.01, 0.1, 1],  # Controls the contribution of each model
    #     'algorithm': ['SAMME']  # Algorithms to use
    # }
    # find_param(ada, param_grid, df)


    print('logistic regression')
    # regression=LogisticRegression(class_weight='balanced')
    regression = LogisticRegression(class_weight= None, C=0.01, max_iter=1000, penalty="l2", solver="lbfgs")


    a, b = train_builtin(regression, train_features, train_labels, test_features, test_labels)
    f1.append(a)
    m.append(b)
    model_names.append('Logistic Regression')


    param_grid = [
        {'solver': ['lbfgs'], 'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10, 100], 'class_weight': [None, 'balanced'], 'max_iter': [1000,10000, 20000, 30000]},
        {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100], 'class_weight': [None, 'balanced'], 'max_iter': [100, 200, 300]}
    ]
   # find_param(regression, param_grid, df)
    


    print('svc rbf')
    svc_rbf=SVC(class_weight='balanced', kernel='rbf', max_iter=20000, C = 10, gamma= 1)
    a, b = train_builtin(svc_rbf, train_features, train_labels, test_features, test_labels)
    f1.append(a)
    m.append(b)
    model_names.append('SVC RBF')

    # #
    # param_grid = {
    #     'C': [0.1, 1, 10],  # Regularization parameter
    #     'gamma': ['scale', 'auto', 0.01, 0.1, 1]  # Kernel coefficient
    # }
    # find_param(svc_rbf, param_grid, df)

    print('SVC Poly')
    svc_poly = SVC(
    class_weight='balanced',
    kernel='poly',
    C=10,         
    degree=5,      
    gamma='scale', 
    coef0=2  
            
    a, b = train_builtin(svc_poly, train_features, train_labels, test_features, test_labels)
    f1.append(a)
    m.append(b)
    model_names.append('SVC Poly')

    param_grid = {
        'C': [0.1, 1, 10],  # Regularization parameter
        'degree': [2, 3, 4],  # Degree of the polynomial
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient
        'coef0': [0, 0.5, 1]  # Independent term in kernel function
    }

    param_grid = {
        'C': [0.1, 1, 5, 10],  # Regularization parameter
        'degree': [2, 3, 4, 5],  # Degree of the polynomial
        'gamma': ['scale'],  # Kernel coefficient
        'coef0': [0, 0.5, 1,1.5, 2]  # Independent term in kernel function
    }
    # find_param(svc_poly, param_grid, df)


    print('mlp')
    # mlp=MLPClassifier(max_iter=1000)
    mlp = MLPClassifier(max_iter=1000, solver="adam",
                        learning_rate='constant',
                        hidden_layer_sizes=(50,), alpha=0.001,
                        activation='relu')
    a, b = train_builtin(mlp, train_features, train_labels, test_features, test_labels)
    f1.append(a)
    m.append(b)
    model_names.append('MLP')

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],  # Different configurations of layers
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],  # Regularization strength
        'learning_rate': ['constant', 'adaptive']
    }
    # find_param(mlp, param_grid, df)


    train_dataset=Customdataset(train_df)
    test_dataset=Customdataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    print('ournet')

    depths = [3,5,7,9]  # Example depths
    widths = [5,50, 100, 200,1000]  # Example widths


    # best_config, best_f1, model = find_param_for_net(train_df,test_df,weight1,depths, widths)

    # print("\n✅ Best Configuration Found:")
    # print(f"Depth: {best_config[0]}, Width: {best_config[1]}")
    # print(f"Best F1 Score: {best_f1:.4f}")
    # train_dataset = Customdataset(train_df)
    # test_dataset = Customdataset(test_df)
    # train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)
    # test_features = test_df[
    #     ['ALQ130', 'DBD900', 'DBD910', 'SMD650', 'PAD660', 'PAD675',
    #      'WHQ040', 'SLD012', 'OCQ180']].values
    # test_labels = test_df['KIQ022'].values
    #
    #
    model = ournet(9, 1, 7,50)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(weight1))
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(100):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.reshape((-1, 1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        

    model.eval()
    predict=torch.sigmoid(model(torch.tensor(test_features).float()).detach())
    predict=(predict>=0.5).float()
    f1score = metrics.f1_score(test_labels, predict)
    auroc = metrics.roc_auc_score(test_labels, predict)
    accuracy = metrics.accuracy_score(test_labels, predict)

    print(f"f1_socre: {f1score}, auroc: {auroc}, accuracy: {accuracy}")
    f1.append(f1score)
    m.append(model)
    model_names.append('Ournet')




    # f1_df = pd.DataFrame.from_dict(plot_data, orient='index')
    # f1_df.reset_index(inplace=True)
    # f1_df.rename(columns={'index': 'Race'}, inplace=True)
    # f1_df = f1_df[['Race'] + [col for col in f1_df.columns if col != 'Race']]
    # print(f1_df)
    # f1_df.to_csv(file_name, index=False)

    return m,f1, model_names





if __name__ == '__main__':
    df = get_data()
    category = ['Mexican American', 'Hispanic', 'White', 'Black', 'Asian']
    col_name = 'RIDRETH3'
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Comment out the original data distribution plot
    # plot_graphs([train_df, test_df])
    
    train_features, train_labels, test_features, test_labels = process_df(train_df, test_df)
    
    param_grid1 = {
        'rf_n_estimators': [50, 100, 200, 300],
        'rf_max_depth': [5, 10, 15, 20],
        'svc_C': [0.1, 1, 10, 100],
        'svc_gamma': [0.001, 0.01, 0.1, 1],
        'log_C': [0.01, 0.1, 1, 10, 100]
    }

    # Train models
    combined_models, combined_f1_scores, model_names = train_big_model(train_df, test_df)

    # Hyperparameter grids (not used directly here)
    param_grid3 = {
        'knn_n_neighbors': [11, 15, 20, 30, 40],
        'knn_weights': ['uniform', 'distance'],
        'knn_metric': ['euclidean', 'manhattan'],
        'ab_n_estimators': [50, 100, 150, 200],
        'ab_learning_rate': [0.01, 0.1, 0.5, 1],
        'log_C': [0.01, 0.1, 1, 10, 100],
    }

    # =========================
    # New: Plot model performance comparison
    # =========================
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    accuracy = []
    auroc = []

    for i, model in enumerate(combined_models):
        if hasattr(model, 'predict'):
            preds = model.predict(test_features)
        else:
            preds = torch.sigmoid(model(torch.tensor(test_features).float())).detach().numpy()
            preds = (preds >= 0.5).astype(float)
        preds = preds.reshape(-1)
        accuracy.append(metrics.accuracy_score(test_labels, preds))
        try:
            auroc.append(metrics.roc_auc_score(test_labels, preds))
        except:
            auroc.append(0.5)  # Default AUROC if error occurs

    if not os.path.exists('plots_model'):
        os.makedirs('plots_model')

    sns.set(style="whitegrid")
    metrics_to_plot = {
        'F1 Score': combined_f1_scores,
        'Accuracy': accuracy,
        'AUROC': auroc
    }

    for metric_name, metric_values in metrics_to_plot.items():
        plt.figure(figsize=(10,6))
        ax = sns.barplot(x=model_names, y=metric_values, palette='muted')
        plt.title(f'Model Comparison - {metric_name}', fontsize=16)
        plt.ylabel(metric_name, fontsize=14)
        plt.xlabel('Models', fontsize=14)
        plt.xticks(rotation=45)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=5, fontsize=12)
        plt.tight_layout()
        plt.savefig(f'plots_model/{metric_name.replace(" ", "_")}_comparison.png')
        plt.close()
