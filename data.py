import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# class Customdataset(Dataset):
#     def __init__(self, dataframe):
#         self.dataframe = dataframe
#
#     def __len__(self):
#         return len(self.dataframe)
#
#     def __getitem__(self, idx):
#         sample = self.dataframe.iloc[idx]
#         features = torch.tensor(sample[['ALQ130','DBD900', 'DBD910','SMD650','PAD660','PAD675','WHQ040','SLD012','OCQ180']].values).float()
#         label =  torch.tensor(sample['DIQ010']).float()
#         return features, label
#

class Customdataset(Dataset):
    def __init__(self, dataframe, device='cpu'):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing all the data.
            feature_columns (list of str): List of column names to be used as features.
            label_column (str): Column name for the label.
            device (str, optional): Device to store tensors ('cpu' or 'cuda'). Default is 'cpu'.
        """
        feature_columns = ['ALQ130', 'DBD900', 'DBD910', 'SMD650', 'PAD660',
                           'PAD675', 'WHQ040', 'SLD012', 'OCQ180']
        label_column = 'DIQ010'
        self.features = torch.tensor(dataframe[feature_columns].values, dtype=torch.float32).to(device)
        self.labels = torch.tensor(dataframe[label_column].values, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Get features and label by index
        features = self.features[idx]
        label = self.labels[idx]
        return features, label



def data_collection(filenames,columns):
    #merge the needed columns in the dataframes

    dataframes=[]

    for i in range(len(columns)):
        name = filenames[i]
        c = columns[i]
        dataframes.append(pd.read_sas(name)[c])

    df_all=dataframes[0]
    for df in dataframes[1:]:
        df_all=pd.merge(df_all, df, on='SEQN', how='inner')
    return df_all

def data_collection_merge(filenames,columns):
    #if we need more than one years of data, use this
    df_all=None
    for filename in filenames:
        if df_all==None:
            df_all=data_collection(filename,columns)
        else:
            df_all=pd.concat([df_all, data_collection(filename,columns)], ignore_index=True)
    return df_all

def categorize_income(row):
    if row['INDFMPIR'] < 1.00:
        return 0  # Poor
    elif row['INDFMPIR'] < 5.00:
        return 1  # Medium
    else:
        return 2  # Rich

def data_cleaning(df,columns):
    for key in columns.keys():
        old_value,new_value=columns[key]
        df[key]=df[key].replace(old_value, new_value)

    if 'INDFMPIR' in df.columns:
        df['INDFMPIR'] = df.apply(categorize_income, axis=1)

    return df

def get_data():
    filenames_years = [
         # [['DEMO_2013.XPT', 'DIQ_2013.XPT', 'ALQ_2013.XPT', 'DBQ_2013.XPT','SMQ_2013.XPT','PAQ_2013.XPT','WHQ_2013.XPT', 'SLQ_2013.XPT', 'OCQ_2013.XPT']],
        [['DEMO_2015.XPT', 'DIQ_2015.XPT', 'ALQ_2015.XPT', 'DBQ_2015.XPT', 'SMQ_2015.XPT', 'PAQ_2015.XPT','WHQ_2015.XPT', 'SLQ_2015.XPT', 'OCQ_2015.XPT']],
        [['P_DEMO.XPT', 'P_DIQ.XPT' ,'P_ALQ.XPT', 'P_DBQ.XPT', 'P_SMQ.XPT','P_PAQ.XPT','P_WHQ.XPT','P_SLQ.XPT','P_OCQ.XPT']]
    ]

    columns_to_append = [
        ['SEQN', 'RIDRETH3','RIDAGEYR', 'INDFMPIR'], #race,age,income to ratio
        ['SEQN', 'DIQ010'], #diabete label
        ['SEQN', 'ALQ130'], #Avg # alcoholic drinks/day - past 12 mos
        ['SEQN', 'DBD900', 'DBD910'], ## of meals from fast food or pizza place,# of frozen meals/pizza in past 30 days
        ['SEQN', 'SMD650'], #Avg # cigarettes/day during past 30 days
        ['SEQN','PAD660','PAD675'], #Minutes vigorous recreational activities, Minutes moderate recreational activities
        ['SEQN','WHQ040'], #Like to weigh more, less or same
        ['SEQN','SLD012'], #Sleep hours - weekdays or workdays
        ['SEQN','OCQ180'], #Hours worked last week in total all jobs
    ]

    columns_replace = {'RIDRETH3': ([7, np.nan], [np.nan, np.nan]),
                       'DIQ010': ([2, 3, 7, 9, np.nan], [0.0, 1.0, np.nan, np.nan, np.nan]),
                       'ALQ130': ([777, 999, np.nan], [np.nan, np.nan, 0.0]),
                       'DBD900': ([5555,7777,9999,np.nan], [25.0, np.nan, np.nan,np.nan]),
                       #'DBD905': ([6666,7777,9999,np.nan],[100.0,np.nan,np.nan,np.nan]),
                       'DBD910': ([6666, 7777, 9999, np.nan],[90.0, np.nan, np.nan, np.nan]),
                       'SMD650': ([777, 999, np.nan], [np.nan, np.nan, 0.0]),
                       'PAD660':([7777,9999,np.nan],[np.nan,np.nan,0.0]),
                       'PAD675':([7777,9999,np.nan],[np.nan,np.nan,0.0]),
                       'WHQ040':([7,9,np.nan],[np.nan,np.nan,np.nan]),
                       'SLD012':([np.nan],[np.nan]),
                       'OCQ180':([7777,9999,np.nan],[np.nan,np.nan,0.0]),
                       }

    dataframes = []
    for filenames in filenames_years:
        df = data_collection_merge(filenames, columns_to_append)
        df = data_cleaning(df, columns_replace)
        df = df.dropna()
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    df = df.drop('SEQN', axis=1)
    return df

if __name__ == '__main__':


    df = get_data()
    print(df)

    # for c in df.columns:
    #     print(df[c].value_counts())
