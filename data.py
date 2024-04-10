import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class Customdataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        features = torch.tensor(sample[['ALQ121','DBQ700', 'DBD910','SMQ040']].values).float()
        label =  torch.tensor(sample['DIQ010']).float()
        return features, label

def data_collection(filenames,columns):
    #merge the needed columns in the dataframes
    dataframes=[]
    for name,c in zip(filenames,columns):
        dataframes.append(pd.read_sas(name)[c])

    df_all=dataframes[0]
    for df in dataframes[1:]:
        df_all=pd.merge(df_all, df, on='SEQN', how='outer')
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

def data_cleaning(df,columns):
    for key in columns.keys():
        old_value,new_value=columns[key]
        df[key]=df[key].replace(old_value, new_value)
    return df

if __name__ == '__main__':
    filenames=[['P_DEMO.XPT','P_DIQ.XPT','P_ALQ.XPT','P_DBQ.XPT','P_SMQ.XPT']]
    columns_to_append=[['SEQN','RIDRETH3'],['SEQN','DIQ010'],['SEQN','ALQ121'],['SEQN', 'DBQ700', 'DBD910'],['SEQN','SMQ040']]
    columns_replace={'RIDRETH3':([7,'.'],[np.nan,np.nan]),
                     'DIQ010':([2,3,7,9,'.'],[0,1,np.nan,np.nan,np.nan]),
                     'ALQ121':([77,99,'.'],[np.nan,np.nan,0]),
                     'DBQ700':([7,9,'.'],[np.nan,np.nan,np.nan]),
                     'DBD910':([6666,7777,9999,'.'],[90,np.nan,np.nan,np.nan]),
                     'SMQ040':([7,9,'.'],[np.nan,np.nan,3])}
    df=data_collection_merge(filenames,columns_to_append)
    df=data_cleaning(df,columns_replace)
    df=df.dropna()
    print(df['DIQ010'].value_counts())
    print(df['ALQ121'].value_counts())
    print(df['DBQ700'].value_counts())
    print(df['DBD910'].value_counts())
    print(df['SMQ040'].value_counts())
