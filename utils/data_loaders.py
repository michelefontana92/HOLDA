import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
from sklearn.model_selection import StratifiedShuffleSplit


class HOLDADataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        target = self.targets[item]
        return {
            'data': torch.tensor(data, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader_test(df):
    ds = HOLDADataset(data=df,
                      targets=df)

    return DataLoader(ds,
                      batch_size=len(df),
                      num_workers=2,
                      shuffle=False)


def create_data_loader(df, target_label, batch_size,
                       shuffle=True, num_workers=2):

    ds = HOLDADataset(data=df[[col for col in df.columns
                               if col not in [target_label]]].to_numpy(),
                      targets=df[target_label].to_numpy())

    return DataLoader(ds,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=shuffle)


def load_dataset(dataset_path, sample=-1):
    df = pd.read_csv(dataset_path)
    if sample > 0:
        df = df.sample(sample)
    dataset_constants = json.load(open('constants/dataset_constants.json',))
    numerical_attributes = dataset_constants['Numerical_Attributes']
    target_label = dataset_constants['Target_Label']
    todrop_label = dataset_constants['Labels_To_Drop']
    df = df.drop(todrop_label, axis=1)
    scaler = StandardScaler()
    df[numerical_attributes] = scaler.fit_transform(df[numerical_attributes])
    categorical_attributes = [col for col in df.columns
                              if col not in numerical_attributes +
                              [target_label]]
    scaler = MinMaxScaler()
    df[categorical_attributes] = scaler.fit_transform(
        df[categorical_attributes])
    return df


def preprocess_dataset(df, sample=-1):
    if sample > 0:
        df = df.sample(sample)
    dataset_constants = json.load(open('constants/dataset_constants.json',))
    numerical_attributes = dataset_constants['Numerical_Attributes']
    target_label = dataset_constants['Target_Label']
    todrop_label = dataset_constants['Labels_To_Drop']
    df = df.drop(todrop_label, axis=1)
    scaler = StandardScaler()
    df[numerical_attributes] = scaler.fit_transform(df[numerical_attributes])
    categorical_attributes = [col for col in df.columns
                              if col not in numerical_attributes +
                              [target_label]]
    scaler = MinMaxScaler()
    df[categorical_attributes] = scaler.fit_transform(
        df[categorical_attributes])
    return df


def read_dataset(df_path, encoded=False):
    df = pd.read_csv(df_path)
    num_attr = read_constants()['Numerical_Attributes']
    columns = [col for col in df.columns if col not in num_attr]
    if not encoded:
        cat_col_dict = {}
        for col in columns:
            cat_col_dict[col] = str
        df = pd.read_csv(df_path, dtype=cat_col_dict)
    else:
        df = pd.read_csv(df_path)
    return df


def stratified_shuffle_split(df, test_size, random_state=42, label='PRINC_SURG_PROC_CODE'):
    data = df[[col for col in df.columns if col != label]]
    target = df[label]

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state)

    for train_index, test_index in sss.split(data, target):
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]

    return df_train, df_test
