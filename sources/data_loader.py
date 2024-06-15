import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder


class CustomDataset(Dataset):
    def __init__(self, config):
        data_path = config["data_path"]
        user_col = config["user_col"]
        label_col = config["label_col"]
        data = pd.read_csv(data_path)
        self.data = data_preprocessing(data, user_col, label_col)
        self.X = self.data.drop(columns=[user_col, label_col]).values
        self.y = self.data[label_col].values
        print(f"Setup Data: {self.X.shape[0]} data, {self.X.shape[1]} features")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return X, y


def get_loader(config):
    dataset = CustomDataset(config)

    train_idx, valid_idx = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        stratify=dataset.y,
        random_state=config["seed"],
    )

    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config["batch_size"], shuffle=False
    )

    return train_loader, valid_loader


def data_preprocessing(data, user_col, label_col):
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == "object" and column not in [label_col, label_col]:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le
    return data
