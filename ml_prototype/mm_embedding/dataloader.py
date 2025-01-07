"""
This file contains the implementation of a PyTorch Dataset and DataLoader for tabular data stored in Parquet files.

Run this script with the command:
    python -m ml_prototype.mm_embedding.dataloader --config ml_prototype/mm_embedding/config.yaml
"""
import os
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from jsonargparse import ArgumentParser, ActionConfigFile
from ml_prototype.mm_embedding.config import MyConfig, DataModuleConfig
from tqdm import tqdm
from ml_prototype.mm_embedding.util import Logger
from typing import Optional
from torch import Tensor


class TabularDataset(Dataset):
    def __init__(self, folder_path: str, columns: Optional[list[str]] = None):
        """
        Dataset that loads tabular data from Parquet files into a dictionary keyed by feature names.
        """
        self.logger = Logger()
        self.folder_path = folder_path
        self.columns = columns
        self.data = self.load_data()

    def load_data(self) -> dict[str, pd.Series]:
        """
        Load data from Parquet files, returning a dictionary where each key is a column name
        and each value is a pandas Series for that column.

        Returns:
            dict[str, pd.Series]: Dictionary of feature names to pandas Series.
        """
        dataframes = []
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith(".parquet"):
                file_path = os.path.join(self.folder_path, file_name)
                table = pq.read_table(file_path)
                df = table.to_pandas()
                if self.columns:
                    # Normalize the data per column
                    for col in self.columns:
                        # Make sure 'col.name' is actually in the dataframe columns
                        if col.name not in df.columns:
                            continue  # or raise an error, depending on your needs

                        # Optional: Check if it's numeric
                        if not pd.api.types.is_numeric_dtype(df[col.name]):
                            self.logger.error(f"Warning: {col.name} is not numeric; skipping.")
                            # Remove or skip this column from df
                            df.drop(columns=[col.name], inplace=True)
                            continue

                        # Convert to float32 just in case
                        df[col.name] = df[col.name].astype("float32")

                        # If we don't have valid min_val or max_val, just keep it as float
                        if col.min_val is None or col.max_val is None:
                            continue  # Already converted to float32; no further normalization.

                        # Convert to float in case min_val/max_val are not numeric types
                        min_val = float(col.min_val)
                        max_val = float(col.max_val)

                        if min_val == max_val:
                            # Avoid dividing by zero; consider whether you should set these to 0 or do nothing
                            # e.g. if all data is the same value, you can store it as 0.0
                            df[col.name] = 0.0
                        else:
                            # Perform min–max scaling
                            df[col.name] = (df[col.name] - min_val) / (max_val - min_val)
                    # Drop columns not in self.columns
                    df = df[[col.name for col in self.columns]]
                dataframes.append(df)

        concatenated_df = pd.concat(dataframes, ignore_index=True)
        feature_dict = {col: concatenated_df[col] for col in concatenated_df.columns}
        return feature_dict

    def __len__(self) -> int:
        """Return the number of rows in the dataset.
        Returns:
            int: Number of rows.
        """
        first_key = next(iter(self.data.keys()))
        return len(self.data[first_key])

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return a dictionary keyed by feature name with tensors of shape [1, 1].
        Args:
            idx (int): Index of the row to retrieve.

        Returns:
            Dict[str, Tensor]: Dictionary of feature tensors with shape [1, 1].
        """
        return {key: torch.tensor([[value.iloc[idx]]], dtype=torch.float32) for key, value in self.data.items()}


def create_dataloaders(data_module_config: DataModuleConfig) -> dict[str, DataLoader]:
    """
    Create DataLoaders for the TabularDataset, split into training and validation sets.

    Args:
        folder_path (str): Path to the folder containing Parquet files.
        columns (list[str]): List of columns to load.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        drop_last (bool): Whether to drop the last incomplete batch.
        val_split (float): Proportion of the dataset to use for validation.

    Returns:
        dict[str, DataLoader]: Dictionary containing 'train' and 'val' DataLoaders.
    """
    folder_path = os.path.expanduser(data_module_config.folder_path)
    columns = getattr(data_module_config, 'columns', None)
    batch_size = data_module_config.batch_size
    shuffle = getattr(data_module_config, 'shuffle', True)
    val_ratio = getattr(data_module_config, 'val_ratio', 0.1)
    drop_last = True

    dataset = TabularDataset(folder_path, columns)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, drop_last=drop_last
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=drop_last
    )

    return {"train": train_loader, "val": val_loader}


def collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """
    Custom collate function to combine a batch of dictionaries into a single dictionary
    with tensors of shape [batch_size, 1] for each feature.

    Args:
        batch (List[Dict[str, Tensor]]): List of dictionaries with feature tensors.

    Returns:
        Dict[str, Tensor]: Dictionary of tensors with shape [batch_size, 1].
    """
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = torch.cat([sample[key] for sample in batch], dim=0)

    return batch_dict


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', action=ActionConfigFile, help='Path to YAML config file')
    parser.add_class_arguments(MyConfig, 'my_config')
    args = parser.parse_args()

    # Create DataLoaders
    data_module_config = parser.instantiate_classes(args).my_config.data_module
    dataloaders = create_dataloaders(data_module_config)

    for idx, batch in enumerate(tqdm(dataloaders["train"])):
        print(batch)
        if idx > 10:
            break


if __name__ == "__main__":
    main()
