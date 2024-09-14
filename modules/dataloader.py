import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

class DataLoaderBuilder:
    def __init__(self, emb_path, tag_emb_path, batch_size):
        self.emb_path = emb_path
        self.tag_emb_path = tag_emb_path
        self.batch_size = batch_size

    def load_data(self):
        item_embeddings = np.load(self.emb_path).astype(np.float32)
        tag_embeddings = np.load(self.tag_emb_path).astype(np.float32)

        # Remove zero rows
        zero_rows = np.all(item_embeddings == 0, axis=1)
        item_embeddings_cleaned = item_embeddings[~zero_rows]
        tag_embeddings_cleaned = tag_embeddings[~zero_rows]

        print(f'Cleaned item data: {item_embeddings_cleaned.shape}, Cleaned tag data: {tag_embeddings_cleaned.shape}')

        train_items, temp_items, train_tags, temp_tags = train_test_split(
            item_embeddings_cleaned, tag_embeddings_cleaned, test_size=0.2, random_state=42)
        valid_items, test_items, valid_tags, test_tags = train_test_split(
            temp_items, temp_tags, test_size=0.5, random_state=42)

        return train_items, valid_items, test_items, train_tags, valid_tags, test_tags

    def prepare_dataloaders(self, train_items, valid_items, test_items, train_tags, valid_tags, test_tags):
        train_dataset = TensorDataset(torch.tensor(train_items, dtype=torch.float32), torch.tensor(train_tags, dtype=torch.float32))
        valid_dataset = TensorDataset(torch.tensor(valid_items, dtype=torch.float32), torch.tensor(valid_tags, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(test_items, dtype=torch.float32), torch.tensor(test_tags, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader
