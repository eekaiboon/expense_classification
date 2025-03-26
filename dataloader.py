import torch
from torch.utils.data import Dataset

class TransactionsDataset(Dataset):
    def __init__(self, df, tokenizer, label_col="label_id", max_length=64):
        self.df = df
        self.tokenizer = tokenizer
        self.label_col = label_col
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1) Extract main text description
        description = str(row["description"])

        # 2) Extract original_category and append it to the text
        #    (If your DataFrame might not have original_category, handle that carefully)
        original_category = str(row["original_category"])
        combined_text = description + " [SEP] " + original_category
        
        # 3) Read the correct label column
        label = int(row[self.label_col])

        # 4) Numeric feature (amount)
        amount_val = float(row["amount"])

        # 5) Encode the combined text
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )

        # 6) Convert tokenizer output to torch tensors
        item = {k: torch.tensor(v) for k, v in encoding.items()}

        # 7) Add label and amount to the item dict
        item["labels"] = torch.tensor(label, dtype=torch.long)
        item["amount"] = torch.tensor([amount_val], dtype=torch.float)

        return item
