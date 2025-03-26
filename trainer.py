import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dataloader import TransactionsDataset
from model import BertWithNumeric
import torch.nn as nn

def data_collator(features):
    input_ids = torch.stack([f["input_ids"] for f in features])
    attention_mask = torch.stack([f["attention_mask"] for f in features])
    labels = torch.stack([f["labels"] for f in features])
    amount = torch.stack([f["amount"] for f in features])  # shape (batch_size, 1)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "amount": amount,
    }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    # Use weighted averaging to account for class imbalance
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Custom Trainer that uses weighted CrossEntropyLoss
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Remove unwanted keyword argument
        kwargs.pop("num_items_in_batch", None)
        labels = inputs.get("labels")
        outputs = model(**inputs, **kwargs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(self.args.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def main():
    # 1) Load one data file from the data/ folder
    data_path = os.path.join("data", "data.csv")  # CSV with columns: description, amount, label, etc.
    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded {len(df)} rows from '{data_path}'")

    # 2) Split into train & validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"[INFO] Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # 3) Label encode string labels -> integer IDs (fit on training labels only)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["label"])
    train_df["label_id"] = label_encoder.transform(train_df["label"])
    val_df["label_id"] = label_encoder.transform(val_df["label"])
    num_labels = len(label_encoder.classes_)
    print(f"[INFO] Found {num_labels} labels: {list(label_encoder.classes_)}")

    # 4) Scale the 'amount' feature (fit on train only, then transform both)
    scaler = StandardScaler()
    scaler.fit(train_df[["amount"]].values)   # Fit using a NumPy array
    train_df["amount"] = scaler.transform(train_df[["amount"]].values)
    val_df["amount"]   = scaler.transform(val_df[["amount"]].values)
    print("[INFO] Scaled the 'amount' column using StandardScaler.")

    # 5) Save the scaler for later usage
    os.makedirs("output_final", exist_ok=True)
    scaler_path = os.path.join("output_final", "amount_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Saved scaler to '{scaler_path}'")

    # 6) Compute class weights based on training label frequencies
    label_counts = Counter(train_df["label_id"])
    total_samples = len(train_df)
    class_weights = []
    for i in range(num_labels):
        count_i = label_counts[i]
        weight = total_samples / (num_labels * count_i)  # inverse proportional weight
        class_weights.append(weight)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"[INFO] Computed class weights: {class_weights}")

    # 7) Initialize tokenizer
    pretrained_model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    # 8) Build datasets (using 'label_id' for labels)
    train_dataset = TransactionsDataset(train_df, tokenizer, label_col="label_id")
    val_dataset = TransactionsDataset(val_df, tokenizer, label_col="label_id")

    # 9) Define model config with # of labels
    config = AutoConfig.from_pretrained(pretrained_model_name, num_labels=num_labels)

    # 10) Load custom BERT-based model with numeric feature
    model = BertWithNumeric.from_pretrained(pretrained_model_name, config=config)

    # 11) Training arguments
    training_args = TrainingArguments(
        output_dir="output",
        eval_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch"
    )

    # 12) Initialize our WeightedTrainer instead of the default Trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 13) Train
    print("[INFO] Starting training...")
    trainer.train()
    print("[INFO] Training complete.")

    # 14) Evaluate
    metrics = trainer.evaluate()
    print("[INFO] Validation metrics:", metrics)

    # 15) Save final model
    print("[INFO] Saving model to 'output_final'...")
    trainer.save_model("output_final")

    # 16) Save the tokenizer so inference can load from 'output_final'
    tokenizer.save_pretrained("output_final")

    # 17) Save the label encoder for decoding predictions later
    label_encoder_path = os.path.join("output_final", "label_encoder.pkl")
    joblib.dump(label_encoder, label_encoder_path)
    print("[INFO] All done! Model, tokenizer, scaler, and label encoder saved in 'output_final'")

if __name__ == "__main__":
    main()
