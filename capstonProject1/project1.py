import os
import re
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy
import mlflow
import mlflow.pytorch
import gradio as gr

warnings.filterwarnings("ignore")

# 1. MLFLOW WINDOWS CONFIGURATION
mlruns_dir = os.path.abspath('mlruns')
tracking_uri = f"file:{mlruns_dir}"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("ABSA_Sentiment_Analysis")

# Data Paths (Adjust if these vary on your system)
train_path = r"E:\capstone_project_1\data\train (2).csv"
test_path = r"E:\capstone_project_1\data\test (2).csv"


# --- Helper Functions ---
def normalize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = " ".join(text.split())
    return text


def tokenize(text):
    return text.split()


def build_vocab(df):
    token_2_id = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    corpus = df["review"].tolist()
    for text in corpus:
        norm_text = normalize(text)
        tokens = tokenize(norm_text)
        for t in tokens:
            if t not in token_2_id:
                token_2_id[t] = idx
                idx += 1
    return token_2_id


# --- Dataset & DataModule ---
label_map = {"negative": 0, "positive": 1, "neutral": 2}
inv_label_map = {0: "Negative 🔴", 1: "Positive 🟢", 2: "Neutral 🟡"}


class ABCDataset(Dataset):
    def __init__(self, df, token_2_id, label_map):
        self.df = df
        self.token_2_id = token_2_id
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        example = self.df.iloc[idx]
        text_aspact_pair = f"{example['review']} {example['aspect']}"
        tokens = tokenize(normalize(text_aspact_pair))
        input_ids = [self.token_2_id.get(t, self.token_2_id["<UNK>"]) for t in tokens]
        label_id = self.label_map.get(example["sentiment"], 2)  # Default to neutral if missing
        return {"input_ids": input_ids, "label_ids": label_id}


class ABSADataModule(pl.LightningDataModule):
    def __init__(self, train_path, test_path, batch_size=32):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        self.token_2_id = build_vocab(train_df)
        self.train_set = ABCDataset(train_df, self.token_2_id, label_map)
        self.test_set = ABCDataset(test_df, self.token_2_id, label_map)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        batch_input_ids = [item['input_ids'] for item in batch]
        batch_labels = [item['label_ids'] for item in batch]
        max_len = max(len(ids) for ids in batch_input_ids)
        pad_id = self.token_2_id["<PAD>"]
        padded_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in batch_input_ids]
        return {
            "batch_input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "batch_label": torch.tensor(batch_labels, dtype=torch.long)
        }


# --- Model Architectures ---
class ABSA(nn.Module):
    def __init__(self, vocab_size, num_labels=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        self.lstm = nn.LSTM(256, 512, batch_first=True)
        self.fc = nn.Linear(512, num_labels)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        return self.fc(h_n[-1])


class ABSAModel(pl.LightningModule):
    def __init__(self, vocab_size, num_labels=3):
        super().__init__()
        self.model = ABSA(vocab_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_labels)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["batch_input_ids"], batch["batch_label"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["batch_input_ids"], batch["batch_label"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=3e-4)


# --- Gradio Prediction Function ---
def predict_sentiment(review, aspect):
    # Ensure vocab exists (from DataModule)
    if not hasattr(dm, 'token_2_id'):
        return "Error: Vocabulary not loaded."

    # Preprocess text pair
    text_pair = f"{normalize(review)} {normalize(aspect)}"
    tokens = tokenize(text_pair)

    # Tokenize and create tensor
    input_ids = [dm.token_2_id.get(t, dm.token_2_id["<UNK>"]) for t in tokens]
    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        prediction = torch.argmax(logits, dim=1).item()
        probs = torch.softmax(logits, dim=1).squeeze()

    # Return friendly result with confidence
    result = inv_label_map[prediction]
    confidence = probs[prediction].item()
    return f"{result} ({confidence * 100:.1f}% confidence)"


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Initialize Data and Model
    dm = ABSADataModule(train_path, test_path)
    dm.setup()
    model = ABSAModel(vocab_size=len(dm.token_2_id))

    # 2. Train with MLflow Tracking
    mlf_logger = pl.loggers.MLFlowLogger(experiment_name="ABSA_Sentiment_Analysis", tracking_uri=tracking_uri)

    with mlflow.start_run() as run:
        mlflow.log_params({"lr": 3e-4, "batch_size": 32, "epochs": 3})

        trainer = pl.Trainer(max_epochs=3, logger=mlf_logger)
        trainer.fit(model, dm)

        # Log Metrics and Model to MLflow
        trainer.test(model, dm)
        mlflow.pytorch.log_model(model.model, "absa_lstm_v1")

        # --- ADDED FOR HUGGING FACE PREPARATION ---
        print("Saving model weights and vocabulary for Hugging Face...")

        # Save only the inner LSTM model weights (not the whole Lightning wrapper)
        torch.save(model.model.state_dict(), "model_weights.pth")

        # Save the vocabulary as a JSON file
        import json

        with open("vocab.json", "w") as f:
            json.dump(dm.token_2_id, f)

        print("Files saved: model_weights.pth and vocab.json")
        # ------------------------------------------

        print("\n" + "=" * 50)
        print(f"TRACKING COMPLETE. RUN ID: {run.info.run_id}")
        print(f'UI CMD: mlflow ui --backend-store-uri "{tracking_uri}"')
        print("=" * 50)

    # 3. Launch Gradio Interface
    interface = gr.Interface(
        fn=predict_sentiment,
        inputs=[
            gr.Textbox(lines=2, label="Review Text", placeholder="e.g., The battery is great but the screen is dim."),
            gr.Textbox(label="Aspect", placeholder="e.g., battery")
        ],
        outputs=gr.Textbox(label="Analysis Result"),
        title="Aspect-Based Sentiment Analyzer",
        description="Provide a review and a specific aspect (part of the product) to analyze.",
        examples=[
            ["The food was delicious, but the service was terrible.", "food"],
            ["The food was delicious, but the service was terrible.", "service"],
            ["Overall a solid phone, though the camera is just okay.", "camera"]
        ]
    )

    interface.launch(share=True)
# Run this locally once training is done
