import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import os

# ===== 1. Load Dataset =====
def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

train_data = load_dataset("data/train.json")
eval_data = load_dataset("data/eval.json")

# ===== 2. Prepare Tokenizer & Model =====
MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ===== 3. Tokenization =====
def tokenize_dataset(dataset):
    inputs = [f"Generate step-by-step instructions for: {d['instruction']}" for d in dataset]
    targets = ["\n".join(d["output"]) for d in dataset]

    encodings = tokenizer(
        inputs,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    labels = tokenizer(
        targets,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    encodings["labels"] = labels["input_ids"]
    return encodings

train_encodings = tokenize_dataset(train_data)
eval_encodings = tokenize_dataset(eval_data)

# ===== 4. DataLoader =====
train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    train_encodings["labels"]
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# ===== 5. Optimizer =====
optimizer = AdamW(model.parameters(), lr=5e-5)

# ===== 6. Training Loop =====
epochs = 100
model.train()

for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

# ===== 7. Save Model =====
os.makedirs("models/final_model", exist_ok=True)
model.save_pretrained("models/final_model")
tokenizer.save_pretrained("models/final_model")
print("✅ Model saved at models/final_model")
