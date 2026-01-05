import torch
from datasets import load_dataset
from tqdm import tqdm

from dataset import get_dataLoaders
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 6
BATCH_SIZE = 32 if torch.cuda.is_available() else 16
CHECKPOINT_PATH = "./bert_emotion_epoch_5.pt"



# load dataset
data = load_dataset("emotion")
_, _, test_dl = get_dataLoaders(data, BATCH_SIZE)

# load model
model = get_model(num_labels=NUM_LABELS, model_name=MODEL_NAME)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

print(checkpoint)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()

total_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for batch in tqdm(test_dl, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

avg_loss = total_loss / len(test_dl)
accuracy = correct / total

print(f"Test Loss: {avg_loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")