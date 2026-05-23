import torch
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
from dataset import get_dataLoaders
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 6
BATCH_SIZE = 32 if torch.cuda.is_available() else 16
CHECKPOINT_PATH = "./best_model.pt"
id2label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# load dataset
data = load_dataset("emotion")
_, _, test_dl = get_dataLoaders(data, BATCH_SIZE)

# load model
model = get_model(num_labels=NUM_LABELS, model_name=MODEL_NAME)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

print(f"Loaded checkpoint from: {CHECKPOINT_PATH}")
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()

def evaluate(model, dataloader, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / total,
        "preds": all_preds,
        "labels": all_labels
    }


metrics = evaluate(model, test_dl, device)
macro_f1 = f1_score(
    metrics["labels"],
    metrics["preds"],
    average="macro"
)

print(f"Test Loss: {metrics['loss']:.4f}")
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        metrics["labels"],
        metrics["preds"],
        target_names=list(id2label.values()),
        digits=4
    )
)