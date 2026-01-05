import torch
from datasets import load_dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import get_dataLoaders
from model import get_model

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 6
EPOCHS = 5
LR = 2e-5
CHECKPOINT_SAVE_PATH= ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32 if torch.cuda.is_available() else 16

#dataloaders
data = load_dataset("emotion")
train_dl, val_dl, test_dl = get_dataLoaders(data, BATCH_SIZE)


model = get_model(num_labels=NUM_LABELS, model_name=MODEL_NAME)
model.to(device)
optimizer = AdamW(model.parameters(), lr=LR)


# 10% warmup
total_steps = len(train_dl) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

#train
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0

    progress_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # reset gradients
        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()

        # clip gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / len(train_dl)

    #validation
    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0

    #disable gradients
    with torch.no_grad():
        for batch in val_dl:
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

            total_val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = total_val_loss / len(val_dl)
    val_accuracy = correct / total

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    #save weights
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        },
        f"./bert_emotion_epoch_{epoch+1}.pt"
    )

print("Training complete")