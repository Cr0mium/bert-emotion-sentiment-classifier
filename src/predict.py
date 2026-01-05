import torch
from transformers import BertTokenizer
from model import get_model

# config
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 6
MAX_LENGTH = 128
CHECKPOINT_PATH = "./bert_emotion_epoch_5.pt"

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = get_model(num_labels=NUM_LABELS, model_name=MODEL_NAME)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model, tokenizer


def predict(text, model, tokenizer):
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()

    return LABELS[pred_id], confidence


if __name__ == "__main__":
    model, tokenizer = load_model()

    print("\n🧠 BERT Emotion Classifier")
    print("Type a sentence and press Enter")
    print("Type 'exit' or 'quit' to stop\n")

    while True:
        text = input(">> ")

        if text.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break

        if len(text.strip()) == 0:
            print("⚠️ Please enter some text.")
            continue

        label, confidence = predict(text, model, tokenizer)
        print(f"Prediction: {label} | Confidence: {confidence:.2f}\n")