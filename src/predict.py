import torch
from transformers import BertTokenizer
from model import get_model

MODEL_NAME = "bert-base-uncased"
CHECKPOINT_PATH = "./bert_emotion_epoch_5.pt"
LABELS = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]
MAX_LEN = 128

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer= BertTokenizer.from_pretrained(MODEL_NAME)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model = get_model(num_labels=len(LABELS),model_name=MODEL_NAME)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

model.eval()

def predict(text: str):
    """
    Predict emotion label for input text
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )

    inputs = {
        'input_ids':inputs['input_ids'],
        'attention_mask':inputs['attention_mask']
    }

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    return LABELS[prediction]


if __name__ == "__main__":
    sample_texts = [
        "I am feeling very happy today!",
        "This is the worst day of my life",
        "I'm scared about the results",
        "I love spending time with my family"
    ]


if __name__ == "__main__":
    sample_texts = [
        "I am feeling very happy today!",
        "This is the worst day of my life",
        "I'm scared about the results",
        "I love spending time with my family"
    ]

    for text in sample_texts:
        print(f"Text: {text}")
        print(f"Prediction: {predict(text)}\n")


