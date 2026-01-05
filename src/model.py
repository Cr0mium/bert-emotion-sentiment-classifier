from transformers import BertForSequenceClassification

def get_model(num_labels=6, model_name="bert-base-uncased"):
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return model