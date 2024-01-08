from transformers import AutoModelForSequenceClassification, AutoTokenizer


def create_model():
    # Initialize the cross-encoder model
    cross_encoder_model_name = 'nboost/pt-bert-base-uncased-msmarco'
    cross_encoder = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model_name)
    tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model_name)
    return cross_encoder, tokenizer