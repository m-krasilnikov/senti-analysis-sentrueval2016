from text_preprocessing_utils.data_processing import process_data
from models.bert_classifier.bert_model import BertClassifier

bert_config_path = "models/bert_classifier/bert_config.yaml"

if __name__ == "__main__":
    train_data, test_data = process_data('config.yaml')
    model = BertClassifier(test_data, bert_config_path)
    preds = model.predict()
