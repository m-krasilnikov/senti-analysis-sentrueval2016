import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from .dataset import *
from .model import *
from .trainer import Trainer
import re
import yaml

print("Program is started")
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"The {device} is used for model inference.")


MAX_LEN = 150
BATCH_SIZE = 16
print(f"MaxLen is {MAX_LEN} and batch size is {BATCH_SIZE} ")


# Fast preprocessing to shortcut main preprocessing step in main.
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Удаляем ссылки
    text = re.sub(r'[^\w\s]', '', text)  # Удаляем специальные символы
    text = text.lower()  # Приводим к нижнему регистру
    return text


class BertClassifier:
    def __init__(self, test_data, bert_config_path):
        self.test_data = test_data
        self.config = self._load_config(bert_config_path)

    def _preprocess_data(self):
        print("Data preprocession is started.")
        self.test_data['text'] = self.test_data['text'].apply(lambda x: clean_text(x))
        le = LabelEncoder()
        self.test_data['label'] = le.fit_transform(self.test_data['label'])

    def _model_init(self):
        try:
            file_path = os.path.join('models/bert_classifier', self.config["bert_checkpoint_path"])
            with open(file_path, 'r') as f:
                print("File with checkpoint is founded.")
                model = Trainer.load(file_path)
        except FileNotFoundError:
            print("File with checkpoint is not founded.")

        return model

    def _inference(self, model):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['model'], truncation=True, do_lower_case=True)
        test_dataset = FiveDataset(self.test_data, tokenizer, MAX_LEN)
        test_params = {"batch_size": BATCH_SIZE,
                       "shuffle": False,
                       "num_workers": 0
                       }
        test_dataloader = DataLoader(test_dataset, **test_params)
        predictions = model.predict(test_dataloader)
        return predictions

    @staticmethod
    def _load_config(bert_config_path):
        with open(bert_config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def predict(self):
        self._preprocess_data()
        model = self._model_init()
        preds = self._inference(model)
        return preds
