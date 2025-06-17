import os

import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from dataset import *
from model import *
from trainer import Trainer
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def _to_lower_case(text: str):
    return text.lower()

torch.manual_seed(42)

PATH = r"models/bert_classifier/"
MAX_LEN = 200
BATCH_SIZE = 64

train_data = pd.read_csv('../../data/input/preprocessed_tkk_train.csv')
test_data = pd.read_csv('../../data/input/preprocessed_tkk_test.csv')
#train_data = pd.read_csv(os.path.join(PATH, "train.csv"))
#test_data = pd.read_csv(os.path.join(PATH, "test.csv"))
for k, v in data_refs.items():
    print(f"Statistics for {k} dataset.")
    v['text'] = v['text'].apply(lambda x: _to_lower_case(x))

print(train_data.head())
train_data.head()

le = LabelEncoder()

train_data.label = le.fit_transform(train_data.label)
train_data.head()

train_split, val_split = train_test_split(train_data, test_size=0.85, random_state=42)

tokenizer = AutoTokenizer.from_pretrained(
    "r1char9/rubert-base-cased-russian-sentiment", truncation=True, do_lower_case=True)

train_dataset = FiveDataset(train_split, tokenizer, MAX_LEN)
val_dataset = FiveDataset(val_split, tokenizer, MAX_LEN)
test_dataset = FiveDataset(test_data, tokenizer, MAX_LEN)

train_params = {"batch_size": BATCH_SIZE,
                "shuffle": True,
                "num_workers": 0
                }

test_params = {"batch_size": BATCH_SIZE,
               "shuffle": False,
               "num_workers": 0
               }

train_dataloader = DataLoader(train_dataset, **train_params)
val_dataloader = DataLoader(val_dataset, **test_params)
test_dataloader = DataLoader(test_dataset, **test_params)

config = {
    "num_classes": 3,
    "dropout_rate": 0.1
}
model = ModelForClassification(
    "r1char9/rubert-base-cased-russian-sentiment",
    config=config
)

trainer_config = {
    "lr": 2e-5,
    "n_epochs": 5,
    "weight_decay": 1e-6,
    "batch_size": BATCH_SIZE,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
}
t = Trainer(trainer_config)

t.fit(
    model,
    train_dataloader,
    val_dataloader
)

t.save("baseline_model.ckpt")

t = Trainer.load("baseline_model.ckpt")
predictions = t.predict(test_dataloader)
ground_true = test_data['label'].tolist()
print("======Stats===========")
print(f1_score(ground_true, predictions, average='micro'))
print(f1_score(ground_true, predictions, average='macro'))
print(f1_score(ground_true, predictions, average='weighted'))
print(accuracy_score(ground_true, predictions))

print("======================")

# pred_labels = le.inverse_transform(preds)
cm = confusion_matrix(ground_true, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


# sample_submission = pd.read_csv(r"C:\Users\ThinkPad T15 Gen1\OneDrive\Desktop\senti-analysis-sentrueval2016\models\bert_classifier\sample_submission.csv")
# sample_submission["rate"] = predictions
# sample_submission.rate = le.inverse_transform(sample_submission.rate)
# sample_submission.head()
#
# sample_submission.to_csv("submission.csv", index=False)