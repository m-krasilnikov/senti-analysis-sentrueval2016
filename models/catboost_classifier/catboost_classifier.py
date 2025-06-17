import os
import pandas as pd
import numpy as np
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def make_features_targets(data: pd.DataFrame):
    return data['text'], data['labels']

def describe_data(data):
    pass



train_data_tkk = pd.read_csv('../../data/input/preprocessed_tkk_train.csv')
test_data_tkk = pd.read_csv('../../data/input/preprocessed_tkk_test.csv')
print(f"Number of rows and columns in the train data set: {train_data_tkk.shape}")
print(f"Number of rows and columns in the test data set: {test_data_tkk.shape}")
test_data_tkk.head()

train_data_tkk[train_data_tkk['text'].str.contains("Уникальный продукт &amp")==True]
train_data_tkk.groupby('label').count()

X_train = train_data_tkk.text
y_train = train_data_tkk.label

X_test = test_data_tkk.text
y_test = test_data_tkk.label


model = CatBoostClassifier(
    iterations=150,
    depth=3,
    learning_rate=0.05,
    l2_leaf_reg=3,
    min_data_in_leaf=3,
    random_seed=42
)

model.fit(
    X_train,
    y_train,
    text_features=[0],
    verbose=True
)


dataset_test = Pool(
    data=X_test,
    text_features=[0]
)
predict_classes = model.predict(dataset_test)
preds = predict_classes

print(f"The f1- micro score equals: { f1_score(y_test, preds, average='micro')}")
print(f"The f1- micro score equals: { f1_score(y_test, preds, average='macro')}")
print(f"The f1- weighted score equals: {f1_score(y_test, preds, average='weighted')}")

print(classification_report(y_test, preds))
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)
disp.plot()
plt.show()