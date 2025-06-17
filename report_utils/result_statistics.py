from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from typing import List
import torch
from collections import Counter

#TODO Add methods for stats calc, and visualization
class StatDescription:
    def __init__(self, preds: List, ground_true_lst: List):
        self.preds = preds
        self.ground_true_lst = ground_true_lst

if __name__ == "__main__":
    a = [1, 2, 3]
    cnt = Counter(a)
    ans = (max(a) == len(a) and max(cnt.values()) == 1)
    print(ans)


