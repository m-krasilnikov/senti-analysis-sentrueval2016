# Combination of FRIDA and Catboost for sentrueval2016 dataset.

This project deals with sentiment analysis nlp sub task. 
In particular, the SentiRuEval2016 dataset has been selected 
as the object of study. 
The process of dataset acquisition and text preprocessing 
is described. Previous approaches based on BERT models are
reviewed. As a novel approach, we trained a combination of 
the FRIDA embedder and the CatBoost classifier. 
The results demonstrate that this method yields higher $F_1$ scores 
for both negative and positive classes, surpassing earlier 
models based on BERT architecture. 

The project 
```
├───data
│   ├───input
│   └───output
├───models
│   ├───bert_classifier
│   ├───catboost_classifier
│   └───frida_embedder
├───report_utils
├───text_preprocessing_utils
├─main.py
├─config.yaml
```
- `models` — Contains various models and text scripts for training, specifically for text sentiment classification, including BERT, CatBoost with an embedded encoder, and a combination of FRIDA and CatBoost.
- `text_preprocessing_utils` — Contains scripts for text preparation and a set of functions for text preprocessing.
- `report_utils` — Contains scripts for analyzing raw data and exploring data distributions.
- `data` — Contains the data used for analysis.

Results obtained based on the combination of CatBoost and FRIDA:


| Dataset             |       TC      |                    |           |               |               |      Banks     |                    |           |               |               |
|---------------------|---------------|--------------------|-----------|---------------|---------------|----------------|--------------------|-----------|---------------|---------------|
| **Measure**         |   P           |   R                | F1       | macro $F1^{PN}$| micro $F1^{PN}$| P             | R                  |   F1      | macro $F1^{PN}$| micro $F1^{PN}$|
| Current SOTA        | –             | –                  | 68.42     | 66.07         | 74.11         | –              | –                  | 74.06     | 69.53         | 71.76         |
| M-BERT              | 65.73         | 67                 | 66.29     | 61.78         | 72.45         | 62.74          | 70.13              | 65.31     | 58            | 60.52         |
| RuBERT              | 70.82         | 70.57              | 70.68     | 66.4          | 76.71         | 71.05          | 74.62              | 72.83     | 65.89         | 68.43         |
| M-USE-CNN           | 64.3          | 63.12              | 63.64     | 58.97         | 71.31         | 66.06          | 68.28              | 66.71     | 58.73         | 62.41         |
| M-USE-Trans         | 69.45         | 67.44              | 68.27     | 62.77         | 75            | **73.04**      | 71.94              | 72.4      | 65.04         | 68.21         |
| FRIDA+CatBoost      | 78.18         | **75.37**          | **76.55** | **73.46**     | –             | 70.91          | **79.16**          | 73.78     | 69.1          | –             |
| FRIDA+CatBoost_join | –             | –                  | –         | –             | –             | –              |  –                 | **75**    | **71**        | –             |      
