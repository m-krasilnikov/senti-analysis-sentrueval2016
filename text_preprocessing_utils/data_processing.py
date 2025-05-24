# data_processing.py
import pandas as pd
import yaml
from tqdm import tqdm
from text_preprocessing_utils.text_preprocessor import TextProcessor


def process_data(config_path='config.yaml'):
    # Загружаем конфигурацию
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Инициализация обработчика текста
    string_processor = TextProcessor(config_path)

    # Загрузка данных
    train_data = pd.read_csv(config['input_train_data'])
    test_data = pd.read_csv(config['input_test_data'])

    # Настройка tqdm для прогресс-баров в pandas
    tqdm.pandas()

    # Обработка текста с прогресс-баром
    train_data["text"] = train_data["text"].progress_apply(lambda x: string_processor.run_text_processor(x))
    test_data["text"] = test_data["text"].progress_apply(lambda x: string_processor.run_text_processor(x))

    # Замена пустых строк на None
    train_data.replace(r'^\s*$', None, regex=True, inplace=True)
    test_data.replace(r'^\s*$', None, regex=True, inplace=True)

    # Удаление строк с NaN
    train_data = train_data.dropna()
    test_data = test_data.dropna()


    train_data.to_csv(config['output_train_data'], index=False)
    test_data.to_csv(config['output_test_data'], index=False)
    return train_data, test_data