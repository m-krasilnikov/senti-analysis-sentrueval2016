import pandas as pd
import yaml
from tqdm import tqdm

from text_preprocessing_utils.text_preprocessor import TextProcessor
from text_preprocessing_utils.data_processing import process_data



if __name__=="__main__":
    train_data, test_data = process_data('config.yaml')

