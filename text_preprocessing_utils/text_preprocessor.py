import re
from nltk.corpus import stopwords
import yaml
import string

russian_stopwords = set(stopwords.words('russian'))
words_to_remove = ['rt']
class TextProcessor:
    def __init__(self, config_path):
        self.steps = self._load_config(config_path)
        # Создаем карту функций, ссылаясь на методы экземпляра
        self.function_map = {
            'to_lower_case': self._to_lower_case,
            'replace_email_with_token': self._replace_email_with_token,
            'replace_url_with_token': self._replace_url_with_token,
            'replace_mentions_with_token': self._replace_mentions_with_token,
            'replace_hashtags_with_token': self._replace_hashtags_with_token,
            'remove_punctuation': self._remove_punctuation,
            'remove_rus_stop_words': self._remove_rus_stop_words,
            'remove_special_words': self._remove_special_words,
            'remove_empty_text': self._remove_empty_text,
        }

    def run_text_processor(self, text):
        pipeline_functions = [self.function_map[step] for step in self.steps]
        text = self._preprocess_text(text, pipeline_functions)
        return text

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            steps = config['preprocessing']['steps']
        return steps

    def _preprocess_text(self, text: str, pipeline: list):
        for func in pipeline:
            text = func(text)
        return text

    def _remove_rus_stop_words(self, text):
        words = text.split()
        filtered_words = [word for word in words if word not in russian_stopwords]
        return " ".join(filtered_words)

    def _to_lower_case(self, text: str):
        return text.lower()

    def _replace_email_with_token(self, text, token=''):
        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        return re.sub(email_pattern, token, text)

    def _replace_url_with_token(self, text, token=''):
        url_pattern = r'https?://\S+'
        return re.sub(url_pattern, token, text)

    def _replace_mentions_with_token(self, text, token=''):
        mention_pattern = r'@\w+'
        return re.sub(mention_pattern, token, text)

    def _replace_hashtags_with_token(self, text, token=''):
        hashtag_pattern = r'#\w+'
        return re.sub(hashtag_pattern, token, text)

    def _remove_special_words(self, text):
        pattern = r'\b(' + '|'.join(words_to_remove) + r')\b'
        return re.sub(pattern, '', text)

    def _remove_punctuation(self, text):
        # Удаляет все знаки пунктуации
        return text.translate(str.maketrans('', '', string.punctuation))

    def _remove_empty_text(self, text):
        pattern = r'\s+$'
        return text.replace(r'pattern', ' ')