import xml.etree.ElementTree as et
import numpy as np
import pandas as pd

class XML_parser():
    '''
    Класс для парсинга XML-файлов и преобразования данных для обучения или тестирования.
    '''

    def xml_parse(self, filename):
        root = et.parse(filename).getroot()
        tweets = []
        classes = []

        for table in root.iter('table'):
            aux_count = 0
            value = None  # Инициализация переменной
            for column in table[4:]:
                if column.text != 'NULL':
                    value = column.text
                    aux_count += 1
            # Добавляем только если есть ровно один ненулевой столбец
            if aux_count == 1 and value is not None:
                classes.append(value)
                tweets.append(table[3].text)
        target = np.array(classes)
        return tweets, target

if __name__ == '__main__':
    input_xml_file = '../data/bank_train_2016.xml'
    output_csv_file = '../data/output/bank_train.csv'
    parser = XML_parser()
    tweets, target = parser.xml_parse(input_xml_file)
    df = pd.DataFrame({'text': tweets, 'label': target})
    df.to_csv(output_csv_file, index=False)