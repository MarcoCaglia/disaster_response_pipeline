import sqlite3
import re
import pandas as pd
from sklearn.model_selection import train_test_split


class Vectorizer():
    def __init__(self, path_to_db):
        data = self._get_data(path_to_db)
        data.message = data.message.map(self._clean_string)


    def _clean_string(self, string):
        delete_punctuation = re.compile('[^a-zA-Z0-9]')
        clean_string = re.sub(delete_punctuation, ' ', string)
        return clean_string


    def _get_data(self, path):
        connection = sqlite3.connect(path)
        query = 'SELECT * FROM data'
        data = pd.read_sql(query, connection)

        return data
