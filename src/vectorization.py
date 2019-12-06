import sqlite3
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import TaggedDocument


class Vectorizer():
    def __init__(self, path_to_db):
        data = self._get_data(path_to_db)
        data.message = data.message.map(self._clean_string)
        X_train, X_test, y_train, y_test = train_test_split(data.message,
                                                            data.loc[
                                                                :, 'related':
                                                                ],
                                                            train_size=0.75
                                                            )
        X_train = self._label_messages(X_train, 'train')
        X_test = self._label_messages(X_test, 'test')
        y_train, y_test = self._clean_target(y_train, y_test)

    def _clean_target(self, y1, y2):
        for col in y1.columns:
            y1[col] = ###### CURRENT

    def _label_messages(self, column, label_type):
        labeled = [TaggedDocument(cell.lower().split(),
                   [label_type + '_' + str(i)]) for i, cell in
                   enumerate(column)
                   ]
        return labeled

    def _clean_string(self, string):
        delete_punctuation = re.compile('[^a-zA-Z0-9]')
        clean_string = re.sub(delete_punctuation, ' ', string)
        return clean_string

    def _get_data(self, path):
        connection = sqlite3.connect(path)
        query = 'SELECT * FROM data'
        data = pd.read_sql(query, connection)

        return data
