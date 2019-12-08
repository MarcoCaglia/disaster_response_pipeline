import sqlite3
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec as d2v
from tqdm import tqdm
import numpy as np
import pickle

class InvalidParametersError(Exception):
    pass


class Vectorizer():
    def __init__(self, path_to_db):
        data = self._get_data(path_to_db)
        data.message = data.message.map(self._clean_string)
        data = self._clean_targets(data)
        X_train, X_test, y_train, y_test = train_test_split(data.message,
                                                            data.loc[
                                                                :, 'related':
                                                                ],
                                                            train_size=0.75
                                                            )
        self.X_train = self._label_messages(X_train, 'train')
        self.X_test = self._label_messages(X_test, 'test')
        self.y_train, self.y_test = self._convert_to_numeric(y_train, y_test)

    def _clean_targets(self, data):
        drop_list = [col for col in data.columns if data[col].nunique() < 2]
        data.drop(drop_list, axis=1, inplace=True)
        return data

    def fit(self, dm=0, vector_size=300, window=7, sample=1e-3, negative=5, min_count=5, epochs=30, workers=1):
        documents = self.X_train + self.X_test
        self.model = d2v(dm=dm, vector_size=vector_size, window=window, sample=sample, negative=negative, min_count=min_count, workers=workers)
        self.model.build_vocab(documents)

        for _ in tqdm(range(epochs+1)):
            self.model.train(utils.shuffle([x for x in documents]), total_examples=len(documents), epochs=1)

        return_X_train = self._get_vectors(corpus_size=len(self.X_train), vector_size=vector_size, vectors_type='train')
        return_X_test = self._get_vectors(corpus_size=len(self.X_test), vector_size=vector_size, vectors_type='test')

        return return_X_train, return_X_test, self.y_train, self.y_test

    def _get_vectors(self, corpus_size, vector_size, vectors_type):
        vectors = np.zeros((corpus_size, vector_size))
        for i in range(corpus_size):
            prefix = vectors_type + '_' + str(i)
            vectors[i] = self.model.docvecs[prefix]
        return vectors

    def _convert_to_numeric(self, *args):
        return_list = []
        for dataset in args:
            for col in dataset:
                dataset[col] = pd.to_numeric(dataset[col], errors='raise')
            return_list.append(dataset)
        return tuple(return_list)

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
