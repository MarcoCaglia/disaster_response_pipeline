import pandas as pd
import sqlite3
import pprint
import nltk
import re
import pickle
import multiprocessing
import gensim.models.word2vec as w2v
from tqdm import tqdm

class Vectorizor():
    def __init__(self, path_to_db):
        self.data = self._get_data(path_to_db)

def _get_data(self, path):
    conn = sqlite3.connect(path)
    query = 'SELECT * FROM data'
    return pd.read_sql(query, con=conn)

def _get_tokenizer(self):
    pass