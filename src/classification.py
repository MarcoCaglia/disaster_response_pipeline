from etl_pipeline import MessageWrangler

import pandas as pd
import numpy as np
import sqlite3
import re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score

from tqdm import tqdm
tqdm.pandas(desc='progress-bar')
from sklearn import utils
from gensim.models import Doc2Vec as d2v
import gensim
from gensim.models.doc2vec import TaggedDocument
import pprint
from imblearn.over_sampling import SMOTE

import pickle
from pathlib import Path