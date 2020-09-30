import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from configparser import ConfigParser

config=ConfigParser()
config.read('config.ini')


CONNECTION_STRING = config['DEFAULT']['CONNECTION_STRING']
TABLE_NAME = config['DEFAULT']['TABLE_NAME']
COLUMN_NAMES = config['DEFAULT']['COLUMN_NAMES']
COLUMN_NAMES=[COLUMN_NAMES]     #convert str to list

def connect_db(connection_string):
    return create_engine(connection_string)


def generate_model(table_name, engine, col_names):
    raw_data = pd.read_sql(table_name, engine, columns=col_names)  # read data
    data_to_consider = raw_data[['title']]  # column selection
    np.savez('raw_data.npz', title=data_to_consider['title'])
    vectorizer = TfidfVectorizer(stop_words='english', analyzer='word', encoding='utf-8')
    feature_sm = vectorizer.fit_transform(data_to_consider['title'].values.astype('U'))
    cosines = pairwise_distances(feature_sm, metric='cosine', n_jobs=-1)
    np.save('cosine_model.npy', cosines)


if __name__ == '__main__':
    generate_model(table_name=TABLE_NAME, engine=connect_db(CONNECTION_STRING), col_names=COLUMN_NAMES)
    print('model generated')
