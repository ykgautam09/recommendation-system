import numpy as np
from configparser import ConfigParser

config=ConfigParser()
config.read('./config.ini')

BLOG_TITLE = config['DEFAULT']['BLOG_TITLE']
PREDICTION_COUNT = int(config['DEFAULT']['PREDICTION_COUNT'])
MODEL_PATH=config['DEFAULT']['MODEL_PATH']
DATA_PATH=config['DEFAULT']['DATA_PATH']

cosine_model = np.load(MODEL_PATH, allow_pickle=True)
data = np.load(DATA_PATH, allow_pickle=True)

def get_recommendation(blog, n):
    index = np.where(data['title'] == blog)[0]
    result = np.argsort(cosine_model[index])[0][1:int(n) + 1]
    title = data['title'][result]  # recommended title
    return title


if __name__ == '__main__':
    print(get_recommendation(BLOG_TITLE, PREDICTION_COUNT))
