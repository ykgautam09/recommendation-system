import numpy as np

BLOG_TITLE = "Jim Carrey Blasts 'Castrato' Adam Schiff And Democrats In New Artwork"
PREDICTION_COUNT = 5

cosine_model = np.load('./cosine_model.npy', allow_pickle=True)
data = np.load('./raw_data.npz', allow_pickle=True)


def get_recommendation(blog, n):
    index = np.where(data['title'] == blog)[0]
    result = np.argsort(cosine_model[index])[0][1:int(n) + 1]
    title = data['title'][result]  # recommended title
    return title


if __name__ == '__main__':
    print(get_recommendation(BLOG_TITLE, PREDICTION_COUNT))
