import numpy as np
from flask import Flask, request, render_template, jsonify
from model import get_recommendation

DEFAULT_BLOG_TITLE = "Jim Carrey Blasts 'Castrato' Adam Schiff And Democrats In New Artwork"
DEFAULT_RECOMMONDATION_COUNT = 5
AS_API = True
app = Flask(__name__)


@app.route('/')
def nlp_route():
    return render_template('newsInput.html', size=0)


@app.route('/', methods=['POST'])
def cosine_model():
    title = request.form.get('title', DEFAULT_BLOG_TITLE)
    number = request.form.get('number', DEFAULT_RECOMMONDATION_COUNT)
    titles = get_recommendation(str(title), int(number))
    if AS_API:
        return jsonify(titles.tolist())
    return render_template('newsInput.html', size=len(titles), titles=titles)


if __name__ == '__main__':
    app.run(port='5000', debug=True)
