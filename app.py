from flask import Flask, request, render_template, jsonify
from model import get_recommendation
import configparser

config=configparser.ConfigParser()
app = Flask(__name__)

DEFAULT_BLOG_TITLE = config['DEFAULT']['BLOG_TITLE']
DEFAULT_RECOMMONDATION_COUNT = int(config['DEFAULT']['PREDICTION_COUNT'])
AS_API = bool(config['DEFAULT']['AS_API'])



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
