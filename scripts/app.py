import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import *
from werkzeug.utils import secure_filename

from summarizer import *

plt.style.use('seaborn')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'tsv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'foobar'

@app.route("/")
def index():
    if not os.path.isdir(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)

    return render_template('index.html')

@app.route("/upload_articles", methods=['GET', 'POST'])
def upload_articles():
    """ Uploads the input articles for summarization. """
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(UPLOAD_FOLDER, secure_filename("articles.csv")))

    return redirect(url_for('index'))

@app.route("/upload_comments", methods=["GET", "POST"])
def upload_comments():
    """ Uploads the input comments for summarization. """
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(UPLOAD_FOLDER, secure_filename("comments.csv")))

    return redirect(url_for('index'))

@app.route("/summarize", methods=["GET", "POST"])
def summarize():
    """
    Wraps the summarization by loading articles and comments, employing the summarizer and plotting sentiment.

    :return:
    """
    clustering_method = request.form.get('cluster_select')

    comment_path = os.path.join(UPLOAD_FOLDER, 'comments.csv')
    article_path = os.path.join(UPLOAD_FOLDER, 'articles.csv')

    s = Summarizer(comment_path, article_path=article_path, method=clustering_method, T=session["topic_number"])

    top_clusters, top_comments = s.summarize()

    plot_sentiment(top_comments)

    df = pd.concat([pd.DataFrame({"positive": top_clusters[topic]["top_pos"],
                                  "negative": top_clusters[topic]["top_neg"]},
                                 index=[top_clusters[topic]["label"]]) for topic in top_clusters])

    pd.set_option('display.max_colwidth', -1)
    return render_template('summary.html', tables=[df.to_html(col_space=500)], titles=["comments"])


def plot_sentiment(top_comments):
    """
    Plots the sentiment of the top 10 topics in a violin plot scaled by the amount of comments in the topic cluster.

    :param top_comments: pandas.DataFrame containing the comments, a label column containing the topic label of each
                                          comment and a polarity column indicating sentiment.
    :return:
    """
    _fig_path = 'static/sentiment.svg'

    ax = sns.violinplot(x="label", y="polarity", data=top_comments, scale='count', saturation=1)
    ax.axhline(y=0, color='k', linestyle='--', label='neutral sentiment')
    ax.legend()

    plt.xlabel("")
    plt.ylabel('Mean sentiment score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if os.path.isfile(_fig_path):
        os.remove(_fig_path)
    plt.savefig(_fig_path)

@app.route('/topic_number', methods=['POST'])
def topic_number():
    """
    Saves the input topic number in the session.
    :return:
    """
    text = request.form['text']
    session['topic_number'] = int(text)

    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)