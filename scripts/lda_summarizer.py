import os

import pandas as pd

from get_corpus import *
from lda import *


class LDASummarizer:
    """
    This class is used to cluster comments into topic clusters using LDA.
    """

    def __init__(self,
                 article_path,
                 comment_path,
                 directory=None,
                 MIN_DICT_WORDS=1,
                 RUNS=None,
                 T=None,
                 TRAINING_SHARE=None):
        """

        :param article_path: String The path to the file containing the articles.
        :param comment_path: String The path to the file containing the comments.
        :param directory:    String The working directory needed for the LDA implementation.
        """
        self._article_path = article_path
        self._comment_path = comment_path

        if directory is None:
            directory = os.path.join(os.getcwd(), '')
        self.directory = directory

        self._lda = LDA(self.directory, MIN_DICT_WORDS=MIN_DICT_WORDS, RUNS=RUNS, T=T, TRAINING_SHARE=TRAINING_SHARE)

    def summarize(self):
        """
        Runs the model and returns the clustered comments.

        :return: (dict, pandas.DataFrame) The topic clusters and comments with assigned clusters.
        """
        df_articles, df_comments = self.get_corpus()

        self._lda.run()

        clusters, df_comments = self._lda.get_topic_clusters(df_comments)
        return clusters, df_comments

    def get_corpus(self):
        """
        Creates the corpus.txt and meta.txt file and returns article and comment information in two pandas.DataFrame

        :return: (pandas.DataFrame, pandas.DataFrame) The article and comment data.
        """
        df_articles = pd.read_csv(self._article_path, delimiter='\t')
        df_comments = pd.read_csv(self._comment_path, delimiter='\t')

        df_articles, df_comments = get_and_write_corpus(df_articles, df_comments)
        return df_articles, df_comments