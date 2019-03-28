import os

import pandas as pd

from get_corpus import *
from get_metadata import *
from hmdp import *


class HMDPSummarizer:
    """
    This class is used to cluster comments into topic clusters using HMDP.
    """

    def __init__(self,
                 article_path,
                 comment_path,
                 directory=None,
                 MIN_DICT_WORDS=1,
                 RUNS=None,
                 T=None,
                 TRAINING_SHARE=None,
                 doc_id=True,
                 timestamp=True,
                 thread_id=True):
        """

        :param article_path: String  The path to the file containing the articles.
        :param comment_path: String  The path to the file containing the comments.
        :param directory:    String  The working directory needed for the HMDP implementation.
        :param doc_id:       boolean Whether a document_id identifying the article a comment was issued under shall be
                                     included in the metadata,
        :param thread_id:    boolean Whether a thread_id identifying the the comment thread a comment was issued in
                                     shall be included the metadata.
        :param timestamp:    boolean Whether a timestamp shall be included in the metadata.
        """
        self._article_path = article_path
        self._comment_path = comment_path

        if directory is None:
            directory = os.path.join(os.getcwd(), '')
        self.directory = directory

        self._doc_id = doc_id
        self._timestamp = timestamp
        self._thread_id = thread_id

        self._hmdp = HMDP(self.directory, MIN_DICT_WORDS=MIN_DICT_WORDS, RUNS=RUNS, T=T, TRAINING_SHARE=TRAINING_SHARE)

    def run(self):
        """
        Runs the model and returns the clustered comments.

        :return: (dict, pandas.DataFrame) The topic clusters and comments with assigned clusters.
        """
        df_articles, df_comments = self.get_corpus_and_metadata()

        self._hmdp.run()

        clusters, df_comments = self._hmdp.get_topic_clusters(df_comments)
        return clusters, df_comments


    def get_corpus_and_metadata(self):
        """
        Creates the corpus.txt and meta.txt file and returns article and comment information in two pandas.DataFrame

        :return: (pandas.DataFrame, pandas.DataFrame) The article and comment data.
        """
        df_articles = pd.read_csv(self._article_path, delimiter='\t')
        df_comments = pd.read_csv(self._comment_path, delimiter='\t')

        df_articles, df_comments = get_and_write_corpus(df_articles, df_comments)
        df_articles, df_comments = get_and_write_metadata(
            df_articles,
            df_comments,
            doc_id=self._doc_id,
            timestamp=self._timestamp,
            thread_id=self._thread_id
        )
        return df_articles, df_comments