import csv
import os
import shutil
import subprocess

import pandas as pd


class LDA:
    """
    This wrapper around the Promoss LDA implementation (https://github.com/ckling/promoss) is inspired by the
    notebook https://github.com/ckling/promoss/blob/master/ipynb/hmdp.ipynb published with the promoss package
    as is granted under the terms of the GNU General Public License as published by the Free Software Foundation.

    It contains the functions to run the model, connect the documents with their assigned topics, get topic words
    and Perplexity.
    """

    def __init__(self,
                 directory=None,
                 RUNS=None,
                 T=None,
                 SAVE_STEP=10,
                 TRAINING_SHARE=1,
                 BATCHSIZE=128,
                 BURNIN=0,
                 # Setting INIT_RAND to 0 results in alpha increasing very quickly and only one topic found RUNS times.
                 INIT_RAND=1,
                 MIN_DICT_WORDS=1,
                 alpha=1,
                 rhokappa=0.5,
                 rhotau=64,
                 rhos=1,
                 rhokappa_document=0.5,
                 rhotau_document=64,
                 rhos_document=1,
                 processed=True,
                 stemming=False,
                 stopwords=False,
                 language="en",
                 store_empty=True,
                 topk=15,
                 ):
        """
        For a list of model parameters see https://github.com/ckling/promoss.

        :param directory: String the working directory.
        """
        if directory is None:
            directory = os.path.join(os.getcwd(), '')
        self.directory = directory

        if RUNS is None:
            RUNS = 100
        self.RUNS = RUNS

        if T is None:
            T = 100
        self.T = T

        if TRAINING_SHARE is None:
            TRAINING_SHARE = 1
        self.TRAINING_SHARE = TRAINING_SHARE

        self.SAVE_STEP = SAVE_STEP
        self.TRAINING_SHARE = TRAINING_SHARE
        self.BATCHSIZE = BATCHSIZE
        self.BURNIN = BURNIN
        self.INIT_RAND = INIT_RAND
        self.MIN_DICT_WORDS = MIN_DICT_WORDS
        self.alpha = alpha
        self.rhokappa = rhokappa
        self.rhotau = rhotau
        self.rhos = rhos
        self.rhokappa_document = rhokappa_document
        self.rhotau_document = rhotau_document
        self.rhos_document = rhos_document
        self.processed = processed
        self.stemming = stemming
        self.stopwords = stopwords
        self.language = language
        self.store_empty = store_empty
        self.topk = topk

    def run(self):
        """
        Runs the LDA model. See https://github.com/ckling/promoss.

        :return:
        """

        print("Running LDA topic model... (please wait)")

        if os.path.isdir(self.directory + "/output_LDA"):
            shutil.rmtree(self.directory + "/output_LDA")

        if os.path.isfile(self.directory + "/text.txt"):
            os.remove(self.directory + "/text.txt")
        if os.path.isfile(self.directory + "/words.txt"):
            os.remove(self.directory + "/words.txt")
        if os.path.isfile(self.directory + "/wordsets"):
            os.remove(self.directory + "/wordsets")

        if not os.path.isfile("../promoss/promoss.jar"):
            print("Could not find ../promoss/promoss.jar. Exit")
            return
        try:
            with subprocess.Popen(['java', '-jar', '../promoss/promoss.jar',
                                   '-directory', self.directory,
                                   '-method', 'LDA',
                                   '-T', str(self.T),
                                   '-RUNS', str(self.RUNS),
                                   '-SAVE_STEP', str(self.SAVE_STEP),
                                   '-TRAINING_SHARE', str(self.TRAINING_SHARE),
                                   '-BATCHSIZE', str(self.BATCHSIZE),
                                   '-BURNIN', str(self.BURNIN),
                                   '-INIT_RAND', str(self.INIT_RAND),
                                   '-MIN_DICT_WORDS', str(self.MIN_DICT_WORDS),
                                   '-alpha', str(self.alpha),
                                   '-rhokappa', str(self.rhokappa),
                                   '-rhotau', str(self.rhotau),
                                   '-rhos', str(self.rhos),
                                   '-rhokappa_document', str(self.rhokappa_document),
                                   '-rhotau_document', str(self.rhotau_document),
                                   '-rhos_document', str(self.rhos_document),
                                   '-processed', str(self.processed),
                                   '-stemming', str(self.stemming),
                                   '-stopwords', str(self.stopwords),
                                   '-language', str(self.language),
                                   '-store_empty', str(self.store_empty),
                                   '-topk', str(self.topk),
                                   ], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:

                for line in p.stdout:
                    line = str(line)[2:-1].replace("\\n", "").replace("\\t", "   ")
                    print(line, end='\n')
                for line in p.stderr:
                    line = str(line)[2:-1].replace("\\n", "").replace("\\t", "   ")
                    print(line, end='\n')

        except subprocess.CalledProcessError as e:
            print(e.returncode)
            print(e.output)

    def get_topics(self):
        """
        Returns a pandas.DataFrame with the topics indicated by topic words.

        :return: pandas.DataFrame Contains the topics indicated by topic words.
        """
        return pd.read_csv('output_LDA/{}/topktopics'
                           .format(self.RUNS), delimiter='\t', header=None, names=['topkwords']).iloc[::2, :].reset_index()

    def get_doc_topic_dist(self):
        """
        Returns the document-topic distribution for each of the documents in the corpus.

        :return: pandas.DataFrame The document-topic distribution for each of the documents in the corpus.
                                  Each column represents one topic. The row sorting matches that of the corpus.txt
                                  and meta.txt and the column sorting that of the topktopics file created by the LDA
                                  implementation.
        """
        return pd.read_csv('output_LDA/{}/doc_topic'.format(self.RUNS), header=None)

    def get_comment_topics(self, comments):
        """
        Returns the pandas.DataFrame of the comments with an additional column containing a topic identifier.

        :return: pandas.DataFrame The comments with an additional column containing a topic identifier.
                                  The identifier specifies the column in the pandas.DataFrame created by
                                  self.get_doc_topic_dist().
        """
        # only the last k distributions are needed as the first n-k distributions model the document-topic distribution
        # of the articles
        doc_topic_dist = self.get_doc_topic_dist().tail(len(comments.index)).reset_index(drop=True)
        # the column index with the highest value represents the most likely topic
        comments['topic_id'] = doc_topic_dist.idxmax(axis='columns')
        return comments

    def get_topic_clusters(self, comments):
        """
        Returns the topic clusters for the topics identified by LDA containing the top k words and the comments.

        :param comments: pandas.DataFrame The comments with the already assigned topic_id column

        :return: (dict, pandas.DataFrame) A tuple containing a nested dict of cluster information with keys ranging from
                                          0 to |inferred topics|-1. Each subdict contains the keys comments for the
                                          assigned comments, twdist for the topic-word-distribution of the topic and
                                          words containing the topic-word distribution of the topic.
                                          The pandas.DataFrame contains the comment set with assigned topic_id column.
        """
        clusters = {}
        topics = self.get_topics()
        comments = self.get_comment_topics(comments)

        for topic in topics.index.values:
            clusters[topic] = {}
            clusters[topic]['words'] = self.get_topics().iloc[topic].topkwords
            clusters[topic]['twdist'] = self.get_topic_word_dist(topic)
            clusters[topic]['comments'] = comments[comments['topic_id'] == topic]

        return clusters, comments

    def get_perplexity(self):
        """
        Returns the perplexity of the trained model if TRAINING_SHARE < 1. Else 'NaN' is returned.

        :return: float The perplexity of the trained model.
        """
        with open('output_LDA/{}/perplexity'.format(self.RUNS), 'r') as f:
            perplexity = float(f.read())
        return perplexity

    def get_topic_word_dist(self, topic):
        """
        Returns a dictionary mapping each word to its probability under the topic-word distribution of the topic.

        :param   int  The topic number as indicated by its topic_id.

        :return: dict A mapping of word to likelihood under topic-word distribution.
        """
        with open('output_LDA/{}/topktopics'.format(self.RUNS), 'r') as f:
            reader = list(csv.reader(f))
            dict_ = dict(zip(reader[2*topic][:-1], [float(p) for p in reader[2*topic+1][:-1]]))
        return dict_