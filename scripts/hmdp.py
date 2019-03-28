import csv
import os
import shutil
import subprocess

import pandas as pd


class HMDP:
    """
    A wrapper around the Promoss HMDP implementation (https://github.com/ckling/promoss). Parts of it are taken and
    modified from the notebook https://github.com/ckling/promoss/blob/master/ipynb/hmdp.ipynb published with the promoss
    package and as is granted under the terms of the GNU General Public License as published by the Free Software
    Foundation.

    It contains the functions to run the model, connect the documents with their assigned topics, get topic words,
    get Perplexity and to get the weights of the considered context spaces.
    """

    def __init__(self,
                 directory=None,
                 meta_params=None,
                 RUNS=None,
                 T=None,
                 SAVE_STEP=10,
                 TRAINING_SHARE=1,
                 BATCHSIZE=128,
                 BATCHSIZE_GROUPS=128,
                 BURNIN=0,
                 BURNIN_DOCUMENTS=0,
                 INIT_RAND=0,
                 SAMPLE_ALPHA=1,
                 BATCHSIZE_ALPHA=1000,
                 MIN_DICT_WORDS=2,
                 alpha_0=1,
                 alpha_1=1,
                 epsilon="none",
                 delta_fix="none",
                 rhokappa=0.5,
                 rhotau=64,
                 rhos=1,
                 rhokappa_document=0.5,
                 rhotau_document=64,
                 rhos_document=1,
                 rhokappa_group=0.5,
                 rhotau_group=64,
                 rhos_group=1,
                 processed=True,
                 stemming=False,
                 stopwords=False,
                 language="en",
                 store_empty=True,
                 topk=10,
                 gamma=1,
                 learn_gamma=True
                 ):
        """
        For a list of model parameters see https://github.com/ckling/promoss.

        :param directory: String the working directory.
        """
        if directory is None:
            directory = os.path.join(os.getcwd(), '')
        self.directory = directory

        if meta_params is None:
            meta_params = "T(L100,Y10,M10,W20,D10);N;N"
        self.meta_params = meta_params

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
        self.BATCHSIZE_GROUPS = BATCHSIZE_GROUPS
        self.BURNIN = BURNIN
        self.BURNIN_DOCUMENTS = BURNIN_DOCUMENTS
        self.INIT_RAND = INIT_RAND
        self.SAMPLE_ALPHA = SAMPLE_ALPHA
        self.BATCHSIZE_ALPHA = BATCHSIZE_ALPHA
        self.MIN_DICT_WORDS = MIN_DICT_WORDS
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1
        self.epsilon = epsilon
        self.delta_fix = delta_fix
        self.rhokappa = rhokappa
        self.rhotau = rhotau
        self.rhos = rhos
        self.rhokappa_document = rhokappa_document
        self.rhotau_document = rhotau_document
        self.rhos_document = rhos_document
        self.rhokappa_group = rhokappa_group
        self.rhotau_group = rhotau_group
        self.rhos_group = rhos_group
        self.processed = processed
        self.stemming = stemming
        self.stopwords = stopwords
        self.language = language
        self.store_empty = store_empty
        self.topk = topk
        self.gamma = gamma
        self.learn_gamma = learn_gamma

    def run(self):
        """
        Runs the HMDP model. See https://github.com/ckling/promoss.

        :return:
        """

        print("Running HMDP topic model... (please wait)")

        if os.path.isdir(self.directory + "/output_HMDP"):
            shutil.rmtree(self.directory + "/output_HMDP")
        if os.path.isdir(self.directory + "/cluster_desc"):
            shutil.rmtree(self.directory + "/cluster_desc")

        if os.path.isfile(self.directory + "/groups"):
            os.remove(self.directory + "/groups")
        if os.path.isfile(self.directory + "/groups.txt"):
            os.remove(self.directory + "/groups.txt")
        if os.path.isfile(self.directory + "/text.txt"):
            os.remove(self.directory + "/text.txt")
        if os.path.isfile(self.directory + "/words.txt"):
            os.remove(self.directory + "/words.txt")
        if os.path.isfile(self.directory + "/wordsets"):
            os.remove(self.directory + "/wordsets")

        if not os.path.isfile("../lib/promoss.jar"):
            print("Could not find ../lib/promoss.jar. Exit")
            return
        try:
            with subprocess.Popen(['java', '-jar', '../lib/promoss.jar',
                                   '-directory', self.directory,
                                   '-meta_params', self.meta_params,
                                   '-T', str(self.T),
                                   '-RUNS', str(self.RUNS),
                                   '-SAVE_STEP', str(self.SAVE_STEP),
                                   '-TRAINING_SHARE', str(self.TRAINING_SHARE),
                                   '-BATCHSIZE', str(self.BATCHSIZE),
                                   '-BATCHSIZE_GROUPS', str(self.BATCHSIZE_GROUPS),
                                   '-BURNIN', str(self.BURNIN),
                                   '-BURNIN_DOCUMENTS', str(self.BURNIN_DOCUMENTS),
                                   '-INIT_RAND', str(self.INIT_RAND),
                                   '-SAMPLE_ALPHA', str(self.SAMPLE_ALPHA),
                                   '-BATCHSIZE_ALPHA', str(self.BATCHSIZE_ALPHA),
                                   '-MIN_DICT_WORDS', str(self.MIN_DICT_WORDS),
                                   '-alpha_0', str(self.alpha_0),
                                   '-alpha_1', str(self.alpha_1),
                                   '-epsilon', str(self.epsilon),
                                   '-delta_fix', str(self.delta_fix),
                                   '-rhokappa', str(self.rhokappa),
                                   '-rhotau', str(self.rhotau),
                                   '-rhos', str(self.rhos),
                                   '-rhokappa_document', str(self.rhokappa_document),
                                   '-rhotau_document', str(self.rhotau_document),
                                   '-rhos_document', str(self.rhos_document),
                                   '-rhokappa_group', str(self.rhokappa_group),
                                   '-rhotau_group', str(self.rhotau_group),
                                   '-rhos_group', str(self.rhos_group),
                                   '-processed', str(self.processed),
                                   '-stemming', str(self.stemming),
                                   '-stopwords', str(self.stopwords),
                                   '-language', str(self.language),
                                   '-store_empty', str(self.store_empty),
                                   '-topk', str(self.topk),
                                   '-gamma', str(self.gamma),
                                   '-learn_gamma', str(self.learn_gamma)
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
        return pd.read_csv('output_HMDP/{}/topktopic_words'.format(self.RUNS), delimiter='\t', header=None,
                           names=['topkwords'])

    def get_doc_topic_dist(self):
        """
        Returns the document-topic distribution for each of the documents in the corpus.

        :return: pandas.DataFrame The document-topic distribution for each of the documents in the corpus.
                                  Each column represents one topic. The row sorting matches that of the corpus.txt
                                  and meta.txt and the column sorting that of the topktopics file created by the HMDP
                                  implementation.
        """
        return pd.read_csv('output_HMDP/{}/doc_topic'.format(self.RUNS), header=None)

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
        Returns the topic clusters for the topics identified by HMDP containing the top k words, topic-word distribution
        and the comments.

        :param comments: pandas.DataFrame The comments with the already assigned topic_id column.

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
        with open('output_HMDP/{}/perplexity'.format(self.RUNS), 'r') as f:
            perplexity = float(f.read())
        return perplexity

    def get_context_weights(self, contexts=None):
        """
        Returns a pandas.DataFrame with the context weights.

        :param contexts: [String]         A list of names for the different context in the ordering they are specified
                                          in the metadata.
        :return:         pandas.DataFrame The context weights which determine their influence.
        """
        if contexts is None:
            return pd.read_csv('output_HMDP/{}/zeta'.format(self.RUNS), header=None)
        else:
            return pd.read_csv('output_HMDP/{}/zeta'.format(self.RUNS), header=None, names=contexts)

    def get_topic_word_dist(self, topic):
        """
        Returns a dictionary mapping each word to its probability under the topic-word distribution of the topic.

        :param   int  The topic number as indicated by its topic_id.

        :return: dict A mapping of word to likelihood under topic-word distribution.
        """
        with open('output_HMDP/{}/topktopics'.format(self.RUNS), 'r') as f:
            reader = list(csv.reader(f))
            dict_ = dict(zip(reader[2*topic][:-1], [float(p) for p in reader[2*topic+1][:-1]]))
        return dict_

