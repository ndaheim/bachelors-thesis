import csv
import os
import subprocess

import pandas as pd

from get_corpus import *
from similarity_measures import *


class MCL:
    """
    A wrapper around the Markov Cluster Algorithm (MCL) developed by Stijn van Dongen (2000) as part of his
    PhD thesis "GRAPH CLUSTERING by FLOW SIMULATION" at the University of Utrecht. More info can be found in the thesis
    or at https://micans.org/mcl/.
    """

    def read_existing_corpus(self, path):
        """
        If a pre-processed corpus already exists, this function can be used to obtain it for the MCL clustering.

        :param path: String             Path of the pre-processed corpus.

        :return:     pandas.DataFrame   The pre-processed documents.
        """
        return pd.read_csv(path, header=None)[0].tolist()

    def get_corpus_(self, comment_path):
        """
        Obtains the corpus of documents required to run the MCL algorithm.
        Note that the articles are not involved in the clustering but only the comments.

        :param comment_path: String Path of the comments.

        :return:             pandas.DataFrame The processed comments.
        """
        comments = pd.read_csv(comment_path, delimiter='\t')
        processed_comments = get_preprocessed_texts(comments)
        comments['processed'] = processed_comments
        return comments

    def write(self, list_):
        """
        Writes the given list into a graph graph in the A B C format specified in
        https://micans.org/mcl/index.html?sec_discovery, where each line contains (Node,Node,Edgeweight) triples
        separated by a whitespace.

        :param list_: [[String, String, float]] A list of node, node, edgeweight triples.
        """
        graph = pd.DataFrame(list_, columns=["node1", "node2", "edge"])
        # ensures that graph is markovian and its transition matrix stochastic
        # i.e. each outgoing edges of a node sum up to one
        #graph.edge = substitute_to_unit_interval(graph.edge)

        graph.to_csv('graph.txt', sep=' ', header=None, index=None)

    def write_graph(self, corpus):
        """
        Writes the graph in the A B C format specified in https://micans.org/mcl/index.html?sec_discovery, where
        each line contains (Node,Node,Edgeweight) triples separated by a whitespace.
        The weights were obtained as outlined in 'mcl_weights_regression' and rounded.


        :param corpus: pandas.DataFrame containing the corpus, for which the graph shall be established.
        """
        sparse_tfidf = tf_idf_vectorize(corpus["processed"].tolist())
        sparse_tf = tf_vectorize(corpus["processed"].tolist())
        sim_matrix_tfidf = get_cosine_similarity_pairwise(sparse_tfidf)
        sim_matrix_tf = get_cosine_similarity_pairwise(sparse_tf)

        corpus["nes"] = corpus.text.map(lambda row: set(named_entities(row)))

        list_ = []

        for row in range(sim_matrix_tfidf.shape[0]):

            if row % 10 == 0:
                print("Worked off {} vertices".format(row))

            c1 = corpus.iloc[row]
            # reduces the amount of set operations and results in a 1,25x speedup
            c1_set = set(c1.processed.split())

            # establish edge to each comment with a similarity weight
            for column in range(row+1, sim_matrix_tfidf.shape[1]):

                c2 = corpus.iloc[column]

                c2_set = set(c2.processed.split())

                sim = (0.22 * sim_matrix_tf[row][column]
                       + 0.12 * sim_matrix_tfidf[row][column]
                       + 0.15 * cosine_modified(c1_set, c2_set, is_set=True)
                       + 0.1 * dice(c1_set, c2_set, is_set=True)
                       + 0.1 * jaccard(c1_set, c2_set, is_set=True)
                       + 0.35 * same_thread(c1, c2)
                       + 0.1 * ne_overlap(c1.nes, c2.nes, chunked=True))

                # reduces the number of edges, which have to be written, as in Aker et al. 2016 0.3 is used
                if sim > 0.3:
                    list_.append([row, column, sim])

        self.write(list_)

    def write_graph_basic(self, corpus):
        """
        Writes the graph in the A B C format specified in https://micans.org/mcl/index.html?sec_discovery, where
        each line contains (Node,Node,Edgeweight) triples separated by a whitespace.
        Only cosine similarity and thread-relationship are considered for graph build-up.
        The weights were obtained as outlined in the notebook 'mcl_weights_regression'.

        :param corpus: pandas.DataFrame containing the corpus, for which the graph shall be established.
        """
        sparse_tfidf = tf_idf_vectorize(corpus["processed"].tolist())
        sim_matrix_tfidf = get_cosine_similarity_pairwise(sparse_tfidf)

        list_ = []

        for row in range(sim_matrix_tfidf.shape[0]):

            if row % 10 == 0:
                print("Worked off {} vertices".format(row))

            c1 = corpus.iloc[row]

            # establish edge to each comment with a similarity weight
            for column in range(row, sim_matrix_tfidf.shape[1]):

                c2 = corpus.iloc[column]

                sim = (1.6 * sim_matrix_tfidf[row][column]
                       + 0.1 * same_thread(c1, c2))

                # reduces the number of edges, which have to be written, in Aker et al. 2016 0.3 is used
                if sim > 0.1:
                    list_.append([row, column, sim])

        self.write(list_)

    def get_topic_clusters(self, corpus):
        """
        Returns the topic clusters as found by the MCL algorithm.

        :param corpus: pandas.DataFrame   The comment corpus.

        :return: (dict, pandas.DataFrame) A tuple containing a nested dict of cluster information with keys ranging from
                                          0 to |inferred topics|-1. Each subdict contains the key comments for the
                                          assigned comments.
                                          The pandas.DataFrame contains the comment set with assigned topic_id column.
        """
        with open('output_mcl/clusters.txt', 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for index, row in enumerate(reader):
                for elem in row:
                    corpus.at[int(elem), 'topic_id'] = int(index)

        clusters = {}
        for topic in corpus['topic_id'].unique():
            clusters[topic] = {}
            clusters[topic]['comments'] = corpus[corpus['topic_id'] == topic]
        return clusters, corpus

    def run(self, i=1.6):
        """
        Runs the MCL algorithm as implemented by Stijn van Dongen and found at https://micans.org/mcl/.

        :param i float inflation parameter ranging from 1.0 to 5.0.
                       Large i results in a fine granulated clustering.
                       Small i results in a coarse granulated clustering.
        :return:
        """
        if not os.path.isdir('output_mcl'):
            os.mkdir('output_mcl')

        try:
            with subprocess.Popen([
                    "./mcl", "../scripts/graph.txt",
                    "--abc", "-o",
                    "../scripts/output_mcl/clusters.txt",
                    "-I", str(i)],
                    cwd="../mcl", stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:

                #print(p.stdout.read())

                for line in p.stdout:
                    line = str(line)[2:-1].replace("\\n", "").replace("\\t", "   ")
                    print(line, end='\n')
                for line in p.stderr:
                    line = str(line)[2:-1].replace("\\n", "").replace("\\t", "   ")
                    print(line, end='\n')

        except subprocess.CalledProcessError as e:
            print(e.returncode)
            print(e.output)
