from mcl import *


class MCLSummarizer():
    """
    This class is used to cluster comments into topic clusters using MCL.
    """

    def __init__(self, comment_path, basic=True):
        """

        :param comment_path: String  Path to the file containing the comments.
        :param basic:        boolean Determines the method of graph build up.
                                     True for Cosine Similarity and thread-relationship.
                                     False for the version as reported in Aker et al. 2016.
        """
        self._mcl = MCL()
        self.comment_path = comment_path
        self.basic = basic

    def run(self, i=1.6):
        """
        Runs the algorithm and returns the clustered comments.

        :return: (dict, pandas.DataFrame) The topic clusters and comments with assigned clusters.
        """
        comments = self._mcl.get_corpus_(self.comment_path)

        if self.basic:
            self._mcl.write_graph_basic(comments)
        else:
            self._mcl.write_graph(comments)

        self._mcl.run(i=i)

        clusters, comments = self._mcl.get_topic_clusters(comments)
        return clusters, comments