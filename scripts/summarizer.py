from topic_labeling import *
from lda_summarizer import *
from hmdp_summarizer import *
from mcl_summarizer import *
from mmr import *
from sentiment import *


class Summarizer:
    """
    Wrapper class for the overall summarization effort.
    """

    def __init__(self,
                 comment_path,
                 article_path=None,
                 method='hmdp',
                 RUNS=None,
                 T=None,
                 TRAINING_SHARE=None,
                 basic=True):
        """

        :param comment_path:   String  Path to the file containing the comments.
        :param article_path:   String  Path to the file containing the articles. If the chosen method is 'mcl' this parameter
                                       stays unused.
        :param method:         String  Model / Algorithm to use for topic clustering.
                                       'hmdp' for Hierarchical multi-Dirichlet Process model
                                       'lda' for Latent Dirichlet Allocation
                                       'mcl' for Markov Clustering Algorithm
        :param RUNS:           integer Number of runs to be performed by the clustering algorithm / model.
        :param T:              integer Number of topics, used as a truncation in HMDP and fixation in LDA.
        :param TRAINING_SHARE: float   Share of data going into training, the rest will be withheld and considered in perplexity
                                       measurement of LDA and HMDP. For MCL this stays without effect.
        :param basic:          boolean Determines the graph build-up method for MCL.
                                       True for Cosine Similarity and Thread-relationship
                                       False for Cosine Similarity based on TF and TFIDF, Jaccard, Dice, NE Overlap and
                                       Thread-relationship
        """
        if method == 'hmdp':
            self.model = HMDPSummarizer(article_path, comment_path, RUNS=RUNS, T=T, TRAINING_SHARE=TRAINING_SHARE)

        if method == 'lda':
            self.model = LDASummarizer(article_path, comment_path, RUNS=RUNS, T=T, TRAINING_SHARE=TRAINING_SHARE)

        if method == 'mcl':
            self.model = MCLSummarizer(comment_path, basic=basic)

        self._mmr = MMR()


    def summarize(self):
        """
        Summarizes the comments based on the chosen method.
        The summarization consists of the following:
        1) Clustering comments by their topic
        2) Ranking and selecting clusters by the amount of contained comments
        3) Labeling topic clusters
        4) Calculating polarity for comments
        5) Ranking and selecting comments by MMR score

        :return: (dict, pandas.DataFrame) The selected topic clusters and comments.
        """
        clusters, comments = self.model.run()

        clusters = self._mmr.get_ranked_clusters(clusters, 0.8)

        top_topics = comments.topic_id.value_counts().nlargest(10)
        top_clusters = {topic: clusters[topic] for topic in top_topics.index}
        top_clusters = trigram_labels_topics(top_clusters)

        for topic in top_clusters:
            _comments = polarity(top_clusters[topic]["comments"])
            top_clusters[topic]["top_pos"] = _comments.loc[
                _comments[_comments["polarity"] >= 0]["mmr_score"].idxmax()].text
            top_clusters[topic]["top_neg"] = _comments.loc[
                _comments[_comments["polarity"] < 0]["mmr_score"].idxmax()].text
            top_clusters[topic]["polarity"] = top_clusters[topic]["comments"]["polarity"].mean()

        top_comments = comments[comments["topic_id"].isin(top_topics.index)]

        top_comments = polarity(top_comments)
        top_comments["label"] = top_comments.topic_id.map(lambda t: top_clusters[t]["label"])

        return top_clusters, top_comments