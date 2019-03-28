from sklearn.feature_extraction.text import TfidfVectorizer

from similarity_measures import *


class MMR:
    """
    This class implements ranking with Maximal Marginal Relevance as defined by Carbonell and Goldstein in
    "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries" (1998)
    in the context of the summarizer.
    It contains functions for scoring and ranking.
    """

    def __init__(self, vectorizer=None):
        """

        :param vectorizer: Vectorizer for document transformation. If none is given,
                           sklearn.feature_extraction.text.TfidfVectorizer is used.
        """
        if vectorizer is None:
            vectorizer = TfidfVectorizer()
        self._vectorizer = vectorizer


    def get_ranked_clusters(self, clusters, lambda_):
        """
        Returns the ranked clusters, with comments sorted by MMR score.

        :param clusters: dict  Containing the topic clusters with a dict for each topic_id and the comments of each
                               topic cluster under the comments key of said dict.
        :param lambda_:  float Parameter determining the diversity of the ranking.
                               lamda_ = 1 results in a ranking purely based on query similarity.
                               Larger lambda_ results in a more query-oriented (but possibly redundant) ranking.
                               Smaller lambda_ focusses more on eliminating redundancy than query-similarity.

        :return:         dict  Containing the topic clusters with comments ranked by MMR score. The pandas.DataFrame
                               of comments of each dict has an additional mmr_score column containing it.
        """
        for topic in clusters:
            # Hard clustering on topics can produce topics which are not the most likely one for any comment
            if len(clusters[topic]["comments"]) != 0.0:
                clusters[topic]["comments"] = self.rank(clusters[topic], lambda_)
        return clusters

    def score(self, comment, selected, query, lambda_):
        """
        Scores the given comment based on the already selected comments, the query and parameter lambda_.

        :param comment: pandas.DataFrame  The pandas.DataFrame row of the comment with a column called "processed"
                                          containing the preprocessed comment text.
        :param selected: pandas.DataFrame The comments which have already been scored.
        :param query:   String            Query. In the case of the summarizer this equals the top words under the
                                          topic-word distribution of the topic.
        :param lambda_: float             Parameter determining the diversity of the ranking.
                                          lamda_ = 1 results in a ranking purely based on query similarity.
                                          Larger lambda_ results in a more query-oriented (but possibly redundant)
                                          ranking.
                                          Smaller lambda_ focusses more on eliminating redundancy than query-similarity.

        :return:        float MMR score of the comment.
        """
        query_sim = get_cosine_similarity_pairwise(self._vectorizer.transform([comment.processed, query]))[0][1]

        if len(selected) > 0:
            M = get_cosine_similarity_pairwise(self._vectorizer.transform(selected.append(comment).processed))
            # comment is always the last row
            max_doc_sim = np.sort(M[-1])[-2]
        else:
            max_doc_sim = 0.0

        return lambda_* (query_sim - (1 - lambda_) * max_doc_sim)

    def rank(self, cluster, lambda_):
        """
        Ranks the comments of the topic cluster by calculating MMR score and ordering them.

        :param cluster: dict The topic cluster containing a pandas.DataFrame under the key "comments" and a topic
                             representation under the key "words".

        :return:        dict The ranked topic cluster.
        """
        comments = cluster["comments"].reset_index(drop=True)
        comments["mmr_score"] = -1.0

        self._vectorizer = self._vectorizer.fit(comments.processed)

        query = cluster["words"]

        for index, comment in comments.iterrows():
            selected = comments[comments["mmr_score"] > -1.0]
            comments.at[index, "mmr_score"] = self.score(comment, selected, query, lambda_)
        return comments.sort_values("mmr_score", ascending=False)

