import numpy as np
import pandas as pd

"""
Contains functions to calculate BCubed cluster evaluation metrics as reported by Amigo et al. in
"A comparison of extrinsic clustering evaluation metrics based on formal constraints" (2009)
and by Bagga and Baldwin in "Algorithms for scoring coreference chains" (1998).
"""

def bcubed_precision_item(gold_standard, clustered_corpus, i):
    """
    Calculates the BCubed Precision for a single comment indicated by i.

    :param gold_standard:    pandas.DataFrame The gold standard with a cluster column indicating gold standard grouping.

    :param clustered_corpus: pandas.DataFrame The inferred clusters with a topic_id column indicating topic assignment.
    :param i:                int              Index of the comment.

    :return:                 float            BCubed Precision for comment i.
    """
    count = 0
    for j in gold_standard.index:
        if gold_standard.at[i, "cluster"] == gold_standard.at[j, "cluster"]:
            if clustered_corpus.at[i, "topic_id"] == clustered_corpus.at[j, "topic_id"]:
                count += 1
    return count / len(clustered_corpus[clustered_corpus["topic_id"] == clustered_corpus.at[i, "topic_id"]])


def bcubed_recall_item(gold_standard, clustered_corpus, i):
    """
    Calculates the BCubed Recall for a single comment indicated by i.

    :param gold_standard:    pandas.DataFrame The gold standard with a cluster column indicating gold standard grouping.

    :param clustered_corpus: pandas.DataFrame The inferred clusters with a topic_id column indicating topic assignment.
    :param i:                int              Index of the comment.

    :return:                 float            BCubed Precision for comment i.
    """
    count = 0
    for j in gold_standard.index:
        if gold_standard.at[i, "cluster"] == gold_standard.at[j, "cluster"]:
            if clustered_corpus.at[i, "topic_id"] == clustered_corpus.at[j, "topic_id"]:
                count += 1
    return count / len(gold_standard[gold_standard["cluster"] == gold_standard.at[i, "cluster"]])


def bcubed_precision(gold_standard, clustered_corpus):
    """
    Calculates the BCubed Precision.

    :param gold_standard:    pandas.DataFrame The gold standard with a cluster column indicating gold standard grouping.

    :param clustered_corpus: pandas.DataFrame The inferred clusters with a topic_id column indicating topic assignment.

    :return:                 float            BCubed Precision.
    """
    precisions = [bcubed_precision_item(gold_standard, clustered_corpus, i) for i in gold_standard.index]
    return np.mean(precisions)


def bcubed_recall(gold_standard, clustered_corpus):
    """
    Calculates the BCubed Recall.

    :param gold_standard:    pandas.DataFrame The gold standard with a cluster column indicating gold standard grouping.

    :param clustered_corpus: pandas.DataFrame The inferred clusters with a topic_id column indicating topic assignment.

    :return:                 float            BCubed Recall.
    """
    recalls = [bcubed_recall_item(gold_standard, clustered_corpus, i) for i in gold_standard.index]
    return np.mean(recalls)


def bcubed_f_alpha(gold_standard, clustered_corpus, alpha):
    """
    Calculates the BCubed F-measure using alpha for weighting.

    :param gold_standard:    pandas.DataFrame The gold standard with a cluster column indicating gold standard grouping.

    :param clustered_corpus: pandas.DataFrame The inferred clusters with a topic_id column indicating topic assignment.
    :param alpha:            float            Weighting parameter.
                                              alpha > 1 favours recall.
                                              alpha = 1 calculates the harmonic mean and does not favour either.
                                              alpha < 1 favours precision. Here, 0.5 is recommended.

    :return:                 float            BCubed Precision.
    """
    bcubed_p = bcubed_precision(gold_standard, clustered_corpus)
    bcubed_r = bcubed_recall(gold_standard, clustered_corpus)

    return (1 + alpha**2) * ((bcubed_p * bcubed_r) / ((alpha**2) * bcubed_p + bcubed_r))