from collections import Counter

import nltk
import numpy as np
import pandas as pd

from pre_process import *

"""
Contains functions to label topic clusters using either topic word distribution or the most frequent words in comments.
For both cases bigrams and trigrams can be extracted as labels.
"""


def get_words(comments):
    """
    Returns a list of all words in the lowercased comments with stop words and punctuation removed.

    :param comments: pandas.DataFrame The comments with a column "text" containing the raw comment text.
    :return:
    """
    return " ".join(comments.text.map(lambda text: " ".join(
        remove_stop_words(
            to_lower(
                tokenize(
                    replace_apostrophe(text)
                )
            )
        )
    )).tolist()).split(" ")


def bigram_labels_topics(clusters):
    """
    Finds the bigram with the largest sum of size of intersection between label and top words under topic-word
    distribution and probability of label words under topic-word distribution.

    :param clusters: dict The topic clusters with keys indicating topic by topic_id. Each subdict contains a key
                          "comments" containing a pandas.DataFrame with the comments, a key "words" with the top
                          words under topic-word distribution and a key "twdist" with a dictionary mapping each
                          word to its likelihood under topic-word distribution.

    :return:         dict The dict of clusters with an additional key "label" containing the label for each topic
                          cluster.
    """
    filter_ = lambda word1, word2: len({word1, word2}.intersection(set(words.split(" ")))) == 0

    for topic in clusters:

        words = clusters[topic]["words"]

        # find bigrams from comments
        tokenizer = RegexpTokenizer(r'\w+')
        bigrams = nltk.collocations.BigramCollocationFinder.from_words(
            #get_words(clusters[topic]["comments"])
            " ".join(clusters[topic]["comments"].text.apply(lambda text: " ".join(tokenizer.tokenize(text)))
                     .str.lower().tolist()).split(" ")
        )

        # filtering of bigrams with no intersection with the top K words of the topic
        bigrams.apply_ngram_filter(filter_)
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        # filtering of infrequent bigrams for performance
        top = bigrams.nbest(bigram_measures.raw_freq , 15000)

        if len(top) > 0:
            stemmer = PorterStemmer()
            score = lambda tpl: len(set(tpl).intersection(set(words.split(" ")))) + np.sum(
                [clusters[topic]["twdist"][stemmer.stem(word)] if stemmer.stem(word)
                                                                  in clusters[topic]["twdist"] else 0 for word in tpl]
            )
            scores = pd.DataFrame(columns=["bigram", "score"])

            for i, tpl in enumerate(top):
                scores.at[i, "bigram"] = tpl
                scores.at[i, "score"] = score(tpl)

            labels = scores[scores.score == scores.score.max()]
            clusters[topic]["label"] = " ".join((labels.sample(1) if len(labels) > 1 else labels)["bigram"].iloc[0])

    return clusters

def trigram_labels_topics(clusters):
    """
    Finds the trigram with the largest sum of size of intersection between label and top words under topic-word
    distribution and probability of label words under topic-word distribution.

    :param clusters: dict The topic clusters with keys indicating topic by topic_id. Each subdict contains a key
                          "comments" containing a pandas.DataFrame with the comments, a key "words" with the top
                          words under topic-word distribution and a key "twdist" with a dictionary mapping each
                          word to its likelihood under topic-word distribution.

    :return:         dict The dict of clusters with an additional key "label" containing the label for each topic
                          cluster.
    """
    filter_ = lambda word1, word2, word3: len({word1, word2, word3}.intersection(set(words.split(" ")))) == 0

    for topic in clusters:

        words = clusters[topic]["words"]

        # find bigrams from comments
        tokenizer = RegexpTokenizer(r'\w+')
        trigrams = nltk.collocations.TrigramCollocationFinder.from_words(
            #get_words(clusters[topic]["comments"])
            " ".join(clusters[topic]["comments"].text.apply(lambda text: " ".join(tokenizer.tokenize(text)))
                     .str.lower().tolist()).split(" ")
        )

        # filtering of trigrams with no intersection with the top K words of the topic
        trigrams.apply_ngram_filter(filter_)
        trigram_measures = nltk.collocations.TrigramAssocMeasures()
        # filtering of infrequent trigrams for performance
        top = trigrams.nbest(trigram_measures.raw_freq , 15000)

        if len(top) > 0:
            stemmer = PorterStemmer()
            score = lambda tpl: len(set(tpl).intersection(set(words.split(" ")))) + np.sum(
                [
                    clusters[topic]["twdist"][stemmer.stem(word)]
                    if stemmer.stem(word) in clusters[topic]["twdist"]
                    else 0
                    for word in tpl
                ]
            )
            scores = pd.DataFrame(columns=["trigram", "score"])

            for i, tpl in enumerate(top):
                scores.at[i, "trigram"] = tpl
                scores.at[i, "score"] = score(tpl)

            labels = scores[scores.score == scores.score.max()]
            clusters[topic]["label"] = " ".join((labels.sample(1) if len(labels) > 1 else labels)["trigram"].iloc[0])
                #" ".join(list(scores.loc[scores["score"].astype(int).idxmax()]["trigram"]))

    return clusters

def bigram_labels_wordcount(clusters):
    """
    Finds the most frequent bigram containing the most frequent term in the comments.


    :param clusters: dict    The topic clusters with keys indicating topic by topic_id. Each subdict contains a key
                             "comments" containing a pandas.DataFrame with the comments

    :return:         String  The bigram label.
    """

    filter = lambda word1, word2: top_word not in (word1, word2)

    for topic in clusters:

        top_word = wordcount(clusters[topic]["comments"]).most_common(1)[0][0]

        bigrams = nltk.collocations.BigramCollocationFinder.from_words(
            " ".join(clusters[topic]["comments"].text.tolist()).split(" ")
        )
        bigrams.apply_ngram_filter(filter)
        bigram_measures = nltk.collocations.BigramAssocMeasures()

        top = bigrams.nbest(bigram_measures.raw_freq, 1)

        clusters[topic]["label"] = top

    return clusters


def trigram_labels_wordcount(clusters):
    """
    Finds the most frequent bigram containing the most frequent term in the comments.


    :param clusters: dict   The topic clusters with keys indicating topic by topic_id. Each subdict contains a key
                            "comments" containing a pandas.DataFrame with the comments

    :return:         String The trigram label.
    """
    filter = lambda word1, word2, word3: top_word not in (word1, word2, word3)

    for topic in clusters:
        top_word = wordcount(clusters[topic]["comments"]).most_common(1)[0][0]

        trigrams = nltk.collocations.TrigramCollocationFinder.from_words(
            " ".join(clusters[topic]["comments"].text.tolist()).split(" ")
        )
        trigrams.apply_ngram_filter(filter)
        trigram_measures = nltk.collocations.TrigramAssocMeasures()

        top = trigrams.nbest(trigram_measures.raw_freq, 1)

        clusters[topic]["label"] = top

    return clusters


def most_frequent_bigrams(clusters):
    """
    Finds the two most frequent bigram for each topic cluster.

    :param clusters: dict The topic clusters with keys indicating topic by topic_id. Each subdict contains a key
                          "comments" containing a pandas.DataFrame with the comments

    :return:         dict The dict of clusters with an additional key "label" containing the label for each topic
                          cluster.
    """

    for topic in clusters:

        bigrams = nltk.collocations.BigramCollocationFinder.from_words(
            get_words(clusters[topic]["comments"])
        )
        bigram_measures = nltk.collocations.BigramAssocMeasures()

        top = bigrams.nbest(bigram_measures.raw_freq, 1)

        clusters[topic]["label"] = top

    return clusters


def most_frequent_trigrams(clusters):
    """
    Finds the two most frequent trigram for each topic cluster.

    :param clusters: dict The topic clusters with keys indicating topic by topic_id. Each subdict contains a key
                          "comments" containing a pandas.DataFrame with the comments

    :return:         dict The dict of clusters with an additional key "label" containing the label for each topic
                          cluster.
    """
    for topic in clusters:

        trigrams = nltk.collocations.TrigramCollocationFinder.from_words(
            get_words(clusters[topic]["comments"])
        )
        trigram_measures = nltk.collocations.TrigramAssocMeasures()

        top = trigrams.nbest(trigram_measures.raw_freq, 1)

        clusters[topic]["label"] = top

    return clusters



def wordcount(comments):
    """
    Returns a Counter with the wordcounts of the words in the comments.

    :param comments: pandas.DataFrame The Comments with a "processed" column containing the preprocessed texts.
    :return:         Counter          Contains the wordcounts of the words in the comment.
    """
    return Counter(' '.join(comments.processed.tolist()).split(' '))