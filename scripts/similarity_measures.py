import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""
Contains a number of similarity measures.
"""


def tf_idf_vectorize(comments, vectorizer=None):
    """
    Transforms a set of comments to a TF-IDF Matrix.
    
    :param comments:   Iterable A set of comments.
    :param vectorizer:          Vectorizer for document transformation. If none is given,
                                sklearn.feature_extraction.text.TfidfVectorizer is used.

    :return:                    TF-IDF Matrix.
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(comments)
    else:
        return vectorizer.transform(comments)


def tf_vectorize(comments, vectorizer=None):
    """
    Transforms a set of comments to a TF Matrix.
    
    :param comments:   Iterable A set of comments.
    :param vectorizer:          Vectorizer for document transformation. If none is given,
                                sklearn.feature_extraction.text.CountVectorizer is used.

    :return:                    TF-IDF Matrix.
    """
    if vectorizer is None:
        vectorizer = CountVectorizer()
        return vectorizer.fit_transform(comments)
    else:
        return vectorizer.fit(comments)


def get_cosine_similarity(v, w):
    """
    Returns the cosine similarity of two vectors.

    :param v: [float] A vector of floating numbers, usually representing a document vector.
    :param w: [float] A vector of floating numbers, usually representing a document vector.

    :return: float    Cosine similarity of v and w.
    """
    v = np.array(v)
    w = np.array(w)

    return np.dot(v, w) / (np.sqrt(np.sum(v ** 2)) * np.sqrt(np.sum(w ** 2)))


def get_cosine_distance(v, w):
    """
    Returns the cosine distance of two vectors.

    :param v: [float] A vector of floating numbers, usually representing a document vector.
    :param w: [float] A vector of floating numbers, usually representing a document vector.

    :return: float    Cosine distance of v and w.
    """
    return 1 - cosine_similarity(v, w)


def get_cosine_similarity_pairwise(M):
    """
    Returns the pairwise cosine similarity of elements of a Matrix containing vector representations of documents.
    Each row and column represent a document according to the inputs ordering.

    :param M:      Matrix containing either vector representations of documents.

    :return: array An array of cosine similarities.
    """
    return cosine_similarity(M)


def get_cosine_distance_pairwise(M):
    """
    Returns the pairwise cosine distance of elements of a Matrix containing vector representations of documents.
    Each row and column represent a document according to the inputs ordering.

    :param M:      Matrix containing either vector representations of documents.

    :return: array An array of cosine distances.
    """
    return 1 - cosine_similarity(M)


def cosine_modified(c1, c2, is_set=False):
    """
    Returns the modified cosine similarity of two comments c1, c2 according to Aker et al.
    "A Graph-based Approach to Topic Clustering for Online Comments to News" (2016).

    :param c1:     String|Set  Text of first comment or set containing its words.
    :param c2:     String|Set  Text of second comment or set containing its words.
    :param is_set: boolean     True if comments are already in set form, false otherwise.

    :return:       1 if c1 and 2 share 5 terms, |shared terms|/5 else
    """
    # pre-processing can result in empty comments, e.g. when a comment only has stopwords.
    if len(c1) == 0 and len(c2) == 0:
        return 0

    if not is_set:
        c1 = set(c1.split())
        c2 = set(c2.split())

    d = len(c1.intersection(c2))

    return 1 if d > 5 else d/5


def dice(c1, c2, is_set=False):
    """
    Returns the dice similarity of two comments c1, c2.

    :param c1:     String|Set  Text of first comment or set containing its words.
    :param c2:     String|Set  Text of second comment or set containing its words.
    :param is_set: boolean     True if comments are already in set form, false otherwise.

    :return:       float       Dice similarity of c1 and c2.
    """
    # pre-processing can result in empty comments, e.g. when a comment only has stopwords.
    if len(c1) == 0 and len(c2) == 0:
        return 0

    if not is_set:
        c1 = set(c1.split())
        c2 = set(c2.split())

    return 2*len(c1.intersection(c2))/(len(c1)+len(c2))


def jaccard(c1, c2, is_set=False):
    """
    Returns the Jaccard similarity index of two comments c1, c2.

    :param c1:     String|Set  Text of first comment or set containing its words.
    :param c2:     String|Set  Text of second comment or set containing its words.
    :param is_set: boolean     True if comments are already in set form, false otherwise.

    :return:       float       Jaccard similarity of c1 and c2.
    """
    # pre-processing can result in empty comments, e.g. when a comment only has stopwords.
    if len(c1) == 0 and len(c2) == 0:
        return 0

    if not is_set:
        c1 = set(c1.split())
        c2 = set(c2.split())

    return len(c1.intersection(c2)) / len(c1.union(c2))


def same_thread(c1, c2):
    """
    Returns whether two comments stem from the same thread.

    :param c1: pandas.DataFrame Row of the first comment.
    :param c2: pandas.DataFrame Row of the second comment.

    :return:   int              1 if c1 and c2 are from the same thread,
                                0 if c1 and c2 are from different threads.
    """
    if c1.sdid == c2.sdid:
        return 1
    else:
        return 0


def substitute_to_unit_interval(similarities):
    """
    Scales the similarity scores of the given pandas.Series from the interval [a,b] to the unit interval [0,1]
    by the substitution s = (t-a)/(b-a) for any given similarity score t.


    :param similarities: pandas.Series Containing floats.

    :return:             pandas.Series Of scaled floats.
    """
    upper_bound = similarities.max()
    lower_bound = similarities.min()
    return similarities.apply(lambda score: (score-lower_bound)/(upper_bound-lower_bound))


def ne_overlap(ne_c1, ne_c2, chunked=False):
    """
    Measures the size of the set of overlapping named entities between the first and second comment.


    :param ne_c1: pandas.DataFrame containing the first comment.
    :param ne_c2: pandas.DataFrame containing the second comment.

    :return:      float            Named Entity Overlap of the two comments.
    """
    if not chunked:
        ne_c1 = set(named_entities(ne_c1.text))
        ne_c2 = set(named_entities(ne_c2.text))

    if len(ne_c1) == 0 and len(ne_c2) == 0:
        return 0

    return len((ne_c1.intersection(ne_c2)))/len((ne_c1.union(ne_c2)))



def named_entities(c):
    """
    Returns a list of named entities for a given comment.
    See https://stackoverflow.com/questions/31836058/nltk-named-entity-recognition-to-a-python-list.


    :param c: String   The comment text.
    :return:  [String] List of named entities of the comment.
    """
    chunked = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(c)))

    continuous_chunk = []
    current_chunk = []

    for i in chunked:
        if type(i) == nltk.tree.Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return continuous_chunk