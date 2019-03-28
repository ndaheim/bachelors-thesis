import datetime

import pandas as pd

"""
Contains functions to write the meta.txt as required by the HMDP implementation of the Promoss package.
See https://github.com/ckling/promoss for more info.
"""


def get_and_write_metadata(articles, comments, doc_id=True, thread_id=True, timestamp=True):
    """
    Writes the meta.txt file as is required for the Hierarchical Multi-Dirichlet Process (HMDP) implementation
    provided by Promoss.
    See https://github.com/ckling/promoss for further information.

    :param articles:  String  The path to the file containing the articles.
    :param comments:  String  The path to the file containing the comments.
    :param doc_id:    boolean Whether a document_id identifying the article a comment was issued under shall be included
                              in the metadata.
    :param thread_id: boolean Whether a thread_id identifying the the comment thread a comment was issued in shall be
                              included the metadata.
    :param timestamp: boolean Whether a timestamp shall be included in the metadata.

    :return: (pandas.DataFrame, pandas.DataFrame) A tuple containing the complete article and comment data.
    """
    if not (doc_id or timestamp or thread_id):
        raise Exception("At least one type of metadata has to be specified")

    meta = []

    if timestamp:
        timestamps = get_timestamps(articles, comments)
        meta.append(timestamps)

    if doc_id:
        doc_ids = get_doc_identifier(articles, comments)
        meta.append(doc_ids)
        articles['doc_id'] = doc_ids.head(len(articles))
        comments['doc_id'] = doc_ids.tail(len(comments))

    if thread_id:
        thread_ids = get_thread_ids(articles, comments)
        meta.append(thread_ids)

    df = pd.concat(meta, axis=1)
    df.to_csv('meta.txt', sep=';', index=None, header=None)

    return articles,comments


def get_thread_ids(articles, comments):
    """
    Returns a pandas.Series containing the thread ids of comments.

    :param articles: pandas.DataFrame The article information.
    :param comments: pandas.DataFrame The comment information.

    :return:         pandas.Series    with an id of each article followed by the thread_ids of the comments.
    """
    return pd.Series(range(len(articles))).append(comments.sdid)


def get_timestamps(articles, comments):
    """
    Returns a pandas.Series containing the timestamps of the article and comment set.

    :param articles: pandas.DataFrame The article information.
    :param comments: pandas.DataFrame The comment information.

    :return:         pandas.Series    Timestamps of the articles followed by the timestamps of the comments.
    """
    series = articles.timestamp.append(comments.timestamp)
    return series.astype(int)


def get_doc_identifier(articles, comments):
    """
    Returns a pandas.Series containing the numeric article identifier of the article and comment set. Articles have
    their own identifier and comments the identifier of the article they were issued under.

    :param articles: pandas.DataFrame The article information.
    :param comments: pandas.DataFrame The comment information.

    :return:         pandas.Series    Identifier of the articles followed by the timestamps of the comments.
    """
    ids = articles.url.to_dict()
    ids = {v:k for k,v in ids.items()}

    comments["doc_id"] = comments.url.copy().map(lambda url: ids[url])
    return pd.Series(articles.index.values).append(comments["doc_id"])


def get_timestamp_for_articles(df):
    """
    Adds a timestamp column to the pandas.DataFrame containing the article set.

    :param df: pandas.DataFrame The article information.

    :return:   pandas.DataFrame The article information with an additional timestamp column.
    """
    df['timestamp'] = df['date'].copy().map(lambda dt: datetime.datetime.strptime(dt, "%m-%d-%Y").timestamp())