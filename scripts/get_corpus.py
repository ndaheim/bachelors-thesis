import pandas as pd

from pre_process import *

"""
Contains functions to write the corpus.txt as required by the LDA and HMDP implementations of the Promoss package.
See https://github.com/ckling/promoss for more info.
"""


def get_and_write_corpus(df_articles, df_comments):
    """
    Transforms the input data into the structure required by the Promoss package, writes it into a corpus.txt file and
    returns the complete data as a pandas.DataFrame.
    See https://github.com/ckling/promoss for more info.

    :param articles: String The location of the tab-separated file containing the articles, which must contain a header
                            row and a text column containing the raw texts.
    :param comments: String The location of the tab-separated file containing the comments, which must contain a header
                            row and a text column containing the raw texts.

    :return: (pandas.DataFrame, pandas.DataFrame) A tuple containing the complete article and comment data.
    """

    processed_articles = get_preprocessed_texts(df_articles)
    processed_comments = get_preprocessed_texts(df_comments)

    df_articles['processed'] = processed_articles
    df_comments['processed'] = processed_comments

    series = processed_articles.append(processed_comments)
    to_txt(series)
    return df_articles, df_comments


def get_preprocessed_texts(df):
    """
    Returns a series with the pre processed documents.

    :param location: pandas.DataFrame Containing the texts in the "text" column.
    :return:         pandas.Series    with the pre processed documents.
    """
    return df.text.copy().map(lambda text: pre_process(text))


def to_txt(series):
    """
    Writes the series to a txt file as required for the Promoss package.
    See https://github.com/ckling/promoss for mor info.

    :param series: The series which should be written into a file.
    :return:
    """
    return series.to_csv('corpus.txt', sep='\n', index=False)


if __name__ == '__main__':
    get_corpus()