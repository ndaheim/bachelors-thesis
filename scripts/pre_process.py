import pandas as pd
from nltk.stem import *
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


"""
Contains functions for the used text preprocessing.
"""


def pre_process(text):
    """
    Preprocesses a given document by removing apostrophes, tokenizing, which includes the removal of punctuation,
    lowercasing and stemming with the PorterStemmer.

    :param text: String The text which should be pre processed.
    :return:     String The pre processed text in the form a whitespace delimited String containing the processed words
                        in original order.
    """
    text = stem(
            remove_stop_words(
                to_lower(
                    tokenize(
                        replace_apostrophe(text)
                    )
                )
            )
    )
    return ' '.join(text)


def stem(tokens):
    """
    Stems the given token with the nltk.stem.PorterStemmer.

    :param tokens: String   The words which should be stemmed.
    :return:       [String] A List containing the stemmed words.
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def tokenize(text):
    """
    Tokenizes the text and removes punctuation.

    :param text: String The text which should be tokenized
    :return:     String The tokenized text.
    """
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(text)


def replace_apostrophe(text):
    """
    Removes apostrophes from the given text.

    :param text: String The text, for which apostrophes should be removed.
    :return:     String The original text with removed apostrophes, eg. isn't results in isnt.
    """
    return text.replace('\'', '')


def to_lower(text):
    """
    Returns a list of the lowercased words.

    :param text: String   The text to be put in lower case.
    :return:     [String] list of the lowercased words.
    """
    return [token.lower() for token in text]


def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    return [token for token in text if not token in stop_words]