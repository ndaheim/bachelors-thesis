from nltk.sentiment.vader import SentimentIntensityAnalyzer

def polarity(comments):
    """
:    Calculates a polarity score for each of the given comments.

    :param comments: pandas.DataFrame Set of comments.

    :return:         pandas.DataFrame Contains an additional column "polarity" with a sentiment score for each comment.
    """
    sid = SentimentIntensityAnalyzer()
    comments["polarity"] = comments.text.map(lambda c: sid.polarity_scores(c)["compound"])
    return comments
