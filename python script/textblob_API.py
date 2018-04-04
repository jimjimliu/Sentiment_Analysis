from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


def get_score(string):
    blob = TextBlob(string, analyzer = NaiveBayesAnalyzer())
    result = []
    result.append(blob.sentiment[1])
    result.append(blob.sentiment[2])
    return result





