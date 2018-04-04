
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_score(string):
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(string)
    result = []
    result.append(ss['neg'])
    result.append(ss['neu'])
    result.append(ss['pos'])
    result.append(ss['compound'])
    return result
