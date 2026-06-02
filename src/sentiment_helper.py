from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def detect_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"


def get_sentiment_prefix(sentiment):
    prefixes = {
        "positive": "😊 We're glad you're having a great experience! ",
        "negative": "😔 We're sorry to hear you're having trouble. We're here to help. ",
        "neutral": ""
    }
    return prefixes[sentiment]
