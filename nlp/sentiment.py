"""News sentiment scoring with VADER.

VADER is lexicon-based, needs no training/API key and is fast enough to run
on every article as it's collected. It's tuned for short, informal text
(headlines, tweets) which matches crypto news titles well. Swap in an LLM
call here later if you need nuance VADER can't capture (irony, complex
financial jargon) - the rest of the pipeline only depends on the
{"compound", "label"} contract below.
"""
from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def score_text(text: str) -> dict:
    """Return {"compound": -1..1, "positive"/"negative"/"neutral" sub-scores, "label"}."""
    if not text:
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "label": "neutral"}
    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= POSITIVE_THRESHOLD:
        label = "positive"
    elif compound <= NEGATIVE_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"
    return {
        "compound": compound,
        "pos": scores["pos"],
        "neu": scores["neu"],
        "neg": scores["neg"],
        "label": label,
    }


def score_article(article: dict) -> dict:
    """Score an article dict (expects `title`, optionally `summary`) in place, non-destructively."""
    text = article.get("title", "")
    if article.get("summary"):
        text = f"{text}. {article['summary']}"
    sentiment = score_text(text)
    return {**article, "sentiment": sentiment}


def aggregate_daily(scored_articles: list[dict]) -> dict:
    """Average compound sentiment per calendar date -> {date: float}.

    Used as the `sentiment_by_date` feature input for ml.features.merge_sentiment.
    """
    from collections import defaultdict

    buckets: dict = defaultdict(list)
    for article in scored_articles:
        published = article.get("published_at")
        if published is None:
            continue
        day = published.date() if hasattr(published, "date") else published
        buckets[day].append(article["sentiment"]["compound"])
    return {day: sum(scores) / len(scores) for day, scores in buckets.items()}
