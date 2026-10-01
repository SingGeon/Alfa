"""Combine the technical price forecast with the news-sentiment signal.

The sentiment score is already folded into the model as a feature
(see ml.features / ml.price_predictor), so "combining" here means:
  1. computing a confidence score that reflects both the model's own
     uncertainty (interval width) AND whether sentiment agrees with the
     predicted direction, and
  2. optionally asking Claude for a short narrative explaining the call
     in plain language, using the numbers as grounding (falls back to a
     rule-based template if no ANTHROPIC_API_KEY is configured, so this
     always returns something).
"""
from __future__ import annotations

import logging

import config

logger = logging.getLogger(__name__)


def compute_confidence(predictions: list[dict], sentiment_avg: float) -> float:
    """Confidence score in [0, 100].

    - Narrower confidence interval (relative to price) -> higher confidence.
    - Sentiment agreeing with the predicted direction -> small boost.
    - Sentiment disagreeing -> small penalty.
    """
    if not predictions:
        return 0.0

    first_price = predictions[0]["predicted_price"]
    last = predictions[-1]
    interval_width_pct = (last["upper"] - last["lower"]) / max(last["predicted_price"], 1e-9)
    # Interval width as a fraction of price, scaled so ~20% width -> 0 confidence.
    base_confidence = min(100.0, max(0.0, 100.0 - interval_width_pct * 500))

    predicted_direction = 1 if last["predicted_price"] >= first_price else -1
    sentiment_direction = 1 if sentiment_avg > 0.05 else (-1 if sentiment_avg < -0.05 else 0)

    if sentiment_direction == 0:
        agreement_adjustment = 0.0
    elif sentiment_direction == predicted_direction:
        agreement_adjustment = 5.0
    else:
        agreement_adjustment = -10.0

    return round(min(100.0, max(0.0, base_confidence + agreement_adjustment)), 1)


def _template_narrative(
    last_known_price: float, predicted_price: float, sentiment_avg: float, headlines: list[str], lang: str = "en",
) -> str:
    """Rule-based fallback narrative, used when no Claude API key is
    configured. Pure string formatting on numbers already computed by the
    caller - no re-running of any model, so a `lang` toggle is instant.
    """
    pct = abs(predicted_price - last_known_price) / last_known_price * 100 if last_known_price else 0.0
    # Same 2-decimal rounding as the outlook tile's change_pct: a move that
    # shows there as 0.00% must not read "will fall by about 0.00%" here.
    flat = round(pct, 2) == 0
    if lang == "ro":
        direction = "creasca" if predicted_price >= last_known_price else "scada"
        mood = "pozitiv" if sentiment_avg > 0.05 else ("negativ" if sentiment_avg < -0.05 else "neutru")
        move = ("ramana aproximativ la acelasi nivel maine" if flat
                else f"{direction} cu aproximativ {pct:.2f}% maine")
        lines = [
            f"Modelul tehnic estimeaza ca pretul ETH va {move} "
            f"fata de pretul curent, pornind de la tendinta recenta a preturilor istorice.",
            f"Sentimentul din stiri este in medie {mood} (scor {sentiment_avg:.2f}), pe baza celor "
            f"{len(headlines)} articole recente analizate.",
        ]
        if headlines:
            lines.append("Titluri recente relevante: " + "; ".join(headlines[:3]))
        return " ".join(lines)

    direction = "rise" if predicted_price >= last_known_price else "fall"
    mood = "positive" if sentiment_avg > 0.05 else ("negative" if sentiment_avg < -0.05 else "neutral")
    move = "stay about where it is" if flat else f"{direction} by about {pct:.2f}%"
    lines = [
        f"The technical model estimates ETH's price will {move} "
        f"tomorrow relative to the current price, based on the recent trend in historical prices.",
        f"News sentiment is on average {mood} (score {sentiment_avg:.2f}), based on the "
        f"{len(headlines)} recent articles analyzed.",
    ]
    if headlines:
        lines.append("Relevant recent headlines: " + "; ".join(headlines[:3]))
    return " ".join(lines)


def generate_narrative_summary(
    last_known_price: float, predicted_price: float, sentiment_avg: float, recent_articles: list[dict],
    lang: str = "en",
) -> str:
    """Daily narrative summary ("why the model thinks tomorrow's price will move").

    Takes the same single next-day (interval="1d", steps=1) prediction as
    the dashboard's "24h prediction" outlook tile (see api.services.
    run_combined_summary) - not a multi-step trend - so the two can never
    contradict each other the way they could before, when this narrative
    was built from whichever interval the main chart happened to be
    showing (a 24-*hour* trend at "1h", a nonsensical 24-*day* one if the
    chart was on "1d"/"1w") while the tile was always about tomorrow.

    Uses Claude when ANTHROPIC_API_KEY is set, otherwise a deterministic
    rule-based template so this endpoint never hard-fails. `lang` ("en" or
    "ro") is pure output-language selection over the same already-computed
    numbers - never triggers a re-prediction.
    """
    headlines = [a["title"] for a in recent_articles[:5]]

    if not config.ANTHROPIC_API_KEY:
        return _template_narrative(last_known_price, predicted_price, sentiment_avg, headlines, lang)

    try:
        import anthropic
    except ImportError:
        logger.warning("ANTHROPIC_API_KEY set but `anthropic` package not installed; using template narrative")
        return _template_narrative(last_known_price, predicted_price, sentiment_avg, headlines, lang)

    language_name = "Romanian" if lang == "ro" else "English"
    prompt = (
        f"You are a crypto analyst. Write a short narrative summary (3-4 sentences, in {language_name}) "
        "explaining why a prediction model might anticipate Ethereum's price movement tomorrow, "
        "based on the following data:\n\n"
        f"- Current price: {last_known_price:.2f} USD\n"
        f"- Predicted price tomorrow: {predicted_price:.2f} USD\n"
        f"- Average news sentiment score (-1..1): {sentiment_avg:.2f}\n"
        f"- Recent headlines: {'; '.join(headlines) if headlines else 'none available'}\n\n"
        "Be concise, don't give financial advice, and explicitly mention that this is an informational estimate."
    )
    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as exc:  # network/auth errors shouldn't break the dashboard
        logger.warning("Claude narrative generation failed (%s); using template narrative", exc)
        return _template_narrative(last_known_price, predicted_price, sentiment_avg, headlines, lang)
