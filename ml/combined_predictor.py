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


def _template_narrative(predictions: list[dict], sentiment_avg: float, headlines: list[str]) -> str:
    """Rule-based fallback narrative, used when no Claude API key is configured."""
    if not predictions:
        return "Date insuficiente pentru o predictie."
    first, last = predictions[0]["predicted_price"], predictions[-1]["predicted_price"]
    direction = "creasca" if last >= first else "scada"
    pct = abs(last - first) / first * 100
    mood = "pozitiv" if sentiment_avg > 0.05 else ("negativ" if sentiment_avg < -0.05 else "neutru")
    lines = [
        f"Modelul tehnic estimeaza ca pretul ETH va {direction} cu aproximativ {pct:.2f}% "
        f"pe orizontul analizat, pornind de la tendinta recenta a preturilor istorice.",
        f"Sentimentul din stiri este in medie {mood} (scor {sentiment_avg:.2f}), pe baza celor "
        f"{len(headlines)} articole recente analizate.",
    ]
    if headlines:
        lines.append("Titluri recente relevante: " + "; ".join(headlines[:3]))
    return " ".join(lines)


def generate_narrative_summary(predictions: list[dict], sentiment_avg: float, recent_articles: list[dict]) -> str:
    """Daily narrative summary ("why the model thinks price will move").

    Uses Claude when ANTHROPIC_API_KEY is set, otherwise a deterministic
    rule-based template so this endpoint never hard-fails.
    """
    headlines = [a["title"] for a in recent_articles[:5]]

    if not config.ANTHROPIC_API_KEY:
        return _template_narrative(predictions, sentiment_avg, headlines)

    try:
        import anthropic
    except ImportError:
        logger.warning("ANTHROPIC_API_KEY set but `anthropic` package not installed; using template narrative")
        return _template_narrative(predictions, sentiment_avg, headlines)

    if not predictions:
        return _template_narrative(predictions, sentiment_avg, headlines)

    first, last = predictions[0]["predicted_price"], predictions[-1]["predicted_price"]
    prompt = (
        "Esti un analist crypto. Scrie un rezumat narativ scurt (3-4 propozitii, in limba romana) "
        "care explica de ce un model de predictie ar putea anticipa evolutia pretului Ethereum, "
        "pe baza urmatoarelor date:\n\n"
        f"- Pret prezis la inceputul orizontului: {first:.2f} USD\n"
        f"- Pret prezis la finalul orizontului: {last:.2f} USD\n"
        f"- Scor mediu de sentiment din stiri (-1..1): {sentiment_avg:.2f}\n"
        f"- Titluri recente: {'; '.join(headlines) if headlines else 'niciunul disponibil'}\n\n"
        "Fii concis, nu da sfaturi financiare, mentioneaza explicit ca este o estimare informativa."
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
        return _template_narrative(predictions, sentiment_avg, headlines)
