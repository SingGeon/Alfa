"""Turns a prediction series + sentiment into a plain-language "why and
when" explanation for the per-asset detail page - the same kind of
template ml/combined_predictor.py writes for ETH, generalized to name a
specific peak date/price for any asset.
"""
from __future__ import annotations


def generate_asset_narrative(
    name: str,
    symbol: str,
    last_price: float,
    predictions: list[dict],
    sentiment_avg: float,
    scored_articles: list[dict],
    lang: str = "en",
) -> str:
    """Pure string formatting over already-computed prediction/sentiment
    numbers - `lang` ("en" or "ro") only picks the sentence templates
    below, it never re-runs the model.
    """
    if not predictions:
        return "Insufficient data for a prediction." if lang != "ro" else "Date insuficiente pentru o predicție."

    peak = max(predictions, key=lambda p: p["predicted_price"])
    trough = min(predictions, key=lambda p: p["predicted_price"])
    final = predictions[-1]

    peak_pct = (peak["predicted_price"] - last_price) / last_price * 100 if last_price else 0.0
    final_pct = (final["predicted_price"] - last_price) / last_price * 100 if last_price else 0.0

    headlines = [a["title"] for a in scored_articles[:3] if a.get("title")]

    if lang == "ro":
        peak_date = peak["timestamp"].strftime("%d %B")
        direction = "crească" if final_pct >= 0 else "scadă"
        mood = "pozitiv" if sentiment_avg > 0.05 else ("negativ" if sentiment_avg < -0.05 else "neutru")

        lines = [
            f"Modelul tehnic estimează că {name} ({symbol}) ar putea {direction} cu aproximativ "
            f"{abs(final_pct):.1f}% pe orizontul de {len(predictions)} zile analizat, pornind de la "
            f"tendința recentă a prețului.",
            f"Punctul maxim estimat este în jur de {peak_date}, la aproximativ "
            f"${peak['predicted_price']:,.4f} ({'+' if peak_pct >= 0 else ''}{peak_pct:.1f}% față de prețul curent "
            f"de ${last_price:,.4f}).",
        ]

        if trough is not peak:
            trough_pct = (trough["predicted_price"] - last_price) / last_price * 100 if last_price else 0.0
            lines.append(
                f"Cel mai slab punct estimat pe acest orizont este în jur de "
                f"{trough['timestamp'].strftime('%d %B')}, la aproximativ ${trough['predicted_price']:,.4f} "
                f"({'+' if trough_pct >= 0 else ''}{trough_pct:.1f}%)."
            )

        sentiment_line = f"Sentimentul din știrile recente este în medie {mood} (scor {sentiment_avg:.2f})"
        if headlines:
            sentiment_line += ", susținut de titluri precum: " + "; ".join(headlines)
        lines.append(sentiment_line + ".")

        lines.append(
            "Aceasta este o estimare informativă generată automat din date istorice și sentiment de știri - "
            "nu reprezintă sfat financiar și nu garantează evoluția reală a prețului."
        )
        return " ".join(lines)

    peak_date = peak["timestamp"].strftime("%B %d")
    direction = "rise" if final_pct >= 0 else "fall"
    mood = "positive" if sentiment_avg > 0.05 else ("negative" if sentiment_avg < -0.05 else "neutral")

    lines = [
        f"The technical model estimates {name} ({symbol}) could {direction} by about "
        f"{abs(final_pct):.1f}% over the {len(predictions)}-day horizon analyzed, based on the "
        f"recent price trend.",
        f"The estimated peak is around {peak_date}, at approximately "
        f"${peak['predicted_price']:,.4f} ({'+' if peak_pct >= 0 else ''}{peak_pct:.1f}% vs. the current price "
        f"of ${last_price:,.4f}).",
    ]

    if trough is not peak:
        trough_pct = (trough["predicted_price"] - last_price) / last_price * 100 if last_price else 0.0
        lines.append(
            f"The weakest estimated point over this horizon is around "
            f"{trough['timestamp'].strftime('%B %d')}, at approximately ${trough['predicted_price']:,.4f} "
            f"({'+' if trough_pct >= 0 else ''}{trough_pct:.1f}%)."
        )

    sentiment_line = f"Recent news sentiment is on average {mood} (score {sentiment_avg:.2f})"
    if headlines:
        sentiment_line += ", supported by headlines such as: " + "; ".join(headlines)
    lines.append(sentiment_line + ".")

    lines.append(
        "This is an informational estimate generated automatically from historical data and news sentiment - "
        "it is not financial advice and does not guarantee the actual price movement."
    )
    return " ".join(lines)
