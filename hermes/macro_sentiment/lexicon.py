"""Context-aware hawkish/dovish lexicon scorer (PROJECT_PLAN.md 29.3).

Hawkish/dovish is not generic sentiment. "Price pressures are rising" is
hawkish; how much depends on whether inflation sits above or below target.
The score takes that context flag explicitly.

Scores are in [-100, +100]: +100 = maximally hawkish, -100 = maximally dovish.
Deterministic by construction so it can serve as the no-LLM fallback and as a
sanity anchor for LLM agents.
"""

import re

# phrase -> base hawkishness weight
HAWKISH_PHRASES = {
    "price pressures are rising": 18.0,
    "price pressures": 10.0,
    "inflation remains elevated": 20.0,
    "inflation is too high": 22.0,
    "upside risks to inflation": 18.0,
    "further tightening": 25.0,
    "restrictive stance": 18.0,
    "restrictive": 12.0,
    "raise rates": 22.0,
    "rate increases": 18.0,
    "hike": 15.0,
    "above target": 12.0,
    "strong labor market": 10.0,
    "wage pressures": 12.0,
    "higher for longer": 20.0,
    "vigilant": 8.0,
    "tightening": 12.0,
}

DOVISH_PHRASES = {
    "downside risks to growth": -18.0,
    "economic activity has weakened": -15.0,
    "rate cuts": -22.0,
    "cut rates": -22.0,
    "lower rates": -18.0,
    "accommodative": -18.0,
    "easing": -15.0,
    "below target": -12.0,
    "disinflation": -14.0,
    "inflation has declined": -14.0,
    "labor market has cooled": -12.0,
    "patient": -8.0,
    "gradual normalization": -10.0,
    "stimulus": -16.0,
    "quantitative easing": -20.0,
}

# Phrases whose hawkish weight is conditional on the inflation regime:
# rising prices when inflation is already above target reads much more
# hawkish than the same words below target.
INFLATION_CONTEXT_PHRASES = {
    "price pressures are rising",
    "price pressures",
    "inflation remains elevated",
    "wage pressures",
}

ABOVE_TARGET_MULTIPLIER = 1.5
BELOW_TARGET_MULTIPLIER = 0.5


def score_text(text: str, inflation_above_target: bool | None = None) -> float:
    """Hawkish-dovish score in [-100, 100] for a policy text."""
    lowered = re.sub(r"\s+", " ", text.lower())
    total = 0.0
    for table in (HAWKISH_PHRASES, DOVISH_PHRASES):
        for phrase, weight in table.items():
            count = lowered.count(phrase)
            if count == 0:
                continue
            w = weight
            if phrase in INFLATION_CONTEXT_PHRASES and inflation_above_target is not None:
                w *= ABOVE_TARGET_MULTIPLIER if inflation_above_target else BELOW_TARGET_MULTIPLIER
            total += w * count
    return float(max(-100.0, min(100.0, total)))
