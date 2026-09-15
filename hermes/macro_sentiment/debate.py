"""Multi-agent hawk/dove debate for central-bank texts (PROJECT_PLAN.md 29.3).

Three agents with distinct priors (dove / neutral / hawk) each score the
text, then iterate debate rounds: every round each agent moves partway toward
the group mean while keeping a shrinking pull toward its prior. This mirrors
the consensus dynamic of debate-based LLM classification.

With an LLM client (local Ollama), each agent's initial read comes from the
model. Without one, the deterministic lexicon supplies it, so tests and
offline runs are reproducible.

Output is bias only - SentimentConsensus.trade_action is hard-typed to
"bias_only_no_direct_trade" and this module has no access to brokers.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from app.schemas.sentiment import AgentStance, SentimentConsensus
from hermes.macro_sentiment.lexicon import score_text

PRIOR_OFFSETS = {"dove": -25.0, "neutral": 0.0, "hawk": 25.0}
CONVERGENCE_RATE = 0.5  # pull toward group mean per round
PRIOR_DECAY = 0.5  # prior influence halves each round

AGENT_PROMPT = """You are a monetary-policy analyst with a {prior} prior.
Score the following central bank text for hawkishness from -100 (maximally
dovish) to +100 (maximally hawkish). Inflation is currently {inflation_state}
target. Respond with a single integer only.

TEXT:
{text}
"""


@dataclass
class DebateAgent:
    name: str
    prior: str  # dove | neutral | hawk

    def initial_score(
        self, text: str, inflation_above_target: bool | None, llm_client
    ) -> float:
        base = None
        if llm_client is not None:
            state = (
                "above" if inflation_above_target else "below or near"
            ) if inflation_above_target is not None else "at an unknown level versus"
            response = llm_client.generate(
                AGENT_PROMPT.format(prior=self.prior, inflation_state=state, text=text[:4000])
            )
            base = _parse_score(response)
        if base is None:
            base = score_text(text, inflation_above_target)
        return _clamp(base + PRIOR_OFFSETS[self.prior])


def _parse_score(response: str | None) -> float | None:
    if not response:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", response)
    if not match:
        return None
    return _clamp(float(match.group()))


def _clamp(v: float) -> float:
    return max(-100.0, min(100.0, v))


def run_debate(
    text: str,
    currency: str,
    inflation_above_target: bool | None = None,
    rounds: int = 3,
    llm_client=None,
) -> SentimentConsensus:
    agents = [
        DebateAgent("agent_dove", "dove"),
        DebateAgent("agent_neutral", "neutral"),
        DebateAgent("agent_hawk", "hawk"),
    ]
    scores = {
        a.name: a.initial_score(text, inflation_above_target, llm_client) for a in agents
    }

    prior_weight = 1.0
    for _ in range(rounds):
        mean = sum(scores.values()) / len(scores)
        prior_weight *= PRIOR_DECAY
        for agent in agents:
            pull_to_mean = CONVERGENCE_RATE * (mean - scores[agent.name])
            pull_to_prior = prior_weight * PRIOR_OFFSETS[agent.prior] * 0.2
            scores[agent.name] = _clamp(scores[agent.name] + pull_to_mean + pull_to_prior)

    consensus = sum(scores.values()) / len(scores)
    spread = max(scores.values()) - min(scores.values())
    confidence = _clamp(100.0 - spread) / 100.0  # tight consensus -> high confidence

    if consensus >= 15:
        bias = "bullish"  # hawkish -> bullish the currency
    elif consensus <= -15:
        bias = "bearish"
    else:
        bias = "neutral"

    return SentimentConsensus(
        timestamp=datetime.now(timezone.utc),
        currency=currency,
        macro_bias=bias,
        hawkish_dovish_score=round(consensus, 2),
        confidence=round(confidence, 3),
        reason=(
            f"3-agent debate over {rounds} rounds; final stances "
            + ", ".join(f"{name}={s:+.0f}" for name, s in scores.items())
            + f"; spread {spread:.0f}"
        ),
        agent_stances=[
            AgentStance(agent=a.name, prior=a.prior, score=round(scores[a.name], 2))
            for a in agents
        ],
        debate_rounds=rounds,
        inflation_above_target=inflation_above_target,
    )
