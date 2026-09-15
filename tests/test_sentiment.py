from hermes.macro_sentiment.debate import run_debate
from hermes.macro_sentiment.lexicon import score_text
from hermes.macro_sentiment.retrieval import TfidfRetriever

HAWKISH_TEXT = (
    "Inflation remains elevated and price pressures are rising. The committee "
    "judges that further tightening may be appropriate and intends to keep a "
    "restrictive stance with rates higher for longer given wage pressures."
)
DOVISH_TEXT = (
    "Economic activity has weakened and the labor market has cooled. With "
    "disinflation underway and inflation below target, the committee will "
    "consider rate cuts and a more accommodative policy with gradual easing."
)


def test_lexicon_signs():
    assert score_text(HAWKISH_TEXT) > 30
    assert score_text(DOVISH_TEXT) < -30


def test_inflation_context_changes_hawkishness():
    # short text so the score does not clamp at +/-100
    text = "Price pressures are rising."
    above = score_text(text, inflation_above_target=True)
    below = score_text(text, inflation_above_target=False)
    assert above > below  # same words, more hawkish when inflation above target
    assert above > 0 and below > 0  # still hawkish either way


def test_debate_consensus_deterministic_and_bias_only():
    a = run_debate(HAWKISH_TEXT, currency="USD", inflation_above_target=True)
    b = run_debate(HAWKISH_TEXT, currency="USD", inflation_above_target=True)
    assert a.hawkish_dovish_score == b.hawkish_dovish_score
    assert a.trade_action == "bias_only_no_direct_trade"
    assert a.macro_bias == "bullish"  # hawkish -> bullish currency
    assert len(a.agent_stances) == 3
    assert 0.0 <= a.confidence <= 1.0


def test_debate_converges():
    result = run_debate(DOVISH_TEXT, currency="EUR", rounds=4)
    scores = [s.score for s in result.agent_stances]
    assert max(scores) - min(scores) < 30  # agents pulled toward consensus
    assert result.macro_bias == "bearish"


def test_retriever_finds_relevant_chunk(tmp_path):
    (tmp_path / "fomc.txt").write_text(
        "The committee discussed inflation dynamics and the policy rate path.\n\n"
        "Members noted that housing markets remain stable.",
        encoding="utf-8",
    )
    (tmp_path / "boj.txt").write_text(
        "Yield curve control remains in place.\n\nWage growth is monitored.",
        encoding="utf-8",
    )
    retriever = TfidfRetriever(tmp_path)
    results = retriever.retrieve("inflation policy rate", top_k=2)
    assert results
    assert results[0].source == "fomc.txt"


def test_retriever_empty_corpus(tmp_path):
    assert TfidfRetriever(tmp_path).retrieve("anything") == []
