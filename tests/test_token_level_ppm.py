import pytest

from scripts.token_level_ppm import TokenLevelPPM, mix_token_distributions


def test_token_ppm_distribution_normalizes():
    scorer = TokenLevelPPM(order=2, vocab_bytes=[b"a", b"b", b"ab"])

    probs = scorer.token_distribution()

    assert abs(sum(probs) - 1.0) < 1e-9


def test_score_before_update_changes_after_observe():
    scorer = TokenLevelPPM(order=0, vocab_bytes=[b"a", b"b"])
    before = scorer.token_distribution()[0]

    scorer.observe_token(0)
    after = scorer.token_distribution()[0]

    assert after > before


def test_mix_distribution_normalizes():
    mixed = mix_token_distributions([0.7, 0.3], [0.25, 0.75], lam=0.2)

    assert abs(sum(mixed) - 1.0) < 1e-9
    assert mixed[0] == pytest.approx(0.2 * 0.7 + 0.8 * 0.25)


def test_observe_token_rejects_bad_token_id():
    scorer = TokenLevelPPM(order=2, vocab_bytes=[b"a"])

    with pytest.raises(IndexError):
        scorer.observe_token(1)


def test_mixture_rejects_unnormalized_inputs():
    with pytest.raises(ValueError, match="normalized"):
        mix_token_distributions([0.8, 0.3], [0.5, 0.5], lam=0.5)
