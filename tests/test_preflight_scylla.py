from scripts.preflight_scylla import validate_env


REFERENCE_ENV = {
    "VOCAB_SIZE": "998",
    "XSA_LAST_N": "11",
    "USE_GPTQ": "1",
    "GPTQ_RESERVE_MS": "9000",
    "TTT_ENABLED": "0",
    "BIGRAM_VOCAB_SIZE": "2816",
    "BIGRAM_DIM": "40",
    "QK_GAIN_INIT": "5.25",
    "NUM_LOOPS": "2",
    "LOOP_START": "3",
    "LOOP_END": "5",
    "ENABLE_LOOPING_AT": "0.35",
    "VAL_LOSS_EVERY": "0",
}


def test_reference_env_passes():
    assert validate_env(REFERENCE_ENV) == []


def test_ttt_is_rejected():
    env = dict(REFERENCE_ENV, TTT_ENABLED="1")
    errors = validate_env(env)
    assert any("TTT_ENABLED" in e for e in errors)


def test_wrong_loop_schedule_is_rejected():
    env = dict(REFERENCE_ENV, LOOP_START="4")
    errors = validate_env(env)
    assert any("LOOP_START" in e for e in errors)
