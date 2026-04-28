import json

from scripts.extract_scylla_modal_result import parse_result, result_to_json


def test_parse_result_extracts_final_metrics():
    text = """
    seed:42
    step:5260/9000 val_loss:1.9492 val_bpb:0.9596 train_time:591041ms step_avg:112.36ms
    Serialized model int6+lzma: 15762468 bytes
    Total submission size int6+lzma: 15868333 bytes
    final_int6_roundtrip_exact val_loss:1.95612000 val_bpb:0.96234529
    final_int6_sliding_window_exact val_loss:1.91501600 val_bpb:0.94238042
    """

    result = parse_result(text)

    assert result.seed == 42
    assert result.step == 5260
    assert result.train_time_ms == 591041
    assert result.model_bytes == 15762468
    assert result.total_bytes == 15868333
    assert result.roundtrip_bpb == 0.96234529
    assert result.sliding_bpb == 0.94238042
    assert result.margin_bytes == 131667


def test_result_to_json_is_stable():
    result = parse_result(
        """
        seed:1337
        step:4923/9000 val_loss:1.9537 val_bpb:0.9615 train_time:591155ms step_avg:120.08ms
        Serialized model int6+lzma: 15744092 bytes
        Total submission size int6+lzma: 15849957 bytes
        final_int6_roundtrip_exact val_loss:1.95825204 val_bpb:0.96377557
        final_int6_sliding_window_exact val_loss:1.91753823 val_bpb:0.94372928
        """
    )

    payload = json.loads(result_to_json(result))

    assert payload == {
        "seed": 1337,
        "step": 4923,
        "train_time_ms": 591155,
        "model_bytes": 15744092,
        "total_bytes": 15849957,
        "margin_bytes": 150043,
        "roundtrip_bpb": 0.96377557,
        "sliding_bpb": 0.94372928,
    }
