import json
from pathlib import Path

from minisweagent.run.utilities.deepseek_logprobs import (
    analyze_generated,
    analyze_gold_from_rows,
    extract_continuation_rows,
    first_difference,
    matches_target,
    normalize_prediction,
    TokenDetail,
    _load_cases,
)


def test_normalize_and_match_target() -> None:
    assert normalize_prediction(" \x001234567.\n") == "1234567."
    assert matches_target(" 1234567.\nextra", "1234567")


def test_extract_continuation_rows_uses_offsets() -> None:
    choice = {
        "logprobs": {
            "tokens": ["Prompt", " 12", "34", "."],
            "token_logprobs": [None, -0.2, -0.3, -0.4],
            "top_logprobs": [{}, {" 12": -0.2}, {"34": -0.3}, {".": -0.4}],
            "text_offset": [0, 6, 9, 11],
        }
    }
    rows = extract_continuation_rows(choice, "Prompt", " 1234")
    assert [row.token for row in rows] == [" 12", "34"]


def test_analyze_gold_tracks_missing_and_top1_mismatch() -> None:
    gold = analyze_gold_from_rows(
        "1234",
        [" 12", "34"],
        [
            TokenDetail(token=" 12", logprob=-0.2, top_logprobs={" 12": -0.2, " 99": -1.0}, offset=None),
            TokenDetail(token="34", logprob=None, top_logprobs={"35": -0.1, "36": -0.2}, offset=None),
        ],
    )
    assert gold.total_logprob is None
    assert gold.first_missing_position == 2
    assert gold.first_top1_mismatch == 2


def test_analyze_generated_uses_ruler_style_matching() -> None:
    choice = {
        "text": " 1234567.\nThe grass is green.",
        "logprobs": {
            "tokens": [" 123", "4567", "."],
            "token_logprobs": [-0.1, -0.2, -0.3],
            "top_logprobs": [{" 123": -0.1}, {"4567": -0.2}, {".": -0.3}],
        },
    }
    generated = analyze_generated(choice, "1234567")
    assert generated.matches
    assert generated.tokens == [" 123", "4567", "."]


def test_first_difference_handles_equal_and_length_mismatch() -> None:
    assert first_difference(["a", "b"], ["a", "c"]) == 2
    assert first_difference(["a"], ["a", "b"]) == 2
    assert first_difference(["a"], ["a"]) is None


def test_load_cases_reads_expected_prompt_shape(tmp_path: Path) -> None:
    sample_path = tmp_path / "samples.jsonl"
    rows = [
        {
            "arguments": {"gen_args_0": {"arg_0": "prompt-1"}},
            "target": ["111"],
            "prompt_hash": "a",
        },
        {
            "arguments": {"gen_args_0": {"arg_0": "prompt-2"}},
            "target": ["222"],
            "prompt_hash": "b",
        },
    ]
    sample_path.write_text("\n".join(json.dumps(row) for row in rows))
    cases = _load_cases(sample_path, (2,), (1,))
    assert [(case.sample_id, case.kind, case.prompt, case.target) for case in cases] == [
        (1, "control", "prompt-1", "111"),
        (2, "miss", "prompt-2", "222"),
    ]
