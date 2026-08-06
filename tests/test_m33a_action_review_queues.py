from scripts.prepare_m33a_action_review_queues import build_review_queues


def _type_row(*, text: bool, visual: bool) -> dict[str, str]:
    return {
        "text_candidate_oracle": str(text),
        "gold_visible": str(visual),
        "r16_gold_covered": str(visual),
        "record_id": "0",
    }


def test_review_queue_partition_and_boundary_counts() -> None:
    type_rows = (
        [_type_row(text=True, visual=True) for _ in range(21)]
        + [_type_row(text=False, visual=True) for _ in range(4)]
        + [_type_row(text=True, visual=False) for _ in range(86)]
        + [_type_row(text=False, visual=False) for _ in range(28)]
    )
    span_rows = [
        {"safe_replacement": "True", "safe_promotion": "False"}
        for _ in range(55)
    ] + [
        {"safe_replacement": "False", "safe_promotion": "True"}
        for _ in range(61)
    ]
    queues = build_review_queues(type_rows, span_rows)
    assert len(queues["type_union"]) == 111
    assert len(queues["replacement"]) == 55
    assert len(queues["promotion"]) == 61
    assert all(
        row["review_partition"] != "neither" for row in queues["type_union"]
    )
