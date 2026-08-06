from scripts.audit_final_m33a_dev import (
    Entity,
    maximum_weight_matching,
    overlap_features,
    primary_class,
    target_calculator,
    validate_gate,
    write_rows,
)


def test_phase0_locked_contract_passes() -> None:
    observed = {
        "gold": 2450,
        "predicted": 2504,
        "span_correct": 2162,
        "mner_correct": 2023,
        "span_errors": 288,
        "type_errors": 139,
        "span_f1": 0.8728300363342755,
        "mner_f1": 0.8167137666532096,
    }
    assert all(validate_gate(observed).values())


def test_overlap_features_use_half_open_word_spans() -> None:
    overlap_f1, iou, distance = overlap_features((2, 5), (3, 6))
    assert overlap_f1 == 2 / 3
    assert iou == 1 / 2
    assert distance == 2


def test_component_classes_are_mutually_exclusive() -> None:
    assert primary_class(1, 1) == "boundary_shift"
    assert primary_class(1, 2) == "split"
    assert primary_class(2, 1) == "merge"
    assert primary_class(2, 2) == "complex_split_merge"
    assert primary_class(1, 0) == "pure_miss"
    assert primary_class(0, 1) == "pure_false_positive"


def test_matching_prefers_higher_overlap_before_confidence() -> None:
    gold = [Entity((0, 2), 1), Entity((3, 5), 1)]
    predictions = [Entity((0, 1), 1), Entity((0, 2), 1), Entity((3, 5), 1)]
    matching = maximum_weight_matching(
        gold,
        predictions,
        {(0, 1): 100.0, (0, 2): 0.0, (3, 5): 0.0},
    )
    assert matching == {0: 1, 1: 2}


def test_target_calculator_matches_preregistered_expected_precision_math() -> None:
    report = target_calculator()
    assert report["target_correct_fixed_prediction_count"] == 2056
    assert report["required_net_gain"] == 33
    assert [
        row["minimum_actions"]
        for row in report["pure_promotions_expected_precision"]
    ] == [57, 68, 86, 99, 116]


def test_row_export_defaults_to_csv_only(tmp_path) -> None:
    output = tmp_path / "audit"
    write_rows(output, [{"record_id": "0", "value": 1}])
    assert output.with_suffix(".csv").exists()
    assert not output.with_suffix(".jsonl").exists()
