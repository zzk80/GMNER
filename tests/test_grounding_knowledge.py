from scripts.build_grounding_knowledge import build_groundability_stats


def test_build_groundability_stats_by_type_and_mention():
    occurrences = [
        {
            "normalized_mention": "white house",
            "entity_type": "LOC",
            "groundable": True,
            "best_region_index": 1,
        },
        {
            "normalized_mention": "white house",
            "entity_type": "ORG",
            "groundable": False,
            "best_region_index": None,
        },
        {
            "normalized_mention": "chelsea",
            "entity_type": "ORG",
            "groundable": True,
            "best_region_index": None,
        },
    ]

    type_rows, mention_rows = build_groundability_stats(occurrences)
    by_type = {row["entity_type"]: row for row in type_rows}
    by_mention = {(row["mention"], row["entity_type"]): row for row in mention_rows}

    assert by_type["LOC"]["groundability_rate"] == 1.0
    assert by_type["ORG"]["groundability_rate"] == 0.5
    assert by_type["ORG"]["region_match_rate"] == 0.0
    assert by_mention[("white house", "ORG")]["null_prior"] == 1.0
