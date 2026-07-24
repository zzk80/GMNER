from pathlib import Path

from gmner.knowledge import (
    build_entity_inventory,
    extract_entities_from_record,
    normalize_mention,
    read_conll_records,
)


def test_normalize_mention_removes_social_prefixes():
    assert normalize_mention("#Jordan") == "jordan"
    assert normalize_mention("@Apple  Inc") == "apple inc"


def test_extract_entities_handles_bio_boundaries():
    record = {
        "tokens": ["New", "York", "beats", "Boston"],
        "ner_tags": ["B-LOC", "I-LOC", "O", "B-ORG"],
    }

    entities = extract_entities_from_record(record)

    assert [(item["mention"], item["entity_type"]) for item in entities] == [
        ("New York", "LOC"),
        ("Boston", "ORG"),
    ]


def test_read_conll_records_and_inventory(tmp_path: Path):
    conll = tmp_path / "train.txt"
    conll.write_text(
        "IMGID:1\n"
        "White\tB-LOC\n"
        "House\tI-LOC\n"
        "\n"
        "IMGID:2\n"
        "White\tB-ORG\n"
        "House\tI-ORG\n"
        "\n",
        encoding="utf-8",
    )

    records = read_conll_records(conll)
    occurrences, inventory, summary = build_entity_inventory(conll)

    assert len(records) == 2
    assert summary["entities"] == 2
    assert occurrences[0]["normalized_mention"] == "white house"
    assert inventory[0]["ambiguous"] is True
    assert inventory[0]["type_counts"] == {"LOC": 1, "ORG": 1}
