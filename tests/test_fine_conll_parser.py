from pathlib import Path

from gmner.constants import DEFAULT_LABEL2ID
from gmner.utils.io import maybe_convert_conll, read_jsonl


def test_fine_conll_keeps_coarse_ner_and_fine_tags(tmp_path: Path):
    source = tmp_path / "train.txt"
    source.write_text(
        "\n".join(
            [
                "IMGID:1",
                "Pak B-building B-building_other",
                "Webber B-person B-intellectual",
                "MS I-person I-intellectual",
                "",
            ]
        ),
        encoding="utf-8",
    )

    output = maybe_convert_conll(source, tmp_path / "out")
    records = read_jsonl(output)

    assert records[0]["ner_tags"] == [
        DEFAULT_LABEL2ID["B-LOC"],
        DEFAULT_LABEL2ID["B-PER"],
        DEFAULT_LABEL2ID["I-PER"],
    ]
    assert records[0]["fine_ner_tags"] == [
        "B-building_other",
        "B-intellectual",
        "I-intellectual",
    ]
