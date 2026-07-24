"""Break down FMNERG subtype errors without reading the test split."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidecars.fmnerg_subtype.data import read_fine_conll
from sidecars.fmnerg_subtype.evaluator import save_json_atomic
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--train-source", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / max(denominator, 1)


def group_metrics(counter: Counter) -> dict[str, float]:
    gold = int(counter["gold"])
    exact_span = int(counter["exact_span"])
    metrics = {
        "gold": float(gold),
        "exact_span_recall": safe_rate(exact_span, gold),
        "coarse_recall": safe_rate(int(counter["coarse"]), gold),
        "fine_mner_recall": safe_rate(int(counter["subtype"]), gold),
        "eeg_recall": safe_rate(int(counter["region"]), gold),
        "gmner_recall": safe_rate(int(counter["gmner"]), gold),
        "fmnerg_recall": safe_rate(int(counter["fmnerg"]), gold),
        "subtype_accuracy_given_exact_span": safe_rate(
            int(counter["subtype"]),
            exact_span,
        ),
        "subtype_accuracy_given_correct_parent": safe_rate(
            int(counter["subtype"]),
            int(counter["coarse"]),
        ),
    }
    predicted = int(counter["predicted"])
    correct = int(counter["subtype"])
    precision = safe_rate(correct, predicted)
    recall = safe_rate(correct, gold)
    metrics.update(
        {
            "predicted": float(predicted),
            "subtype_precision": precision,
            "subtype_recall": recall,
            "subtype_f1": (
                2 * precision * recall
                / max(precision + recall, 1e-8)
            ),
            "prediction_to_gold_ratio": predicted / max(gold, 1),
        }
    )
    return metrics


def frequency_bands(train_counts: Counter) -> dict[str, str]:
    ordered = sorted(train_counts, key=lambda label: (train_counts[label], label))
    size = len(ordered)
    output = {}
    for index, label in enumerate(ordered):
        if index < size / 3:
            output[label] = "low"
        elif index < 2 * size / 3:
            output[label] = "medium"
        else:
            output[label] = "high"
    return output


def main() -> None:
    args = parse_args()
    taxonomy = SubtypeTaxonomy.from_file(args.taxonomy)
    training = read_fine_conll(
        args.train_source,
        taxonomy,
        require_all_subtypes=True,
    )
    train_counts = Counter(
        entity.subtype for record in training for entity in record.entities
    )
    bands = frequency_bands(train_counts)
    payload = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("split") != "dev" or metadata.get("test_accessed") is not False:
        raise ValueError("Subtype error analysis is dev-only before test release.")
    records = list(payload.get("records") or [])
    if not records:
        raise ValueError("Evaluation has no records; use --include-records.")

    overall = Counter()
    per_parent: dict[str, Counter] = defaultdict(Counter)
    per_subtype: dict[str, Counter] = defaultdict(Counter)
    per_frequency: dict[str, Counter] = defaultdict(Counter)
    per_visibility: dict[str, Counter] = defaultdict(Counter)
    subtype_confusions = Counter()

    for record in records:
        raw_predictions = list(record.get("predictions") or [])
        predictions_by_span = {
            tuple(map(int, prediction["span"])): prediction
            for prediction in raw_predictions
        }
        if len(predictions_by_span) != len(raw_predictions):
            raise ValueError(
                f"Duplicate predicted span in record {record['record_id']}."
            )
        for prediction in raw_predictions:
            predicted_subtype = str(prediction["subtype"])
            if predicted_subtype not in taxonomy.label2id:
                raise ValueError(
                    f"Unknown predicted subtype: {predicted_subtype!r}."
                )
            per_subtype[predicted_subtype]["predicted"] += 1
        for target in record.get("gold_entities") or []:
            subtype_id = int(target["subtype_id"])
            subtype = taxonomy.labels[subtype_id]
            parent = taxonomy.parent_by_label[subtype]
            visibility = "visible" if bool(target.get("visible", False)) else "null"
            counters = (
                overall,
                per_parent[parent],
                per_subtype[subtype],
                per_frequency[bands[subtype]],
                per_visibility[visibility],
            )
            for counter in counters:
                counter["gold"] += 1
            prediction = predictions_by_span.get(tuple(map(int, target["span"])))
            if prediction is None:
                continue
            coarse_ok = int(prediction["type_id"]) == int(target["type_id"])
            subtype_ok = int(prediction["subtype_id"]) == subtype_id
            region_ok = int(prediction["region_index"]) in {
                int(value)
                for value in target.get("region_positive_indices") or []
            }
            for counter in counters:
                counter["exact_span"] += 1
                counter["coarse"] += int(coarse_ok)
                counter["subtype"] += int(subtype_ok)
                counter["region"] += int(region_ok)
                counter["gmner"] += int(coarse_ok and region_ok)
                counter["fmnerg"] += int(subtype_ok and region_ok)
            if not subtype_ok:
                predicted_label = str(prediction.get("subtype", "<missing>"))
                subtype_confusions[f"{subtype}->{predicted_label}"] += 1

    per_subtype_metrics = {
        key: {
            "train_count": float(train_counts[key]),
            "frequency_band": bands[key],
            **group_metrics(value),
        }
        for key, value in sorted(per_subtype.items())
    }
    parent_macro_f1 = {
        parent: sum(
            values["subtype_f1"]
            for label, values in per_subtype_metrics.items()
            if taxonomy.parent_by_label[label] == parent
        )
        / max(
            sum(
                1
                for label in per_subtype_metrics
                if taxonomy.parent_by_label[label] == parent
            ),
            1,
        )
        for parent in taxonomy.coarse_type_ids
    }
    result = {
        "metadata": {
            "kind": "fmnerg_subtype_error_analysis",
            "format_version": 1,
            "split": "dev",
            "records": len(records),
            "test_accessed": False,
        },
        "overall": group_metrics(overall),
        "per_parent": {
            key: group_metrics(value)
            for key, value in sorted(per_parent.items())
        },
        "per_subtype": per_subtype_metrics,
        "parent_macro_f1": parent_macro_f1,
        "per_frequency": {
            key: group_metrics(value)
            for key, value in sorted(per_frequency.items())
        },
        "per_visibility": {
            key: group_metrics(value)
            for key, value in sorted(per_visibility.items())
        },
        "top_subtype_confusions": dict(subtype_confusions.most_common(50)),
    }
    save_json_atomic(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
