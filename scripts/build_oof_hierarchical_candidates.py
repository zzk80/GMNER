"""Build and merge hierarchical candidate caches from trained OOF Stage1 folds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.merge_oof_record_candidate_caches import merge_caches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--fold-summary",
        default="knowledge/evidence/oof/folds/fold_summary.json",
    )
    parser.add_argument(
        "--checkpoint-root", default="outputs/fmnerg_stage1_oof"
    )
    parser.add_argument(
        "--work-dir", default="knowledge/record_candidates/oof/folds"
    )
    parser.add_argument(
        "--output",
        default="knowledge/record_candidates/oof/fmnerg_train_hierarchical_oof.pt",
    )
    parser.add_argument("--k-best", type=int, default=6)
    parser.add_argument("--max-span-candidates", type=int, default=12)
    parser.add_argument("--top-m-types", type=int, default=3)
    parser.add_argument("--boundary-shift", type=int, default=0)
    parser.add_argument("--boundary-penalty", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    fold_summary_path = resolve(args.fold_summary, root)
    checkpoint_root = resolve(args.checkpoint_root, root)
    work_dir = resolve(args.work_dir, root)
    output_path = resolve(args.output, root)
    summary = json.loads(fold_summary_path.read_text(encoding="utf-8"))
    folds = list(summary.get("folds") or [])
    if len(folds) < 2:
        raise ValueError(f"Invalid fold summary: {fold_summary_path}")
    work_dir.mkdir(parents=True, exist_ok=True)

    fold_caches: list[Path] = []
    for fold in folds:
        fold_id = int(fold["fold"])
        heldout_file = Path(fold["heldout_file"]).resolve()
        checkpoint = checkpoint_root / f"fold{fold_id}" / "best_model.pt"
        fold_cache = work_dir / f"fold{fold_id}.pt"
        fold_caches.append(fold_cache)
        if fold_cache.exists() and not args.force:
            print(f"Skipping existing fold cache: {fold_cache}", flush=True)
            continue
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing OOF Stage1 checkpoint: {checkpoint}. Run the fold training "
                "without --cleanup-fold-checkpoints first."
            )
        command = [
            sys.executable,
            str(root / "scripts" / "build_record_candidate_cache.py"),
            "--config",
            str(resolve(args.config, root)),
            "--checkpoint",
            str(checkpoint),
            "--split",
            "train",
            "--input-file",
            str(heldout_file),
            "--oof-fold-id",
            str(fold_id),
            "--output",
            str(fold_cache),
            "--k-best",
            str(args.k_best),
            "--max-span-candidates",
            str(args.max_span_candidates),
            "--top-m-types",
            str(args.top_m_types),
            "--boundary-shift",
            str(args.boundary_shift),
            "--boundary-penalty",
            str(args.boundary_penalty),
            "--batch-size",
            str(args.batch_size),
            "--device",
            str(args.device),
        ]
        print("+ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=root, check=True)

    result = merge_caches(
        fold_caches,
        output_path,
        expected_records=int(summary.get("records", 0)) or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
