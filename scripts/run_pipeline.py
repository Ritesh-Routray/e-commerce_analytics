"""End-to-end run: raw CSVs -> fact table -> analyses -> model -> figures -> findings report.

    python scripts/run_pipeline.py              # everything
    python scripts/run_pipeline.py --skip-model # analyses and charts only (fast)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from olist_delivery import analysis, build, config, figures, model, report  # noqa: E402


def _step(label: str):
    print(f"[{time.strftime('%H:%M:%S')}] {label}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-model", action="store_true", help="skip model training")
    args = parser.parse_args()

    config.ensure_dirs()

    _step("Building order fact table")
    fact = build.build_order_fact()
    build.save_fact(fact)
    _step(f"  {len(fact):,} orders, {int(fact['is_delivered'].sum()):,} delivered")

    _step("Running analyses")
    results = analysis.run_all(fact)
    for name, value in results.items():
        if isinstance(value, dict):
            (config.TABLES_DIR / f"{name}.json").write_text(json.dumps(value, indent=2))
        else:
            value.to_csv(config.TABLES_DIR / f"{name}.csv", index=False)

    model_result = None
    if not args.skip_model:
        _step("Training late-delivery risk model")
        model_result = model.train_and_evaluate(fact)
        (config.TABLES_DIR / "model_metrics.json").write_text(
            json.dumps(model_result["metrics"], indent=2)
        )
        model_result["importance"].to_csv(config.TABLES_DIR / "feature_importance.csv", index=False)
        model_result["deciles"].to_csv(config.TABLES_DIR / "risk_deciles.csv", index=False)
        model_result["folds"].to_csv(config.TABLES_DIR / "model_folds.csv", index=False)
        model_result["ablation"].to_csv(config.TABLES_DIR / "model_ablation.csv", index=False)

    _step("Drawing figures")
    figures.make_all(results, model_result)

    _step("Writing findings report")
    report.write_findings(results, model_result)
    _step("Done. See outputs/ and reports/FINDINGS.md")


if __name__ == "__main__":
    main()
