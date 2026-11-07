"""Run the local pipeline on synthetic fixtures and compare OCR text sources."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app  # noqa: E402


FIELDS = ("document_type", "supplier_name", "customer_name", "date", "amount", "status")


def norm(field: str, value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().casefold()
    if field == "amount":
        return "".join(ch for ch in value if ch.isdigit() or ch == ".")
    if field == "date":
        parsed, status = app._parse_date(value)
        return parsed if status == "extracted" else value
    return " ".join(value.split())


def run() -> dict:
    truth = json.loads((ROOT / "samples/evaluation/ground_truth.json").read_text(encoding="utf-8"))
    app.configure_tesseract()
    model = app.load_model(app.DEFAULT_WEIGHTS)
    docs = []
    for expected in truth:
        path = ROOT / expected["file"]
        result = app.process_document(path, model)
        app.DocumentResult.model_validate(result)
        variants = result["ocr_text_variants"]
        per_source = {}
        for source in ("full_page", "layout_crop", "merged"):
            source_extraction = app.extract_fields(variants[source])
            extracted = source_extraction["fields"]
            per_source[source] = {
                field: {"value": extracted[field]["value"], "status": extracted[field]["extraction_status"]}
                for field in FIELDS
            }
            per_source[source]["needs_review"] = source_extraction["needs_review"]
        docs.append({"expected": expected, "result": result, "per_source": per_source})

    report = {"fixture_count": len(docs), "synthetic_only": True, "sources": {}}
    for source in ("full_page", "layout_crop", "merged"):
        metrics = {field: {"correct": 0, "wrong": 0, "missing": 0, "status_correct": 0, "status_wrong": 0} for field in FIELDS}
        type_correct = 0
        needs_review_true_positive = 0
        needs_review_false_negative = 0
        wrong_docs = 0
        for doc in docs:
            expected, actual = doc["expected"], doc["per_source"][source]
            has_error = False
            for field in FIELDS:
                exp = norm(field, expected.get(field))
                got = norm(field, actual[field]["value"])
                if got == exp:
                    metrics[field]["correct"] += 1
                elif got is None:
                    metrics[field]["missing"] += 1
                    has_error = True
                else:
                    metrics[field]["wrong"] += 1
                    has_error = True
                if actual[field]["status"] == expected["expected_statuses"][field]:
                    metrics[field]["status_correct"] += 1
                else:
                    metrics[field]["status_wrong"] += 1
                    has_error = True
            type_correct += norm("document_type", actual["document_type"]["value"]) == norm("document_type", expected["document_type"])
            if has_error:
                wrong_docs += 1
                if doc["per_source"][source]["needs_review"]:
                    needs_review_true_positive += 1
                else:
                    needs_review_false_negative += 1
        report["sources"][source] = {
            "fields": metrics,
            "document_type_accuracy": type_correct / len(docs),
            "needs_review_recall": needs_review_true_positive / wrong_docs if wrong_docs else None,
            "documents_with_any_wrong_or_missing_field": wrong_docs,
        }
    report["selected_text_source_counts"] = {
        source: sum(doc["result"]["text_source"] == source for doc in docs)
        for source in ("full_page", "merged", "mixed")
    }
    out = ROOT / "evaluation/results.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
