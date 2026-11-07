"""Run the actual application pipeline on bundled synthetic samples."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app


OUTPUT_DIR = Path(__file__).resolve().parent
QUESTIONS = ["Who is the supplier?", "What is the total amount?"]


def main() -> None:
    tesseract = app.configure_tesseract()
    if tesseract is None:
        raise RuntimeError("Tesseract was not found; install it before generating examples.")
    model = app.load_model(app.DEFAULT_WEIGHTS)

    for filename in ("sample_invoice.png", "sample_invoice.pdf"):
        sample_path = ROOT / "samples" / filename
        result = app.process_document(sample_path, model, imgsz=1024)
        result["question_answers"] = [
            app.answer_question(question, result) for question in QUESTIONS
        ]
        result = app.DocumentResult.model_validate(result).model_dump(mode="python")
        kind = "image" if sample_path.suffix.lower() != ".pdf" else "pdf"
        output_path = OUTPUT_DIR / f"{sample_path.stem}_{kind}_output.json"
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Saved {output_path.name}")
        print("Extracted fields:", {
            name: field["value"]
            for name, field in result["fields"].items()
        })
        print("Needs review:", result["needs_review"])

    uncertain_text = (
        "Invoice\nSupplier: Example Supplier\n"
        "Invoice Date: 03/04/2026\n"
    )
    validation = app.process_document(ROOT / "samples" / "sample_invoice.png", model)
    validation["source_file"] = "synthetic_ocr_validation_fixture"
    validation["full_text"] = uncertain_text
    validation.update(app.extract_fields(uncertain_text))
    validation["question_answers"] = [
        app.answer_question("What is the invoice date?", validation),
        app.answer_question("What is the total amount?", validation),
        app.answer_question("When is the payment due?", validation),
    ]
    validation = app.DocumentResult.model_validate(validation).model_dump(mode="python")
    validation_path = OUTPUT_DIR / "missing_and_uncertain_fields.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {validation_path.name}")


if __name__ == "__main__":
    main()
