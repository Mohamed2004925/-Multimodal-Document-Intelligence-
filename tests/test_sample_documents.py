"""Integration check: process every checked-in sample and validate its schema."""

import unittest
from pathlib import Path

import app


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FILES = sorted(
    path for path in (ROOT / "samples").rglob("*")
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}
)


@unittest.skipUnless(
    app.configure_tesseract() and app.DEFAULT_WEIGHTS.is_file(),
    "integration requires local Tesseract and bundled YOLO weights",
)
class SampleOutputSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = app.load_model(app.DEFAULT_WEIGHTS)

    def test_every_sample_output_passes_schema(self):
        self.assertGreater(len(SAMPLE_FILES), 0, "No sample documents found")
        for path in SAMPLE_FILES:
            with self.subTest(path=path.name):
                result = app.process_document(path, self.model)
                app.DocumentResult.model_validate(result)

    def test_best_case_invoice_has_expected_fields_for_image_and_pdf(self):
        for filename in ("best_case_invoice.png", "best_case_invoice.pdf"):
            with self.subTest(filename=filename):
                result = app.process_document(ROOT / "samples" / filename, self.model)
                fields = result["fields"]
                expected = {
                    "document_type": "invoice",
                    "name": "Northstar Supply",
                    "supplier_name": "Northstar Supply",
                    "date": "2026-10-31",
                    "amount": "$1,248.50",
                    "status": "unpaid",
                }
                for field, value in expected.items():
                    self.assertEqual(fields[field]["value"], value, field)
                    self.assertEqual(fields[field]["extraction_status"], "extracted", field)
                self.assertEqual(fields["customer_name"]["extraction_status"], "missing")
                self.assertFalse(result["needs_review"])

    def test_complex_report_is_classified_and_date_extracted(self):
        result = app.process_document(ROOT / "samples" / "complex_operations_report.png", self.model)
        fields = result["fields"]
        self.assertEqual(fields["document_type"]["value"], "report")
        self.assertEqual(fields["date"]["value"], "2026-04-15")
        self.assertEqual(fields["status"]["value"], "pending")
        self.assertFalse(result["needs_review"])


if __name__ == "__main__":
    unittest.main()
