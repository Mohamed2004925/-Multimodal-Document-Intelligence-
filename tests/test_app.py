import unittest

from app import DocumentResult, answer_question, extract_fields


class ExtractFieldsTests(unittest.TestCase):
    def test_due_date_is_not_extracted_as_amount(self):
        result = extract_fields("Invoice\nDue Date: 10/31/2026")
        self.assertEqual(result["fields"]["amount"]["extraction_status"], "missing")

    def test_section_number_is_not_extracted_as_date(self):
        result = extract_fields("Section 4 2012 applies here")
        self.assertEqual(result["fields"]["date"]["extraction_status"], "missing")

    def test_ambiguous_numeric_date_is_not_guessed(self):
        result = extract_fields("Invoice Date: 03/04/2026")
        self.assertEqual(result["fields"]["date"]["value"], "03/04/2026")
        self.assertEqual(result["fields"]["date"]["extraction_status"], "uncertain")

    def test_unambiguous_date_is_normalized_to_iso(self):
        result = extract_fields("Invoice Date: 31/10/2026")
        self.assertEqual(result["fields"]["date"]["value"], "2026-10-31")
        self.assertEqual(result["fields"]["date"]["extraction_status"], "extracted")

    def test_supplier_and_customer_are_separate_fields(self):
        result = extract_fields("Invoice\nSupplier: Acme Ltd\nBill To: Example Buyer")
        self.assertEqual(result["fields"]["supplier_name"]["value"], "Acme Ltd")
        self.assertEqual(result["fields"]["customer_name"]["value"], "Example Buyer")

    def test_multiple_supplier_candidates_are_uncertain(self):
        result = extract_fields("Invoice\nSupplier: Acme\nVendor: Acme Ltd")
        self.assertEqual(result["fields"]["supplier_name"]["extraction_status"], "uncertain")

    def test_missing_optional_status_does_not_require_review(self):
        result = extract_fields(
            "Invoice\nSupplier: Acme Ltd\nInvoice Date: 2026-10-31\nTotal: $45.00"
        )
        self.assertEqual(result["fields"]["status"]["extraction_status"], "missing")
        self.assertFalse(result["needs_review"])

    def test_required_missing_amount_requires_review(self):
        result = extract_fields("Invoice\nSupplier: Acme Ltd\nInvoice Date: 2026-10-31")
        self.assertEqual(result["fields"]["amount"]["extraction_status"], "missing")
        self.assertTrue(result["needs_review"])

    def test_generic_name_does_not_choose_between_supplier_and_customer(self):
        result = extract_fields("Invoice\nSupplier: Acme Ltd\nCustomer: Example Buyer")
        self.assertEqual(result["fields"]["name"]["extraction_status"], "uncertain")
        self.assertIsNone(result["fields"]["name"]["value"])

    def test_schema_rejects_extracted_field_without_value(self):
        with self.assertRaises(ValueError):
            from app import FieldResult
            FieldResult(value=None, extraction_status="extracted")

    def test_schema_requires_all_brief_fields(self):
        fields = extract_fields("Invoice\nSupplier: Acme\nDate: 2026-10-31\nTotal: $45.00")["fields"]
        with self.assertRaises(ValueError):
            DocumentResult.model_validate({
                "source_file": "sample.png", "fields": {"document_type": fields["document_type"]},
                "required_fields": [], "needs_review": False, "ocr_engine": "Tesseract",
                "text_source": "full_page", "full_text": "", "preprocessing": {}, "pages": [],
            })

    def test_section_question_returns_actions_below_heading(self):
        result = {
            "full_text": (
                "East Market closed 94 percent of requests. Actions for next quarter\n"
                "1 Complete the three delayed maintenance visits by April 30.\n"
                "2 Review Riverside repeat requests at the May operations meeting.\n"
                "3 Maintain a monthly stock check for critical replacement parts.\n"
                "SYNTHETIC TEST REPORT"
            ),
            "pages": [{"regions": [
                {"layout_label": "Section-header", "bbox_xyxy": [900, 400, 1300, 440], "text": "Actions for next quarter"},
                {"layout_label": "List-item", "bbox_xyxy": [900, 480, 1600, 530], "text": "2 Review Riverside repeat requests at the May operations meeting."},
                {"layout_label": "List-item", "bbox_xyxy": [900, 540, 1600, 590], "text": "3 Maintain a monthly stock check for critical replacement parts."},
                {"layout_label": "Page-footer", "bbox_xyxy": [100, 900, 800, 930], "text": "SYNTHETIC TEST REPORT"},
            ]}],
        }
        answer = answer_question("What are the actions for next quarter?", result)
        self.assertEqual(answer["status"], "answered")
        self.assertIn("Complete the three delayed maintenance visits", answer["answer"])
        self.assertIn("Review Riverside repeat requests", answer["answer"])
        self.assertIn("Maintain a monthly stock check", answer["answer"])
        self.assertNotIn("East Market", answer["answer"])

    def test_section_question_uses_same_column_regions(self):
        result = {
            "full_text": "Service performance Requests by month\ncombined page text",
            "pages": [{"regions": [
                {"layout_label": "Section-header", "bbox_xyxy": [100, 100, 320, 135], "text": "Service performance"},
                {"layout_label": "Caption", "bbox_xyxy": [550, 100, 850, 135], "text": "Requests by month"},
                {"layout_label": "Text", "bbox_xyxy": [110, 160, 480, 240], "text": "Response times improved at four sites."},
                {"layout_label": "Table", "bbox_xyxy": [110, 260, 480, 430], "text": "Central 46 43 3.6 hours"},
                {"layout_label": "Section-header", "bbox_xyxy": [100, 500, 320, 535], "text": "Key observations"},
            ]}],
        }
        answer = answer_question("What is the service performance?", result)
        self.assertEqual(answer["status"], "answered")
        self.assertIn("Response times improved", answer["answer"])
        self.assertIn("Central 46 43", answer["answer"])
        self.assertNotIn("Requests by month", answer["answer"])


if __name__ == "__main__":
    unittest.main()
