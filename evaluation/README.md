# Evaluation fixtures

`ground_truth.json` describes ten generated invoice and receipt images, including examples designed to produce a missing amount, an ambiguous date, and conflicting totals. Run `python samples/create_evaluation_samples.py` to regenerate them.

These generated files check pipeline behavior and output consistency. They are not independent evidence of real-world accuracy. Add 5–10 appropriately licensed public or manually labeled real documents before reporting a representative benchmark. Do not include private documents in this repository.

The evaluation script compares extracted values from full-page OCR, layout-crop OCR, and merged text. It reports exact matches, wrong versus missing values, extraction-status correctness, document-type accuracy, and `needs_review` recall for documents with at least one incorrect value or status. A `null` recall means that source had no errors, so recall had no error cases to measure.
