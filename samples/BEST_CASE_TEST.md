# Best-case manual test

Use `best_case_invoice.png` first in the Streamlit app. It is a deliberately simple, high-resolution, single-column synthetic invoice with strong contrast and clear labels. `best_case_invoice.pdf` contains the same page so you can check both supported input paths.

Expected values:

| Field | Expected result |
| --- | --- |
| Document type | `invoice` — extracted |
| Name | `Northstar Supply` — extracted |
| Supplier name | `Northstar Supply` — extracted |
| Customer name | missing |
| Date | `2026-10-31` — extracted |
| Amount | `$1,248.50` — extracted |
| Status | `unpaid` — extracted |
| Needs review | `false` |

To regenerate the files, run `python samples/create_best_case.py`. In the app, click **Browse files**, select the image or PDF, then click **Analyze document**. Check the extracted field values, their status and evidence, and that `needs_review` is false. The exact OCR text and layout detections can vary by machine; this fixture is optimized for readability but cannot guarantee perfect recognition.

After that known-good check, test the edge cases in `evaluation/`: missing totals, ambiguous dates, and conflicting totals should be marked for review. Finally, try a real scan or phone photo to see how blur, skew, shadows, and compression affect results.
