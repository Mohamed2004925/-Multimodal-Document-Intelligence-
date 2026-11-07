# Limitations

This project is a local document-processing prototype. Review its extracted fields and answers against the original file before using them for business decisions.

## Model and layout

- The bundled `doclaynet_yolov8s_baseline.pt` is a YOLO **detection** model with general document-layout classes. It is not an invoice-specific field detector and does not produce segmentation masks.
- Layout categories such as text and table help choose OCR regions; they do not directly identify supplier, customer, date, or total fields.
- The current reading order sorts regions roughly from top to bottom and left to right. Complex columns, nested tables, rotated pages, and overlapping content can be misordered.
- Table regions are passed to OCR as a crop. The app does not reconstruct table rows, columns, or cell relationships.

## OCR and extraction

- OCR runs locally with Tesseract. Image resolution, blur, skew, handwriting, unusual fonts, language, and low contrast can reduce accuracy.
- Field extraction uses regular expressions and labels in recognized text. It is not a trained information-extraction model and may miss unfamiliar layouts or label wording.
- Document-type classification uses keywords found in OCR text. It can fail when the document has no clear type label or contains multiple document types.
- OCR confidence is not calibrated into field confidence. The JSON reports `confidence: null`; uncertainty and review flags come from missing or conflicting rule-based evidence.

## Question answering

- Answers are grounded in extracted fields or matching OCR lines. This is retrieval and rule-based answering, not an LLM.
- The feature handles common field questions and simple text matching. It may not understand paraphrases, multi-step questions, or questions that require combining information across sections.
- When the evidence is missing or ambiguous, the app labels the response `not_found` or `uncertain`; it does not guarantee that every such label is correct.

## Input and runtime

- Supported inputs are common raster image formats and PDF. Password-protected PDFs are not supported.
- PDF pages are rendered and processed one at a time, but very large or long documents can still take substantial time and memory.
- Processing uses the local machine's installed model, Tesseract, and Python packages. GPU availability can change speed; it does not guarantee better OCR quality.
- The provided sample invoice is synthetic and is only a demonstration, not an accuracy benchmark.
