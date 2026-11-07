"""Document OCR inference prototype.

Usage from the project directory:
    .venv311\\Scripts\\python.exe app.py path\\to\\document.jpg --output result.json

The YOLO layout detector is loaded from best_model/ and Tesseract is called
locally through pytesseract. Neither model is trained or fine-tuned here.
Field extraction is a conservative rule-based baseline, not an LLM.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

import cv2
import pytesseract
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ultralytics import YOLO


DEFAULT_WEIGHTS = Path(__file__).parent / "best_model" / "doclaynet_yolov8s_baseline.pt"
TEXT_CLASSES = {
    "Text", "Title", "Section-header", "List-item", "Caption",
    "Footnote", "Page-header", "Page-footer",
}
# Tesseract can read flattened table crops; cell/row structure is not preserved.
OCR_CLASSES = TEXT_CLASSES | {"Table"}
BASE_UPSCALE = 2
MARGIN = 10
DATE_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
DATE_PATTERN = re.compile(
    rf"\b(?:\d{{4}}[-/]\d{{1,2}}[-/]\d{{1,2}}"
    rf"|\d{{1,2}}[-/]\d{{1,2}}[-/]\d{{2,4}}"
    rf"|\d{{1,2}}\s+{DATE_MONTH}\.?,?\s+\d{{2,4}}"
    rf"|{DATE_MONTH}\.?\s+\d{{1,2}},?\s+\d{{2,4}})\b",
    re.IGNORECASE,
)
AMOUNT_LINE = re.compile(
    r"\b(?:grand\s+total|total|amount\s+due|balance\s+due|amount|balance)\b",
    re.IGNORECASE,
)
MONEY = re.compile(
    r"(?:[$€£]|\b(?:USD|EUR|EGP|GBP)\b)\s*\d[\d,]*(?:\.\d{1,2})?"
    r"|\b\d[\d,]*\.\d{2}\b",
    re.IGNORECASE,
)
REQUIRED_FIELDS_BY_TYPE = {
    "invoice": ("supplier_name", "date", "amount"),
    "receipt": ("supplier_name", "date", "amount"),
    "purchase_order": ("supplier_name", "date", "amount"),
    "contract": ("date",),
    "report": (),
}


class FieldResult(BaseModel):
    value: str | None = None
    extraction_status: Literal["extracted", "uncertain", "missing"]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: str | None = None

    @model_validator(mode="after")
    def check_status_matches_value(self) -> "FieldResult":
        if self.extraction_status == "extracted" and self.value is None:
            raise ValueError("an extracted field must contain a value")
        if self.extraction_status == "missing" and self.value is not None:
            raise ValueError("a missing field cannot contain a value")
        return self


class DocumentFields(BaseModel):
    document_type: FieldResult
    name: FieldResult
    supplier_name: FieldResult
    customer_name: FieldResult
    date: FieldResult
    amount: FieldResult
    status: FieldResult


class QuestionAnswer(BaseModel):
    question: str
    answer: str | None
    status: Literal["answered", "uncertain", "not_found"]
    evidence: str | None = None
    message: str | None = None


class DocumentResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_file: str
    fields: DocumentFields
    required_fields: list[str]
    needs_review: bool
    ocr_engine: str
    text_source: Literal["full_page", "merged", "mixed"]
    full_text: str
    preprocessing: dict[str, Any]
    pages: list[dict[str, Any]]
    question_answers: list[QuestionAnswer] = Field(default_factory=list)


def validate_document_result(result: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a result before returning or serializing it."""
    return DocumentResult.model_validate(result).model_dump(mode="python")


def configure_tesseract(command: str | None = None) -> str | None:
    """Configure pytesseract from an explicit path, PATH, or common Windows install path."""
    configured = command or os.environ.get("TESSERACT_CMD")
    current = configured or pytesseract.pytesseract.tesseract_cmd

    if Path(current).is_file():
        pytesseract.pytesseract.tesseract_cmd = str(Path(current))
        return pytesseract.pytesseract.tesseract_cmd
    located = shutil.which(current) or shutil.which("tesseract")
    if located:
        pytesseract.pytesseract.tesseract_cmd = located
        return located

    for candidate in (
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Tesseract-OCR" / "tesseract.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Tesseract-OCR" / "tesseract.exe",
    ):
        if candidate.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return str(candidate)
    return None


def load_model(weights_path: Path) -> YOLO:
    """Load the existing document-layout detector once for a document/session."""
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"YOLO weights not found: {weights_path}. Set --weights to your best.pt file."
        )
    return YOLO(str(weights_path))


def page_upscale(image_shape: tuple[int, ...]) -> int:
    """Scale small pages up more and large scans less than the 1025 px pilot."""
    largest_side = max(image_shape[:2])
    if largest_side <= 0:
        return BASE_UPSCALE
    return min(3, max(1, round(BASE_UPSCALE * 1025 / largest_side)))


def ocr_crop(image_bgr: Any, scale: int = BASE_UPSCALE) -> str:
    """Run Tesseract with the pilot's 10 px margin and PSM 6."""
    height, width = image_bgr.shape[:2]
    if height == 0 or width == 0:
        return ""
    enlarged = cv2.resize(
        image_bgr,
        (width * scale, height * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    padded = cv2.copyMakeBorder(
        enlarged, MARGIN, MARGIN, MARGIN, MARGIN,
        cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )
    image_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return pytesseract.image_to_string(image_rgb, config="--oem 3 --psm 6").strip()


def detect_and_ocr(
    image: Any,
    model: YOLO,
    confidence: float = 0.25,
    imgsz: int = 640,
) -> list[dict[str, Any]]:
    """Detect page regions and OCR text-like/table regions in rough reading order."""
    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    result = model.predict(
        source=image,
        imgsz=imgsz,
        conf=confidence,
        device=device,
        agnostic_nms=True,
        verbose=False,
    )[0]

    names = model.names
    detections: list[dict[str, Any]] = []
    boxes = sorted(
        result.boxes,
        key=lambda box: (float(box.xyxy[0][1]), float(box.xyxy[0][0])),
    )
    image_height, image_width = image.shape[:2]
    scale = page_upscale(image.shape)

    for box in boxes:
        class_id = int(box.cls.item())
        label = names[class_id]
        if label not in OCR_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, x2 = max(0, x1), min(image_width, x2)
        y1, y2 = max(0, y1), min(image_height, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        text = ocr_crop(image[y1:y2, x1:x2], scale=scale)
        detections.append(
            {
                "layout_label": label,
                "layout_confidence": round(float(box.conf.item()), 4),
                "bbox_xyxy": [x1, y1, x2, y2],
                "text": text,
            }
        )

    # DocLayNet is not invoice-specific. If it finds no OCR regions, try the page.
    if not detections:
        detections.append(
            {
                "layout_label": "full_page_fallback",
                "layout_confidence": None,
                "bbox_xyxy": [0, 0, image_width, image_height],
                "text": ocr_crop(image, scale=scale),
            }
        )
    return detections


def _field(
    value: str | None,
    *,
    extraction_status: str,
    evidence: str | None = None,
) -> dict[str, Any]:
    return {
        "value": value,
        "extraction_status": extraction_status,
        # Tesseract and layout scores are not calibrated field-confidence scores.
        "confidence": None,
        "evidence": evidence,
    }


def _parse_date(candidate: str) -> tuple[str | None, str]:
    """Return ISO date and status; never guess ambiguous numeric day/month order."""
    value = candidate.strip().replace(",", "").replace(".", "")
    iso_match = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", value)
    if iso_match:
        year, month, day = map(int, iso_match.groups())
        try:
            return date(year, month, day).isoformat(), "extracted"
        except ValueError:
            return None, "uncertain"

    numeric_match = re.fullmatch(r"(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})", value)
    if numeric_match:
        first, second, year = map(int, numeric_match.groups())
        if year < 100:
            year += 2000
        if first <= 12 and second <= 12 and first != second:
            return None, "uncertain"
        if first > 12:
            day, month = first, second
        else:
            month, day = first, second
        try:
            return date(year, month, day).isoformat(), "extracted"
        except ValueError:
            return None, "uncertain"

    month_names = {
        name.lower(): number
        for number, names in enumerate(
            [
                (), ("jan", "january"), ("feb", "february"), ("mar", "march"),
                ("apr", "april"), ("may",), ("jun", "june"), ("jul", "july"),
                ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"),
                ("nov", "november"), ("dec", "december"),
            ]
        )
        for name in names
    }
    month_match = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{2,4})", value)
    if month_match:
        day, month_name, year = month_match.groups()
        month = month_names.get(month_name.lower())
    else:
        month_first_match = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2})\s+(\d{2,4})", value)
        if not month_first_match:
            return None, "uncertain"
        month_name, day, year = month_first_match.groups()
        month = month_names.get(month_name.lower())
    if month is None:
        return None, "uncertain"
    year_value = int(year)
    if year_value < 100:
        year_value += 2000
    try:
        return date(year_value, month, int(day)).isoformat(), "extracted"
    except ValueError:
        return None, "uncertain"


def _extract_labeled_name(lines: list[str], labels: str) -> dict[str, Any]:
    pattern = re.compile(rf"\b(?:{labels})\b\s*[:#-]\s*(.+)", re.IGNORECASE)
    candidates = [(match.group(1).strip(), line) for line in lines if (match := pattern.search(line))]
    candidates = list({value.casefold(): (value, evidence) for value, evidence in reversed(candidates)}.values())
    if len(candidates) == 1:
        value, evidence = candidates[0]
        return _field(value, extraction_status="extracted", evidence=evidence)
    if len(candidates) > 1:
        return _field(
            None,
            extraction_status="uncertain",
            evidence="; ".join(line for _, line in candidates),
        )
    return _field(None, extraction_status="missing")


def _name_alias(supplier: dict[str, Any], customer: dict[str, Any]) -> dict[str, Any]:
    """Expose the brief's generic name field without choosing between two parties."""
    supplier_status = supplier["extraction_status"]
    customer_status = customer["extraction_status"]
    extracted_parties = [
        party for party in (supplier, customer)
        if party["extraction_status"] == "extracted"
    ]
    if len(extracted_parties) == 1 and {supplier_status, customer_status} == {"extracted", "missing"}:
        party = extracted_parties[0]
        return _field(
            party["value"],
            extraction_status="extracted",
            evidence=party["evidence"],
        )
    if supplier_status == customer_status == "missing":
        return _field(None, extraction_status="missing")

    evidence = "; ".join(
        party["evidence"]
        for party in (supplier, customer)
        if party.get("evidence")
    ) or None
    return _field(None, extraction_status="uncertain", evidence=evidence)


def extract_fields(text: str) -> dict[str, Any]:
    """Extract labeled fields conservatively and report required-field review needs."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    normalized = "\n".join(lines)

    type_keywords = {
        "invoice": r"\binvoice\b",
        "receipt": r"\breceipt\b",
        "purchase_order": r"\bpurchase\s+order\b",
        "contract": r"\bcontract\b",
        "report": r"\breport\b",
    }
    detected_types = [
        name for name, pattern in type_keywords.items()
        if re.search(pattern, normalized, flags=re.IGNORECASE)
    ]
    if len(detected_types) == 1:
        document_type = _field(detected_types[0], extraction_status="extracted")
    elif len(detected_types) > 1:
        document_type = _field(
            None, extraction_status="uncertain", evidence=", ".join(detected_types)
        )
    else:
        document_type = _field(None, extraction_status="missing")

    supplier_name = _extract_labeled_name(lines, r"supplier|vendor|from")
    customer_name = _extract_labeled_name(lines, r"customer|bill\s+to|ship\s+to")
    name = _name_alias(supplier_name, customer_name)

    date_candidates = [(match.group(0), line) for line in lines for match in DATE_PATTERN.finditer(line)]
    date_candidates = list({value.casefold(): (value, evidence) for value, evidence in reversed(date_candidates)}.values())
    labeled_dates = [item for item in date_candidates if re.search(r"\bdate\b", item[1], re.I)]
    if len(labeled_dates) == 1:
        selected_dates = labeled_dates
    elif len(labeled_dates) > 1:
        selected_dates = labeled_dates
    else:
        selected_dates = date_candidates
    if len(selected_dates) == 1:
        raw_date, evidence = selected_dates[0]
        normalized_date, date_status = _parse_date(raw_date)
        date_field = _field(
            normalized_date or raw_date,
            extraction_status=date_status,
            evidence=evidence,
        )
    elif len(selected_dates) > 1:
        date_field = _field(
            None,
            extraction_status="uncertain",
            evidence="; ".join(line for _, line in selected_dates),
        )
    else:
        date_field = _field(None, extraction_status="missing")

    amount_candidates = []
    for line in lines:
        if AMOUNT_LINE.search(line):
            matches = MONEY.findall(line)
            if matches:
                amount_candidates.append((matches[-1].strip(), line))
    total_candidates = [item for item in amount_candidates if re.search(r"\btotal\b", item[1], re.I)]
    if len(total_candidates) == 1:
        chosen_amounts = total_candidates
    elif total_candidates:
        chosen_amounts = total_candidates
    else:
        chosen_amounts = amount_candidates
    if len(chosen_amounts) == 1:
        amount_value, evidence = chosen_amounts[0]
        amount = _field(amount_value, extraction_status="extracted", evidence=evidence)
    elif len(chosen_amounts) > 1:
        amount = _field(
            None,
            extraction_status="uncertain",
            evidence="; ".join(line for _, line in chosen_amounts),
        )
    else:
        amount = _field(None, extraction_status="missing")

    status_candidates = [
        (word.lower(), line)
        for line in lines
        for word in re.findall(r"\b(?:paid|unpaid|pending|overdue|cancelled|canceled)\b", line, re.I)
    ]
    unique_statuses = {word for word, _ in status_candidates}
    if len(unique_statuses) == 1:
        selected_status = next(
            (item for item in status_candidates if re.search(r"\bstatus\b", item[1], re.I)),
            status_candidates[0],
        )
        document_status = _field(
            selected_status[0], extraction_status="extracted", evidence=selected_status[1]
        )
    elif len(unique_statuses) > 1:
        document_status = _field(
            None,
            extraction_status="uncertain",
            evidence="; ".join(line for _, line in status_candidates),
        )
    else:
        document_status = _field(None, extraction_status="missing")

    fields = {
        "document_type": document_type,
        "name": name,
        "supplier_name": supplier_name,
        "customer_name": customer_name,
        "date": date_field,
        "amount": amount,
        "status": document_status,
    }
    selected_type = document_type["value"]
    required_fields = ("document_type",) + REQUIRED_FIELDS_BY_TYPE.get(selected_type, ())
    needs_review = any(
        fields[name]["extraction_status"] != "extracted" for name in required_fields
    )
    return {
        "fields": fields,
        "required_fields": list(required_fields),
        "needs_review": needs_review,
    }


def _section_answer(question: str, document_result: dict[str, Any]) -> dict[str, Any] | None:
    """Retrieve content under a matching layout heading, respecting columns where possible."""
    ignored = {
        "a", "an", "and", "are", "can", "could", "did", "does", "for", "from",
        "give", "how", "in", "is", "it", "me", "of", "on", "please", "show",
        "tell", "that", "the", "this", "to", "was", "were", "what", "which", "with",
    }
    question_terms = {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+", question)
        if token.casefold() not in ignored
    }
    if len(question_terms) < 2:
        return None

    heading_labels = {"section-header", "title", "caption"}
    content_labels = {"text", "table", "list-item", "caption", "footnote"}
    matches: list[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]] = []
    for page in document_result.get("pages", []):
        regions = page.get("regions", [])
        headings = [
            region for region in regions
            if region.get("layout_label", "").casefold() in heading_labels
            and region.get("text", "").strip()
        ]
        for heading in headings:
            heading_terms = {
                token.casefold()
                for token in re.findall(r"[A-Za-z0-9]+", heading.get("text", ""))
            }
            if not question_terms.issubset(heading_terms):
                continue
            box = heading.get("bbox_xyxy")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = box

            def same_column(other: dict[str, Any]) -> bool:
                other_box = other.get("bbox_xyxy")
                if not other_box or len(other_box) != 4:
                    return False
                ox1, _, ox2, _ = other_box
                overlap = max(0, min(x2, ox2) - max(x1, ox1))
                narrow_width = max(1, min(x2 - x1, ox2 - ox1))
                return overlap / narrow_width >= 0.25

            following_headings = [
                other for other in headings
                if other is not heading
                and other["bbox_xyxy"][1] >= y2
                and same_column(other)
            ]
            next_y = min((other["bbox_xyxy"][1] for other in following_headings), default=float("inf"))
            content = [
                region for region in regions
                if region.get("layout_label", "").casefold() in content_labels
                and region.get("text", "").strip()
                and region.get("bbox_xyxy")
                and region["bbox_xyxy"][1] >= y2 - 5
                and region["bbox_xyxy"][1] < next_y
                and same_column(region)
            ]
            content.sort(key=lambda region: (region["bbox_xyxy"][1], region["bbox_xyxy"][0]))
            matches.append((heading, content, page.get("regions", [])))

    if not matches:
        return None
    if len(matches) > 1:
        evidence = "\n".join(dict.fromkeys(item[0].get("text", "") for item in matches))
        return {
            "question": question,
            "answer": None,
            "status": "uncertain",
            "evidence": evidence,
            "message": "I found multiple matching sections and cannot tell which one you mean.",
        }

    heading, region_content, all_regions = matches[0]
    heading_text = heading.get("text", "").strip()
    content_text = [region.get("text", "").strip() for region in region_content]

    # Full-page OCR can retain numbered entries that the detector misses. Do
    # not use neighboring prose from that pass: multi-column lines can fuse.
    # Recover only numbered items after an Actions heading, then use the
    # same-column layout crops for other sections.
    answer_lines = content_text
    if re.search(r"\bactions?\b", heading_text, re.I):
        full_text_lines = [
            line.strip() for line in document_result.get("full_text", "").splitlines()
            if line.strip()
        ]
        normalized_heading = re.sub(r"\s+", " ", heading_text).casefold()
        heading_line_index = next(
            (
                index for index in range(len(full_text_lines) - 1, -1, -1)
                if normalized_heading in re.sub(r"\s+", " ", full_text_lines[index]).casefold()
            ),
            None,
        )
        numbered_items: list[str] = []
        if heading_line_index is not None:
            item_pattern = re.compile(r"(?<!\w)\d{1,2}[.)]?\s+([A-Z][^.!?]*(?:[.!?]|$))")
            for line in full_text_lines[heading_line_index + 1:]:
                if re.match(r"^(?:page\s+\d+|synthetic test report)\b", line, re.I):
                    break
                for match in item_pattern.finditer(line):
                    item = re.sub(r"\s+", " ", match.group(1)).strip()
                    if item and item not in numbered_items:
                        numbered_items.append(item)
        if numbered_items:
            answer_lines = numbered_items

    answer_lines = [re.sub(r"\s+", " ", line).strip() for line in answer_lines if line.strip()]
    answer = " ".join(answer_lines).strip()
    evidence = "\n".join([heading_text, *answer_lines]) if answer else heading_text
    if not answer:
        return {
            "question": question,
            "answer": None,
            "status": "uncertain",
            "evidence": evidence,
            "message": "I found the section heading but no readable text beneath it.",
        }
    return {
        "question": question,
        "answer": answer,
        "status": "answered",
        "evidence": evidence,
        "message": None,
    }


def answer_question(question: str, document_result: dict[str, Any]) -> dict[str, Any]:
    """Answer from fields, heading-aligned sections, or relevant OCR text with evidence."""
    question = question.strip()
    if not question:
        return {
            "question": question,
            "answer": None,
            "status": "uncertain",
            "evidence": None,
            "message": "Enter a question about the uploaded document.",
        }

    if re.search(r"\bwhen\b", question, re.IGNORECASE) and re.search(
        r"\b(payment|paid|due)\b", question, re.IGNORECASE
    ):
        return {
            "question": question,
            "answer": None,
            "status": "not_found",
            "evidence": None,
            "message": "Payment dates are not extracted as a separate field by this prototype.",
        }

    field_questions = (
        ("supplier_name", r"\b(supplier|vendor|seller|issued by)\b"),
        ("customer_name", r"\b(customer|buyer|bill to|billed to|ship to)\b"),
        ("name", r"\b(name|who is it from|who issued)\b"),
        ("date", r"\b(date|dated|when was the (?:invoice|document)|when was it issued)\b"),
        ("amount", r"\b(amount|total|cost|price|balance|how much|what is owed)\b"),
        ("status", r"\b(status|paid|unpaid|overdue)\b"),
        ("document_type", r"\b(document type|what kind|what type|classify)\b"),
    )
    fields = document_result.get("fields", {})
    for field_name, pattern in field_questions:
        if re.search(pattern, question, flags=re.IGNORECASE):
            field = fields.get(field_name, {})
            status = field.get("extraction_status", "missing")
            if status == "extracted":
                return {
                    "question": question,
                    "answer": field.get("value"),
                    "status": "answered",
                    "evidence": field.get("evidence"),
                    "message": None,
                }
            if status == "uncertain":
                return {
                    "question": question,
                    "answer": None,
                    "status": "uncertain",
                    "evidence": field.get("evidence"),
                    "message": "The document contains conflicting or ambiguous evidence.",
                }
            return {
                "question": question,
                "answer": None,
                "status": "not_found",
                "evidence": None,
                "message": "I could not find this field in the recognized document text.",
            }

    section_answer = _section_answer(question, document_result)
    if section_answer is not None:
        return section_answer

    if re.search(r"\b(description|items? purchased|items? bought|what was bought)\b", question, re.I):
        lines = [line.strip() for line in document_result.get("full_text", "").splitlines() if line.strip()]
        heading_indexes = [
            index for index, line in enumerate(lines)
            if re.match(r"^(?:item\s+)?description\s*:?\s*(.*)$", line, re.I)
        ]
        if len(heading_indexes) == 1:
            index = heading_indexes[0]
            heading = lines[index]
            inline = re.sub(r"^(?:item\s+)?description\s*:?\s*", "", heading, flags=re.I).strip()
            content = [inline] if inline else []
            boundary = re.compile(
                r"^(?:grand\s+)?total\b|^subtotal\b|^tax\b|^amount\s+due\b|"
                r"^balance\b|^status\b|^payment\s+status\b|^supplier\b|"
                r"^vendor\b|^customer\b|^bill\s+to\b|^invoice\s+date\b|^date\b",
                re.I,
            )
            for line in lines[index + 1:]:
                if boundary.search(line):
                    break
                content.append(line)
            if content:
                return {
                    "question": question,
                    "answer": " ".join(content),
                    "status": "answered",
                    "evidence": "\n".join([heading, *content]),
                    "message": None,
                }
            return {
                "question": question,
                "answer": None,
                "status": "uncertain",
                "evidence": heading,
                "message": "I found the description heading but no readable description text after it.",
            }
        if len(heading_indexes) > 1:
            matches = "\n".join(lines[index] for index in heading_indexes)
            return {
                "question": question,
                "answer": None,
                "status": "uncertain",
                "evidence": matches,
                "message": "I found multiple description sections and cannot tell which one you mean.",
            }

    stop_words = {
        "a", "an", "and", "are", "can", "could", "did", "does", "for", "from",
        "give", "in", "is", "it", "me", "of", "on", "please", "show", "tell",
        "that", "the", "this", "to", "was", "what", "which", "who", "with",
    }
    terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9]+", question)
        if term.lower() not in stop_words
    }
    lines = [line.strip() for line in document_result.get("full_text", "").splitlines() if line.strip()]
    scored_lines = []
    for line in lines:
        line_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9]+", line)}
        overlap = len(terms & line_terms)
        score = overlap / max(1, len(terms))
        if overlap:
            scored_lines.append((score, line))

    if not scored_lines:
        return {
            "question": question,
            "answer": None,
            "status": "not_found",
            "evidence": None,
            "message": "I could not find matching evidence in the recognized document text.",
        }
    scored_lines.sort(key=lambda item: item[0], reverse=True)
    best_score = scored_lines[0][0]
    best_lines = list(dict.fromkeys(line for score, line in scored_lines if score == best_score))
    if best_score < 0.5 or len(best_lines) > 1:
        return {
            "question": question,
            "answer": None,
            "status": "uncertain",
            "evidence": "\n".join(best_lines),
            "message": "I found possible text, but not enough unique evidence for a reliable answer.",
        }
    return {
        "question": question,
        "answer": best_lines[0],
        "status": "answered",
        "evidence": best_lines[0],
        "message": None,
    }


def _process_page(
    image: Any,
    model: YOLO,
    *,
    page_number: int,
    confidence: float,
    imgsz: int,
) -> dict[str, Any]:
    regions = detect_and_ocr(image, model, confidence=confidence, imgsz=imgsz)
    layout_crop_text = "\n".join(region["text"] for region in regions if region["text"])
    if len(regions) == 1 and regions[0]["layout_label"] == "full_page_fallback":
        full_page_text = regions[0]["text"]
    else:
        full_page_text = ocr_crop(image, scale=page_upscale(image.shape))
    merged_text = _merge_ocr_text(full_page_text, layout_crop_text)
    # Prefer whole-page reading order if it yields required fields; otherwise
    # supplement it with text recovered from layout crops.
    use_merged = extract_fields(full_page_text)["needs_review"]
    full_text = merged_text if use_merged else full_page_text
    return {
        "page_number": page_number,
        "regions": regions,
        "full_text": full_text,
        "text_source": "merged" if use_merged else "full_page",
        "ocr_text_variants": {
            "full_page": full_page_text,
            "layout_crop": layout_crop_text,
            "merged": merged_text,
        },
    }


def _merge_ocr_text(primary_text: str, supplemental_text: str) -> str:
    """Keep whole-page reading order and add crop OCR lines the page pass missed."""
    primary_lines = [line.strip() for line in primary_text.splitlines() if line.strip()]
    merged = list(primary_lines)
    normalized_primary = [re.sub(r"[^a-z0-9]", "", line.lower()) for line in primary_lines]
    for line in (line.strip() for line in supplemental_text.splitlines() if line.strip()):
        normalized = re.sub(r"[^a-z0-9]", "", line.lower())
        if not normalized:
            continue
        duplicate = any(
            normalized == existing
            or (min(len(normalized), len(existing)) >= 6
                and SequenceMatcher(None, normalized, existing).ratio() >= 0.88)
            for existing in normalized_primary
        )
        if not duplicate:
            merged.append(line)
            normalized_primary.append(normalized)
    return "\n".join(merged)


def process_image(
    image_path: Path,
    model: YOLO,
    confidence: float = 0.25,
    imgsz: int = 640,
) -> dict[str, Any]:
    """Process one image using an already-loaded layout detector."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    page = _process_page(image, model, page_number=1, confidence=confidence, imgsz=imgsz)
    fields_result = extract_fields(page["full_text"])
    result = {
        "source_file": image_path.name,
        "ocr_engine": "Tesseract",
        "text_source": page["text_source"],
        "preprocessing": {
            "upscale": page_upscale(image.shape), "white_margin_px": MARGIN, "psm": 6,
        },
        "pages": [page],
        "ocr_text_variants": page["ocr_text_variants"],
        "full_text": page["full_text"],
        **fields_result,
    }
    return validate_document_result(result)


def process_pdf(
    pdf_path: Path,
    model: YOLO,
    confidence: float = 0.25,
    imgsz: int = 640,
    dpi: int = 150,
) -> dict[str, Any]:
    """Render PDF pages with PyMuPDF, then use the same image pipeline."""
    import fitz
    import numpy as np

    pages = []
    scale = dpi / 72
    with fitz.open(pdf_path) as pdf:
        for index, pdf_page in enumerate(pdf, start=1):
            pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n
            )
            image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pages.append(
                _process_page(
                    image_bgr, model, page_number=index,
                    confidence=confidence, imgsz=imgsz,
                )
            )
    full_text = "\n".join(page["full_text"] for page in pages if page["full_text"])
    page_sources = {page["text_source"] for page in pages}
    text_source = next(iter(page_sources)) if len(page_sources) == 1 else "mixed"
    text_variants = {
        variant: "\n".join(
            page["ocr_text_variants"][variant]
            for page in pages
            if page["ocr_text_variants"][variant]
        )
        for variant in ("full_page", "layout_crop", "merged")
    }
    result = {
        "source_file": pdf_path.name,
        "ocr_engine": "Tesseract",
        "text_source": text_source,
        "preprocessing": {"dpi": dpi, "white_margin_px": MARGIN, "psm": 6},
        "pages": pages,
        "ocr_text_variants": text_variants,
        "full_text": full_text,
        **extract_fields(full_text),
    }
    return validate_document_result(result)


def process_document(
    file_path: Path,
    model: YOLO,
    *,
    confidence: float = 0.25,
    imgsz: int = 640,
    pdf_dpi: int = 150,
) -> dict[str, Any]:
    """Dispatch supported image/PDF files to their shared inference pipeline."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    if file_path.suffix.lower() == ".pdf":
        result = process_pdf(file_path, model, confidence, imgsz, pdf_dpi)
    else:
        result = process_image(file_path, model, confidence, imgsz)
    return validate_document_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect document regions and extract OCR text.")
    parser.add_argument("file", type=Path, help="Path to an image or PDF document")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to the trained YOLO weights")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size (e.g. 640 or 1024)")
    parser.add_argument("--pdf-dpi", type=int, default=150, help="PDF page rendering resolution")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save the JSON result")
    parser.add_argument(
        "--tesseract-cmd",
        default=os.environ.get("TESSERACT_CMD"),
        help="Tesseract executable path; defaults to PATH or TESSERACT_CMD environment variable",
    )
    args = parser.parse_args()

    if configure_tesseract(args.tesseract_cmd) is None:
        raise RuntimeError(
            "Tesseract executable not found. Install Tesseract or pass --tesseract-cmd."
        )

    model = load_model(args.weights)
    result = process_document(
        args.file,
        model,
        confidence=args.conf,
        imgsz=args.imgsz,
        pdf_dpi=args.pdf_dpi,
    )
    result = DocumentResult.model_validate(result).model_dump(mode="python")
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Saved: {args.output}")
    print(rendered)


if __name__ == "__main__":
    main()
