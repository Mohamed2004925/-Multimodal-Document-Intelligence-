"""Create deterministic synthetic invoice/receipt fixtures and their ground truth.

These are development fixtures, not a representative accuracy benchmark.
Run with: python samples/create_evaluation_samples.py
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evaluation"
OUT.mkdir(exist_ok=True)
FONT = Path("C:/Windows/Fonts/arial.ttf")
BOLD = Path("C:/Windows/Fonts/arialbd.ttf")

CASES = [
    ("invoice_01", "invoice", "Northstar Supply", "Example Studio", "2026-10-31", "$1,248.50", "UNPAID", "normal"),
    ("invoice_02", "invoice", "Acme Paper Co", "Bluebird Office", "2026-05-14", "$235.00", "PAID", "normal"),
    ("invoice_03", "invoice", "Cedar Works", "Mason Design", "2026-02-19", "$89.95", "PENDING", "normal"),
    ("invoice_04", "invoice", "Harbor Tools", "Weston Build", "2026-09-08", "$3,405.20", "UNPAID", "normal"),
    ("invoice_05", "invoice", "Maple Print Shop", "Red Kite Media", "2026-12-01", "$610.00", "PAID", "normal"),
    ("receipt_06", "receipt", "Green Market", None, "2026-08-17", "$42.18", "PAID", "normal"),
    ("receipt_07", "receipt", "Corner Cafe", None, "2026-07-03", "$18.50", "PAID", "normal"),
    ("invoice_08_missing_total", "invoice", "Acme Paper Co", "Bluebird Office", "2026-05-14", None, "UNPAID", "missing_total"),
    ("invoice_09_ambiguous_date", "invoice", "Cedar Works", "Mason Design", "03/04/2026", "$89.95", "PENDING", "ambiguous_date"),
    ("receipt_10_conflicting_totals", "receipt", "Corner Cafe", None, "2026-07-03", "$18.50", "PAID", "conflicting_totals"),
]


def font(size: int, bold: bool = False):
    path = BOLD if bold else FONT
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def create(case: tuple) -> dict:
    name, kind, supplier, customer, date, amount, status, edge = case
    image = Image.new("RGB", (1200, 1000), "white")
    draw = ImageDraw.Draw(image)
    heading = kind.upper()
    if kind == "receipt":
        rows = [("Supplier", supplier), ("Receipt Date", date)]
    else:
        rows = [("Supplier", supplier), ("Customer", customer), ("Invoice Date", date)]
    draw.text((75, 45), heading, font=font(52, True), fill="#17324d")
    y = 170
    for label, value in rows:
        draw.text((80, y), f"{label}: {value}", font=font(34), fill="#111111")
        y += 82
    if edge == "conflicting_totals":
        draw.text((80, y), "Total: $16.50", font=font(34), fill="#111111")
        draw.text((80, y + 82), f"Grand Total: {amount}", font=font(34, True), fill="#111111")
    elif amount:
        label = "Total" if kind == "receipt" else "Amount Due"
        draw.text((80, y), f"{label}: {amount}", font=font(34, True), fill="#111111")
    if status:
        draw.text((80, 850), f"Status: {status}", font=font(31), fill="#111111")
    if edge == "ambiguous_date":
        # Use the intentionally ambiguous numeric form exactly as ground truth.
        pass
    if edge == "blurry":
        image = image.filter(ImageFilter.GaussianBlur(radius=2.2))
    path = OUT / f"{name}.png"
    image.save(path)
    return {
        "file": str(path.relative_to(ROOT.parent)).replace("\\", "/"),
        "document_type": kind,
        "supplier_name": supplier,
        "customer_name": customer,
        "date": date,
        "amount": None if edge in {"missing_total", "conflicting_totals"} else amount,
        "status": status.lower() if status else None,
        "scenario": edge,
        "expected_statuses": {
            "document_type": "extracted",
            "supplier_name": "extracted",
            "customer_name": "extracted" if customer else "missing",
            "date": "uncertain" if edge == "ambiguous_date" else "extracted",
            "amount": "missing" if edge == "missing_total" else "uncertain" if edge == "conflicting_totals" else "extracted",
            "status": "extracted",
        },
    }


if __name__ == "__main__":
    truth = [create(case) for case in CASES]
    (OUT / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"Created {len(truth)} synthetic document fixtures in {OUT}")
