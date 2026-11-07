"""Create a short MP4 walkthrough from the real sample output and app workflow."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1280, 720
NAVY = "#183153"
BLUE = "#3478D4"
GREEN = "#16845B"
INK = "#202B3C"
MUTED = "#68778A"
BG = "#F3F6FA"
WHITE = "#FFFFFF"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    return ImageFont.truetype(str(path), size=size) if path.exists() else ImageFont.load_default(size=size)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int = 22,
         color: str = INK, bold: bool = False) -> None:
    draw.text(xy, value, font=font(size, bold), fill=color)


def base_scene(step: int, title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, WIDTH, 76), fill=NAVY)
    text(draw, (44, 22), "DOCUMENT INTELLIGENCE", 26, WHITE, True)
    text(draw, (902, 26), "LOCAL STREAMLIT DEMO", 18, "#C8D8EB", True)
    text(draw, (50, 100), title, 32, NAVY, True)
    draw.rounded_rectangle((50, 158, 495, 665), radius=18, fill=WHITE, outline="#DDE5EF", width=2)
    draw.rounded_rectangle((520, 158, 1230, 665), radius=18, fill=WHITE, outline="#DDE5EF", width=2)
    text(draw, (76, 176), "SAMPLE DOCUMENT", 16, MUTED, True)
    text(draw, (546, 176), f"STEP {step} OF 4", 16, BLUE, True)
    return canvas, draw


def add_invoice_preview(canvas: Image.Image) -> None:
    sample = Image.open(ROOT / "samples" / "sample_invoice.png").convert("RGB")
    sample.thumbnail((385, 430))
    x = 80 + (385 - sample.width) // 2
    y = 212 + (430 - sample.height) // 2
    canvas.paste(sample, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((76, 210, 469, 647), radius=8, outline="#DDE5EF", width=2)
    draw.rounded_rectangle((92, 614, 450, 646), radius=8, fill="#EDF4FF")
    text(draw, (105, 620), "sample_invoice.pdf  •  1 page", 15, BLUE, True)


def make_scenes() -> list[Image.Image]:
    result_path = ROOT / "output_examples" / "sample_invoice_pdf_output.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    fields = result["fields"]
    scenes: list[Image.Image] = []

    canvas, draw = base_scene(1, "Upload an image or PDF")
    add_invoice_preview(canvas)
    draw = ImageDraw.Draw(canvas)
    text(draw, (550, 228), "Choose a file from your computer", 24, INK, True)
    draw.rounded_rectangle((555, 282, 1195, 392), radius=14, fill="#F6F9FD", outline="#BFD3ED", width=2)
    text(draw, (588, 309), "sample_invoice.pdf", 24, NAVY, True)
    text(draw, (588, 348), "PDF document selected", 17, MUTED)
    draw.rounded_rectangle((555, 430, 820, 488), radius=12, fill=BLUE)
    text(draw, (590, 446), "Analyze document", 20, WHITE, True)
    text(draw, (555, 535), "The existing layout model and Tesseract OCR run locally.", 18, MUTED)
    scenes.append(canvas)

    canvas, draw = base_scene(2, "Review extracted fields")
    add_invoice_preview(canvas)
    draw = ImageDraw.Draw(canvas)
    text(draw, (550, 225), "Document type: invoice", 23, NAVY, True)
    y = 277
    display_names = {
        "supplier_name": "Supplier",
        "customer_name": "Customer",
        "date": "Date",
        "amount": "Total amount",
        "status": "Payment status",
    }
    for name, label in display_names.items():
        value = fields[name].get("value") or "Not found"
        draw.rounded_rectangle((550, y, 1195, y + 54), radius=9, fill="#F7F9FC")
        text(draw, (570, y + 16), label, 17, MUTED, True)
        text(draw, (840, y + 15), str(value), 18, INK)
        text(draw, (1090, y + 16), fields[name]["extraction_status"], 14, GREEN, True)
        y += 68
    draw.rounded_rectangle((550, 625, 1195, 655), radius=8, fill="#E8F6EF")
    text(draw, (570, 631), "Required fields extracted • verify against the source", 15, GREEN, True)
    scenes.append(canvas)

    canvas, draw = base_scene(3, "Ask a question about the document")
    add_invoice_preview(canvas)
    draw = ImageDraw.Draw(canvas)
    qa = result["question_answers"][-1]
    draw.rounded_rectangle((555, 240, 1195, 319), radius=12, fill="#F3F7FC")
    text(draw, (580, 255), qa["question"], 21, NAVY, True)
    draw.rounded_rectangle((555, 342, 1195, 435), radius=12, fill="#EAF7F1")
    text(draw, (580, 358), "ANSWER", 15, GREEN, True)
    text(draw, (580, 386), str(qa["answer"]), 27, INK, True)
    text(draw, (555, 472), "Evidence from OCR", 17, MUTED, True)
    text(draw, (555, 505), f'“{qa["evidence"]}”', 22, NAVY)
    draw.rounded_rectangle((555, 568, 770, 615), radius=10, fill=BLUE)
    text(draw, (582, 580), "Status: answered", 17, WHITE, True)
    text(draw, (555, 630), "No supporting text? The app labels the answer not found or uncertain.", 15, MUTED)
    scenes.append(canvas)

    canvas, draw = base_scene(4, "Download consistent JSON output")
    add_invoice_preview(canvas)
    draw = ImageDraw.Draw(canvas)
    text(draw, (555, 225), "Structured result", 22, NAVY, True)
    json_text = json.dumps(
        {
            "source_file": result["source_file"],
            "fields": {
                name: {"value": field["value"], "status": field["extraction_status"]}
                for name, field in fields.items()
            },
            "needs_review": result["needs_review"],
            "question_answers": result["question_answers"],
        },
        indent=2,
        ensure_ascii=False,
    )
    draw.rounded_rectangle((555, 270, 1195, 590), radius=12, fill="#172337")
    y = 291
    for line in json_text.splitlines()[:13]:
        text(draw, (580, y), line, 16, "#D7E6F7")
        y += 22
    draw.rounded_rectangle((555, 610, 855, 654), radius=10, fill=GREEN)
    text(draw, (580, 620), "Download structured JSON", 17, WHITE, True)
    scenes.append(canvas)

    return scenes


def main() -> None:
    scenes = make_scenes()
    output = Path(__file__).resolve().parent / "document_intelligence_demo.mp4"
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError("This OpenCV build cannot write MP4 video files.")

    for index, scene in enumerate(scenes):
        frame = cv2.cvtColor(np.array(scene), cv2.COLOR_RGB2BGR)
        for _ in range(36):
            writer.write(frame)
        if index < len(scenes) - 1:
            next_frame = cv2.cvtColor(np.array(scenes[index + 1]), cv2.COLOR_RGB2BGR)
            for blend_index in range(1, 13):
                amount = blend_index / 13
                writer.write(cv2.addWeighted(frame, 1 - amount, next_frame, amount, 0))
    writer.release()
    print(f"Created {output} ({len(scenes)} workflow scenes).")


if __name__ == "__main__":
    main()
