"""Create a clean known-good invoice for manually testing the Streamlit app."""

from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1275, 1650


def get_font(size: int, bold: bool = False):
    path = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def create_image() -> Path:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    navy, black, gray = "#17324D", "#111111", "#526170"
    draw.rectangle((0, 0, WIDTH, 265), fill=navy)
    draw.text((105, 55), "NORTHSTAR SUPPLY", font=get_font(48, True), fill="white")
    draw.text((105, 135), "INVOICE", font=get_font(38, True), fill="#DCEBFA")

    draw.text((105, 340), "Invoice Number: NS-2026-1042", font=get_font(31), fill=gray)
    draw.text((105, 430), "Supplier: Northstar Supply", font=get_font(38, True), fill=black)
    draw.text((105, 535), "Invoice Date: 2026-10-31", font=get_font(38), fill=black)
    draw.line((105, 630, 1170, 630), fill="#B8C4D0", width=3)

    draw.text((105, 710), "Description", font=get_font(30, True), fill=gray)
    draw.text((105, 775), "Office paper and printer supplies", font=get_font(34), fill=black)
    draw.text((105, 930), "Total: $1,248.50", font=get_font(44, True), fill=navy)
    draw.text((105, 1050), "Status: Unpaid", font=get_font(36), fill=black)
    draw.text((105, 1500), "Synthetic test document — no real person or business", font=get_font(22), fill=gray)

    path = HERE / "best_case_invoice.png"
    image.save(path, optimize=True)
    pdf_path = HERE / "best_case_invoice.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=612, height=792)
    page.insert_image(page.rect, filename=str(path))
    pdf.save(pdf_path)
    pdf.close()
    return path


if __name__ == "__main__":
    image_path = create_image()
    print(f"Created {image_path.name} and best_case_invoice.pdf")
