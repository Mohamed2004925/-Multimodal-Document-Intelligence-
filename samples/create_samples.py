"""Regenerate the synthetic invoice image and image-only PDF sample."""

from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1200, 1600


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / filename
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def create_invoice_image() -> Path:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    navy = "#183153"
    muted = "#5F6B7A"
    light = "#EAF0F6"
    draw.rectangle((0, 0, WIDTH, 250), fill=navy)
    draw.text((80, 58), "NORTHSTAR OFFICE SUPPLY", font=font(38, True), fill="white")
    draw.text((80, 120), "42 Market Street, Boston, MA 02110", font=font(25), fill="#DCE7F2")
    draw.text((850, 65), "INVOICE", font=font(54, True), fill="white")

    draw.text((80, 300), "INVOICE NO.", font=font(20, True), fill=muted)
    draw.text((80, 335), "NS-2026-1042", font=font(30, True), fill=navy)
    draw.text((630, 300), "INVOICE DATE", font=font(20, True), fill=muted)
    draw.text((630, 335), "October 31, 2026", font=font(30, True), fill=navy)

    draw.rounded_rectangle((80, 430, 1120, 665), radius=18, fill="#F7F9FC", outline=light, width=3)
    draw.text((115, 460), "Supplier: Northstar Office Supply", font=font(25, True), fill=navy)
    draw.text((115, 495), "42 Market Street, Boston, MA 02110", font=font(22), fill=muted)
    draw.text((115, 545), "Customer: Example Design Studio", font=font(25, True), fill=navy)
    draw.text((115, 580), "18 River Road, Cambridge, MA 02139", font=font(22), fill=muted)

    draw.rectangle((80, 720, 1120, 785), fill=navy)
    draw.text((110, 738), "DESCRIPTION", font=font(22, True), fill="white")
    draw.text((830, 738), "QTY", font=font(22, True), fill="white")
    draw.text((960, 738), "AMOUNT", font=font(22, True), fill="white")
    draw.text((110, 825), "Printer paper, 10 reams", font=font(26), fill=navy)
    draw.text((850, 825), "10", font=font(26), fill=navy)
    draw.text((960, 825), "$1,248.50", font=font(26), fill=navy)
    draw.line((80, 885, 1120, 885), fill=light, width=3)

    draw.text((720, 960), "Subtotal", font=font(24), fill=muted)
    draw.text((960, 960), "$1,150.00", font=font(24), fill=navy)
    draw.text((720, 1015), "Tax", font=font(24), fill=muted)
    draw.text((960, 1015), "$98.50", font=font(24), fill=navy)
    draw.rounded_rectangle((680, 1080, 1120, 1170), radius=12, fill="#EAF7F1")
    draw.text((715, 1105), "TOTAL: $1,248.50", font=font(31, True), fill="#176B4D")

    draw.text((80, 1250), "PAYMENT STATUS: UNPAID", font=font(27, True), fill="#A94B16")
    draw.text((80, 1310), "Please remit payment within 30 days.", font=font(23), fill=muted)
    draw.text((80, 1480), "SYNTHETIC SAMPLE — NOT A REAL INVOICE", font=font(18, True), fill="#8A94A3")

    output = HERE / "sample_invoice.png"
    image.save(output, optimize=True)
    return output


def create_image_pdf(image_path: Path) -> Path:
    jpeg_path = HERE / "sample_invoice.jpg"
    Image.open(image_path).convert("RGB").save(jpeg_path, quality=88, optimize=True)
    pdf = fitz.open()
    page = pdf.new_page(width=612, height=816)
    page.insert_image(page.rect, filename=str(jpeg_path))
    output = HERE / "sample_invoice.pdf"
    pdf.save(output)
    pdf.close()
    return output


if __name__ == "__main__":
    png = create_invoice_image()
    pdf = create_image_pdf(png)
    print(f"Created synthetic sample documents: {png.name}, sample_invoice.jpg, {pdf.name}")
