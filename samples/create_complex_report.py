"""Generate a synthetic, non-invoice report image for OCR/layout testing."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1800, 1900
NAVY = "#18324B"
BLUE = "#2E6F95"
INK = "#263442"
MUTED = "#627282"
PALE = "#EAF1F6"
GRID = "#C9D4DD"
GREEN = "#4C8B72"
ORANGE = "#D08B3F"


def font(size: int, bold: bool = False):
    filename = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / filename
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def wrap(draw: ImageDraw.ImageDraw, text: str, selected_font, max_width: int) -> list[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=selected_font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def paragraph(draw, text: str, x: int, y: int, width: int, size: int = 27, color: str = INK, leading: int = 11) -> int:
    selected = font(size)
    for line in wrap(draw, text, selected, width):
        draw.text((x, y), line, font=selected, fill=color)
        y += size + leading
    return y


def section_title(draw, text: str, x: int, y: int, width: int) -> int:
    draw.text((x, y), text, font=font(34, True), fill=NAVY)
    draw.line((x, y + 48, x + width, y + 48), fill=BLUE, width=3)
    return y + 68


def make_report() -> Path:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    margin, gap = 105, 70
    column_width = (WIDTH - 2 * margin - gap) // 2
    left_x, right_x = margin, margin + column_width + gap

    # Running header and report title.
    draw.rectangle((0, 0, WIDTH, 255), fill=NAVY)
    draw.text((margin, 38), "NORTHSTAR COMMUNITY SERVICES", font=font(25, True), fill="#C6D8E6")
    draw.text((margin, 92), "Quarterly Operations Report", font=font(55, True), fill="white")
    draw.text((margin, 173), "Facilities and service performance review", font=font(28), fill="#D8E4EC")

    # Metadata band.
    draw.rectangle((margin, 288, WIDTH - margin, 402), fill=PALE)
    metadata = [
        ("REPORTING PERIOD", "January–March 2026"),
        ("PUBLISHED", "April 15, 2026"),
        ("STATUS", "Pending review"),
    ]
    box_width = (WIDTH - 2 * margin) // 3
    for index, (label, value) in enumerate(metadata):
        x = margin + index * box_width + 24
        draw.text((x, 308), label, font=font(19, True), fill=MUTED)
        draw.text((x, 342), value, font=font(26, True), fill=NAVY)

    # Executive summary spans the page width.
    y = 445
    y = section_title(draw, "Executive summary", margin, y, WIDTH - 2 * margin)
    y = paragraph(
        draw,
        "Service teams handled 184 facility requests this quarter. Response times improved at four of five sites, while equipment availability remained below target at Riverside. The figures in this report are synthetic and are provided only to test document reading and layout extraction.",
        margin, y, WIDTH - 2 * margin, size=27,
    )

    # Two-column report body begins.
    body_y = y + 38
    left_y = section_title(draw, "Service performance", left_x, body_y, column_width)
    left_y = paragraph(
        draw,
        "The median time to acknowledge a request fell from 5.2 hours to 4.1 hours. Preventive maintenance completion reached 92 percent, slightly above the quarterly target.",
        left_x, left_y, column_width, size=25,
    )
    left_y += 25

    # Table with clear row and column boundaries.
    table_x, table_y = left_x, left_y
    widths = [190, 125, 135, column_width - 450]
    headers = ["SITE", "REQUESTS", "RESOLVED", "MEDIAN RESPONSE"]
    row_height = 64
    draw.rectangle((table_x, table_y, table_x + column_width, table_y + row_height), fill=NAVY)
    x = table_x
    for header, cell_width in zip(headers, widths):
        draw.text((x + 10, table_y + 20), header, font=font(17, True), fill="white")
        x += cell_width
    rows = [
        ("Central", "46", "43", "3.6 hours"),
        ("Riverside", "39", "33", "6.8 hours"),
        ("East Market", "52", "49", "3.9 hours"),
        ("Hilltop", "27", "25", "4.4 hours"),
        ("West Annex", "20", "19", "3.7 hours"),
    ]
    for row_index, row in enumerate(rows):
        top = table_y + row_height * (row_index + 1)
        fill = "#F5F8FA" if row_index % 2 == 0 else "white"
        draw.rectangle((table_x, top, table_x + column_width, top + row_height), fill=fill, outline=GRID, width=1)
        x = table_x
        for value, cell_width in zip(row, widths):
            draw.text((x + 10, top + 19), value, font=font(21), fill=INK)
            x += cell_width
    left_y = table_y + row_height * (len(rows) + 1) + 28

    left_y = section_title(draw, "Key observations", left_x, left_y, column_width)
    observations = [
        "Riverside recorded the longest response time and the highest number of repeat requests.",
        "East Market closed 94 percent of requests within the internal service target.",
        "Three planned maintenance visits moved to April because replacement parts arrived late.",
    ]
    for item in observations:
        draw.ellipse((left_x + 2, left_y + 11, left_x + 15, left_y + 24), fill=BLUE)
        left_y = paragraph(draw, item, left_x + 28, left_y, column_width - 28, size=24) + 16

    # Right column: chart, findings, actions.
    right_y = section_title(draw, "Requests by month", right_x, body_y, column_width)
    right_y = paragraph(draw, "Submitted service requests, January to March", right_x, right_y, column_width, size=23, color=MUTED)
    chart_top, chart_bottom = right_y + 28, right_y + 365
    chart_left, chart_right = right_x + 70, right_x + column_width - 30
    for tick, label in zip((0, 25, 50, 75), ("0", "25", "50", "75")):
        yy = chart_bottom - int((chart_bottom - chart_top) * tick / 75)
        draw.line((chart_left, yy, chart_right, yy), fill=GRID, width=2)
        draw.text((right_x + 8, yy - 14), label, font=font(18), fill=MUTED)
    values = [52, 61, 71]
    labels = ["January", "February", "March"]
    bar_width = 115
    centers = [chart_left + 125, chart_left + 315, chart_left + 505]
    for center, value, label in zip(centers, values, labels):
        bar_height = int((chart_bottom - chart_top) * value / 75)
        draw.rounded_rectangle((center, chart_bottom - bar_height, center + bar_width, chart_bottom), radius=7, fill=BLUE)
        draw.text((center + 31, chart_bottom - bar_height - 37), str(value), font=font(22, True), fill=NAVY)
        draw.text((center - 16, chart_bottom + 18), label, font=font(19), fill=INK)
    right_y = chart_bottom + 82

    right_y = section_title(draw, "Risk and follow-up", right_x, right_y, column_width)
    right_y = paragraph(
        draw,
        "Parts availability is the main operational risk for the next quarter. Riverside also needs a review of repeat requests to determine whether the issue is related to equipment age, scheduling, or reporting practice.",
        right_x, right_y, column_width, size=24,
    )
    right_y += 22
    right_y = section_title(draw, "Actions for next quarter", right_x, right_y, column_width)
    actions = [
        "Complete the three delayed maintenance visits by April 30.",
        "Review Riverside repeat requests at the May operations meeting.",
        "Maintain a monthly stock check for critical replacement parts.",
    ]
    for index, action in enumerate(actions, start=1):
        draw.ellipse((right_x, right_y + 2, right_x + 36, right_y + 38), fill=PALE)
        draw.text((right_x + 11, right_y + 6), str(index), font=font(19, True), fill=BLUE)
        right_y = paragraph(draw, action, right_x + 52, right_y, column_width - 52, size=24) + 18

    # Footer, page marker, and synthetic-data note.
    footer_y = HEIGHT - 120
    draw.line((margin, footer_y, WIDTH - margin, footer_y), fill=GRID, width=2)
    draw.text((margin, footer_y + 24), "SYNTHETIC TEST REPORT — NOT AN ACTUAL ORGANIZATION OR PERFORMANCE RECORD", font=font(19, True), fill=MUTED)
    draw.text((WIDTH - margin - 100, footer_y + 24), "PAGE 1 OF 1", font=font(18), fill=MUTED)

    path = ROOT / "complex_operations_report.png"
    image.save(path, optimize=True)
    return path


if __name__ == "__main__":
    print(f"Created {make_report()}")
