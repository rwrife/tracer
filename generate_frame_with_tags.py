import argparse
import base64
import io
from pathlib import Path

import numpy as np
import svgwrite
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

# AprilTag generator
from moms_apriltag import TagGenerator2

# ======================================================================
# FRAME LAYOUT CONSTANTS
# ======================================================================

OUTER_WIDTH_MM = 350
OUTER_HEIGHT_MM = 450

INNER_X0 = 25
INNER_Y0 = 25
INNER_X1 = 325
INNER_Y1 = 425

TAG_POSITIONS = {
    0: (25, 25),
    1: (175, 25),
    2: (325, 25),
    3: (25, 225),
    4: (325, 225),
    5: (25, 425),
    6: (175, 425),
    7: (325, 425),
}

DEFAULT_TAG_SIZE_MM = 20.0
DEFAULT_FAMILY = "tag36h11"


# ======================================================================
# APRILTAG IMAGE GENERATION
# ======================================================================

def generate_tag_image(family: str, tag_id: int, img_px: int = 512):
    """
    Generate a PIL image of an AprilTag with pixel-perfect edges.
    """
    print(f"  -> Generating AprilTag {family} ID {tag_id}")
    tg = TagGenerator2(family)
    arr = tg.generate(tag_id)

    img = Image.fromarray(np.array(arr, dtype=np.uint8), mode="L")
    img = img.resize((img_px, img_px), Image.NEAREST)
    return img


def pil_to_data_uri(img: Image.Image) -> str:
    """
    Convert PIL image to inline SVG/HTML data URI.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


# ======================================================================
# SVG OUTPUT
# ======================================================================

def generate_svg(path: str, family: str, tag_size_mm: float):
    print(f"\n[SVG] Rendering SVG ({path})")
    print(f"     AprilTag Family: {family}")
    print(f"     Tag Size: {tag_size_mm}mm\n")

    dwg = svgwrite.Drawing(
        path,
        size=(f"{OUTER_WIDTH_MM}mm", f"{OUTER_HEIGHT_MM}mm"),
        viewBox=f"0 0 {OUTER_WIDTH_MM} {OUTER_HEIGHT_MM}",
    )

    # Title text
    dwg.add(
        dwg.text(
            f"Tracing Frame – 350×450mm, Inner 300×400mm, AprilTag {family}, {tag_size_mm}mm",
            insert=(OUTER_WIDTH_MM / 2, 10),
            text_anchor="middle",
            font_size="6px",
        )
    )

    # Outer border
    dwg.add(
        dwg.rect(
            (0, 0),
            (OUTER_WIDTH_MM, OUTER_HEIGHT_MM),
            fill="none",
            stroke="black",
            stroke_width=0.5,
        )
    )

    # Inner window
    dwg.add(
        dwg.rect(
            (INNER_X0, INNER_Y0),
            (INNER_X1 - INNER_X0, INNER_Y1 - INNER_Y0),
            fill="none",
            stroke="black",
            stroke_dasharray="3,3",
            stroke_width=0.4,
        )
    )

    half = tag_size_mm / 2

    for tag_id, (cx, cy) in TAG_POSITIONS.items():
        img = generate_tag_image(family, tag_id)
        href = pil_to_data_uri(img)

        x = cx - half
        y = cy - half

        # Tag image
        dwg.add(
            dwg.image(
                href=href,
                insert=(x, y),
                size=(tag_size_mm, tag_size_mm),
            )
        )

        # Text label
        dwg.add(
            dwg.text(
                f"ID {tag_id}",
                insert=(cx, cy + half + 6),
                text_anchor="middle",
                font_size="5px",
            )
        )

    dwg.save()
    print(f"[OK] SVG saved → {path}")


# ======================================================================
# PDF OUTPUT
# ======================================================================

def generate_pdf(path: str, family: str, tag_size_mm: float):
    print(f"\n[PDF] Rendering PDF ({path})")
    print(f"     AprilTag Family: {family}")
    print(f"     Tag Size: {tag_size_mm}mm\n")

    c = canvas.Canvas(path, pagesize=(OUTER_WIDTH_MM * mm, OUTER_HEIGHT_MM * mm))

    # Title
    c.setFont("Helvetica", 8)
    c.drawCentredString(
        OUTER_WIDTH_MM * mm / 2,
        OUTER_HEIGHT_MM * mm - 12,
        f"Tracing Frame – 350×450mm, Inner 300×400mm, AprilTag {family}, {tag_size_mm}mm",
    )

    # Outer border
    c.setLineWidth(0.5)
    c.rect(0.5 * mm, 0.5 * mm,
           (OUTER_WIDTH_MM - 1) * mm,
           (OUTER_HEIGHT_MM - 1) * mm)

    # Inner window
    c.setDash(3, 3)
    c.rect(
        INNER_X0 * mm,
        INNER_Y0 * mm,
        (INNER_X1 - INNER_X0) * mm,
        (INNER_Y1 - INNER_Y0) * mm,
    )
    c.setDash()

    half = tag_size_mm / 2

    for tag_id, (cx, cy) in TAG_POSITIONS.items():
        img = generate_tag_image(family, tag_id)
        img_reader = ImageReader(img)

        x = (cx - half) * mm
        y = (cy - half) * mm

        c.drawImage(
            img_reader,
            x,
            y,
            tag_size_mm * mm,
            tag_size_mm * mm,
        )

        c.drawString(cx * mm - 4 * mm, (cy + half + 6) * mm, f"ID {tag_id}")

    c.save()
    print(f"[OK] PDF saved → {path}")


# ======================================================================
# MAIN ENTRY POINT
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default=DEFAULT_FAMILY)
    parser.add_argument("--tag-size-mm", type=float, default=DEFAULT_TAG_SIZE_MM)
    parser.add_argument("--output-prefix", default="frame_layout")
    args = parser.parse_args()

    svg_path = f"{args.output_prefix}.svg"
    pdf_path = f"{args.output_prefix}.pdf"

    generate_svg(svg_path, args.family, args.tag_size_mm)
    generate_pdf(pdf_path, args.family, args.tag_size_mm)


if __name__ == "__main__":
    main()
