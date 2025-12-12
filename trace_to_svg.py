import cv2
import numpy as np
import svgwrite
import time
from pupil_apriltags import Detector
from pathlib import Path

# ============================================================
# FRAME CONSTANTS — match your printed layout
# ============================================================

# AprilTag spacing configuration (in millimeters)
# These define the physical distance between tag centers
TAG_SPACING_CONFIG = {
    'width': 300.0,      # Horizontal distance between left and right tags (mm)
    'height': 400.0,     # Vertical distance between top and bottom tags (mm)
    'offset_x': 25.0,    # X offset from origin to top-left tag (mm)
    'offset_y': 25.0     # Y offset from origin to top-left tag (mm)
}

INNER_X0, INNER_Y0 = TAG_SPACING_CONFIG['offset_x'], TAG_SPACING_CONFIG['offset_y']
INNER_X1, INNER_Y1 = INNER_X0 + TAG_SPACING_CONFIG['width'], INNER_Y0 + TAG_SPACING_CONFIG['height']

INNER_WIDTH_MM  = TAG_SPACING_CONFIG['width']    # 300mm
INNER_HEIGHT_MM = TAG_SPACING_CONFIG['height']   # 400mm

PX_PER_MM = 10
OUT_W = int(INNER_WIDTH_MM * PX_PER_MM)   # 3000px
OUT_H = int(INNER_HEIGHT_MM * PX_PER_MM)  # 4000px

# Tag IDs marking the inner boundary
CORNER_MAP = {
    0: "tl",
    2: "tr",
    7: "br",
    5: "bl"
}

# AprilTag detector
detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0
)

# ============================================================
# CAMERA
# ============================================================

def open_camera(idx):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


# ============================================================
# TAG → HOMOGRAPHY HELPERS
# ============================================================

def detect_tags(gray):
    return detector.detect(gray)


def get_inner_tag_corners(tags):
    """
    Returns the 4 physical inner-corner points in CAMERA coordinates.
    Each AprilTag gives us 4 corners: TL, TR, BR, BL.
    """
    pts = {}

    for t in tags:
        if t.tag_id in CORNER_MAP:
            role = CORNER_MAP[t.tag_id]

            # Map role to the correct tag corner
            if role == "tl":
                pts["tl"] = t.corners[0]
            elif role == "tr":
                pts["tr"] = t.corners[1]
            elif role == "br":
                pts["br"] = t.corners[2]
            elif role == "bl":
                pts["bl"] = t.corners[3]

    if len(pts) != 4:
        return None

    # Clockwise polygon
    return np.array(
        [pts["tl"], pts["tr"], pts["br"], pts["bl"]],
        dtype=np.float32
    )


def compute_homography(tags):
    img_quad = get_inner_tag_corners(tags)
    if img_quad is None:
        return None, None

    mm_quad = np.array([
        [INNER_X0, INNER_Y0],
        [INNER_X1, INNER_Y0],
        [INNER_X1, INNER_Y1],
        [INNER_X0, INNER_Y1]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(img_quad, mm_quad, cv2.RANSAC)
    return H, img_quad


def warp_inner_region(frame, H):
    """
    Warps ONLY the inner 300×400 mm region.
    """
    S = np.array([
        [PX_PER_MM, 0, -INNER_X0 * PX_PER_MM],
        [0, PX_PER_MM, -INNER_Y0 * PX_PER_MM],
        [0, 0, 1]
    ], dtype=np.float32)

    H_total = S @ H
    rectified = cv2.warpPerspective(frame, H_total, (OUT_W, OUT_H))
    return rectified, H_total


# ============================================================
# OUTLINE EXTRACTION
# ============================================================
def extract_outline(rectified):
    # --- Ignore edges near the border ---
    INNER_MARGIN_MM = 10
    margin_px = int(INNER_MARGIN_MM * PX_PER_MM)

    h, w = rectified.shape[:2]
    cropped = rectified[margin_px:h - margin_px, margin_px:w - margin_px]

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # --- Strong thresholding (less sensitive to white space) ---
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --- Remove noise ---
    kernel = np.ones((7,7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, thresh

    # --- Pick only large contours (avoid white-space noise) ---
    min_area = 0.03 * (cropped.shape[0] * cropped.shape[1])
    big_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not big_contours:
        return None, thresh

    cnt = max(big_contours, key=cv2.contourArea)

    # --- Simplify to smooth polygon ---
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)

    # --- Shift back to original rectified coordinate frame ---
    approx[:,0] += margin_px
    approx[:,1] += margin_px

    return approx, thresh



# ============================================================
# SVG EXPORT
# ============================================================

def export_svg(points_px, out_path):

    # 1) Convert rectified px → mm
    # Points are in rectified space (0 to OUT_W, 0 to OUT_H)
    pts_mm = points_px / PX_PER_MM

    # 2) Create SVG with correct physical dimensions
    width_mm  = INNER_WIDTH_MM   # 300mm
    height_mm = INNER_HEIGHT_MM  # 400mm

    dwg = svgwrite.Drawing(
        out_path,
        size=(f"{width_mm}mm", f"{height_mm}mm"),
        viewBox=f"0 0 {width_mm} {height_mm}"
    )

    # 3) Draw inner bounding box for reference
    dwg.add(dwg.rect(
        insert=(0, 0),
        size=(width_mm, height_mm),
        fill="none",
        stroke="lightgray",
        stroke_width=0.5
    ))

    # 4) Build polygon path
    # Clamp coordinates to canvas bounds to prevent overflow
    pts_mm = np.clip(pts_mm, [0, 0], [width_mm, height_mm])
    
    d = f"M {pts_mm[0,0]:.2f} {pts_mm[0,1]:.2f}"
    for x, y in pts_mm[1:]:
        d += f" L {x:.2f} {y:.2f}"
    d += " Z"

    dwg.add(dwg.path(
        d=d,
        stroke="red",
        fill="none",
        stroke_width=2
    ))

    dwg.save()
    print(f"[OK] SVG saved → {out_path}")



# ============================================================
# LIVE UI: SHADING OUTSIDE THE TAG BOUNDARY
# ============================================================

def shade_outside_boundary(display, boundary):
    overlay = display.copy()
    overlay[:] = (overlay * 0.3).astype(np.uint8)

    mask = np.zeros(display.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [boundary], 255)

    # Composite: interior stays bright, outside dimmed
    return np.where(mask[...,None] == 255, display, overlay)


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    Path("captures").mkdir(exist_ok=True)
    Path("debug").mkdir(exist_ok=True)

    cam_index = 0
    cap = open_camera(cam_index)
    if cap is None:
        print("No webcam found.")
        return

    print("SPACE=capture, TAB=switch, ESC=quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detect_tags(gray)

        display = frame.copy()

        # Draw tag centers
        for t in tags:
            cx, cy = map(int, t.center)
            cv2.circle(display, (cx, cy), 5, (0,255,0), -1)

        # If we have 4 corners, compute inner region
        H, boundary_tag_poly = compute_homography(tags)

        outline_pts_screen = None

        if H is not None:
            # Shade outside tag box
            display = shade_outside_boundary(display, boundary_tag_poly.astype(np.int32))
            cv2.polylines(display, [boundary_tag_poly.astype(np.int32)], True, (0,255,255), 3)

            # Rectify region
            rectified, H_total = warp_inner_region(frame, H)

            # Extract object outline
            outline, thresh = extract_outline(rectified)

            if outline is not None:
                # Project outline back to camera space
                H_inv = np.linalg.inv(H_total)
                pts = []
                for x, y in outline:
                    v = np.array([x, y, 1.0])
                    p = H_inv @ v
                    p /= p[2]
                    pts.append([int(p[0]), int(p[1])])
                outline_pts_screen = np.array(pts, dtype=np.int32)

                # Draw on live preview
                cv2.polylines(display, [outline_pts_screen], True, (0,255,255), 3)

        cv2.imshow("Live Trace", display)
        key = cv2.waitKey(1)

        # Quit
        if key == 27:
            break

        # Switch camera
        if key == 9:
            cam_index += 1
            cap.release()
            cap = open_camera(cam_index)
            continue

        # Capture → save SVG
        if key == 32 and outline_pts_screen is not None:
            ts = int(time.time())
            svg_path = f"captures/trace_{ts}.svg"

            # Re-rectify for clean SVG
            rectified, H_total = warp_inner_region(frame, H)
            outline, edges = extract_outline(rectified)

            out_px = outline
            export_svg(out_px, svg_path)

            cv2.imwrite(f"captures/capture_{ts}.png", frame)
            cv2.imwrite(f"debug/edges_{ts}.png", edges)
            print("Captured.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
