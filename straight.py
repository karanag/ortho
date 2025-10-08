#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
import argparse

def create_rounded_rect_mask(w, h, radius):
    """Return a proper rounded rectangle mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    # full white rounded rectangle
    cv2.rectangle(mask, (radius, 0), (w - radius, h), 255, -1)
    cv2.rectangle(mask, (0, radius), (w, h - radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (w - radius - 1, radius), radius, 255, -1)
    cv2.circle(mask, (radius, h - radius - 1), radius, 255, -1)
    cv2.circle(mask, (w - radius - 1, h - radius - 1), radius, 255, -1)
    return mask

def straighten_rug(
    img_path,
    out_path,
    grey_pad=80,          # width of grey border
    corner_radius=60,     # roundness of grey card corners
    canvas_expand=200,    # outer white margin
    shadow_opacity=0.25,  # 0–1, controls shadow darkness
    shadow_offset=(25, 25),
    shadow_blur=120,      # softness of shadow
):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    # Step 1 — Edge detection and rug contour
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 40, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found.")
    c = max(contours, key=cv2.contourArea)

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) != 4:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        approx = np.int0(box)
    pts = approx.reshape(4, 2)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(diff)], pts[np.argmax(diff)]

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH), borderValue=(255, 255, 255))

    # Step 2 — Create grey card layer
    h, w = warped.shape[:2]
    card_h = h + grey_pad * 2
    card_w = w + grey_pad * 2
    grey_card = np.ones((card_h, card_w, 3), np.uint8) * 245
    grey_card[grey_pad:grey_pad + h, grey_pad:grey_pad + w] = warped

    # Step 3 — Rounded corners
    mask = create_rounded_rect_mask(card_w, card_h, corner_radius)
    rounded = np.ones_like(grey_card, np.uint8) * 255
    for i in range(3):
        rounded[:, :, i] = np.where(mask == 255, grey_card[:, :, i], 255)

    # Step 4 — Add soft drop shadow
    if shadow_opacity > 0:
        shadow = np.zeros_like(rounded, np.uint8)
        cv2.rectangle(
            shadow,
            (shadow_offset[0], shadow_offset[1]),
            (card_w, card_h),
            (0, 0, 0),
            -1,
        )
        shadow = cv2.GaussianBlur(shadow, (shadow_blur | 1, shadow_blur | 1), 0)
        rounded = cv2.addWeighted(shadow, shadow_opacity, rounded, 1.0, 0)

    # Step 5 — Compose on white canvas
    final_h = card_h + canvas_expand * 2
    final_w = card_w + canvas_expand * 2
    white_bg = np.ones((final_h, final_w, 3), np.uint8) * 255
    y_off = (final_h - card_h) // 2
    x_off = (final_w - card_w) // 2
    white_bg[y_off:y_off + card_h, x_off:x_off + card_w] = rounded

    cv2.imwrite(out_path, white_bg, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"✅ Saved → {out_path}")

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Straighten rug with rounded grey border + optional shadow.")
    parser.add_argument("--inp", required=True, help="Input rug image")
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--grey_pad", type=int, default=80)
    parser.add_argument("--corner_radius", type=int, default=60)
    parser.add_argument("--canvas_expand", type=int, default=200)
    parser.add_argument("--shadow_opacity", type=float, default=0.25)
    parser.add_argument("--shadow_offset", type=int, nargs=2, default=[25, 25])
    parser.add_argument("--shadow_blur", type=int, default=120)
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    straighten_rug(
        args.inp,
        args.out,
        grey_pad=args.grey_pad,
        corner_radius=args.corner_radius,
        canvas_expand=args.canvas_expand,
        shadow_opacity=args.shadow_opacity,
        shadow_offset=tuple(args.shadow_offset),
        shadow_blur=args.shadow_blur,
    )
