import cv2, numpy as np
from pathlib import Path

def crop_black(img, tol=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]

def tile_images(img_dir, out_path="mosaic_preview.png", cols=3):
    paths = sorted(Path(img_dir).glob("warp_*.png"))
    imgs = [crop_black(cv2.imread(str(p))) for p in paths]

    # normalize heights
    hmin = min(i.shape[0] for i in imgs)
    imgs = [cv2.resize(i, (int(i.shape[1]*hmin/i.shape[0]), hmin)) for i in imgs]

    # stack rows
    rows = []
    for i in range(0, len(imgs), cols):
        row_imgs = imgs[i:i+cols]
        # find max width of this row
        max_w = max(im.shape[1] for im in row_imgs)
        # pad all images in this row to same width
        padded = [cv2.copyMakeBorder(im, 0, 0, 0, max_w - im.shape[1], cv2.BORDER_CONSTANT, value=[0,0,0]) for im in row_imgs]
        rows.append(np.hstack(padded))

    # pad rows vertically to match height
    max_h = max(r.shape[0] for r in rows)
    padded_rows = [cv2.copyMakeBorder(r, 0, max_h - r.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0]) for r in rows]

    mosaic = np.vstack(padded_rows)
    cv2.imwrite(out_path, mosaic)
    print(f"✅ Saved mosaic → {out_path}")

if __name__ == "__main__":
    tile_images("ct_out", "mosaic_preview.png", cols=3)
