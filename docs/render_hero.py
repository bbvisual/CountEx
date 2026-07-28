"""Render a full-bleed CoCount teaser mosaic, in the style of the PairTally banner.

Justified-rows layout: every row is a fixed height, images keep their aspect
ratio, and the last image in each row is centre-cropped so the row fills the
banner width exactly. Row heights vary so the mosaic doesn't read as a grid.
Raw images only -- no dots, no labels; the annotated grid is a separate figure.

Deterministic: candidate selection and ordering are driven by sorted keys only.
"""
import glob
import io
import os

import pyarrow.parquet as pq
from PIL import Image

DATA = "/data2/khanhnguyen/Rex-Omni/data/cocount/cocount_hf/data"
OUT = os.path.dirname(os.path.abspath(__file__))
W = 3000                      # banner width, px
ROWS = [0.30, 0.42, 0.28]     # row heights as a fraction of banner height
H = 1500
PER_ROW = [5, 4, 6]           # images per row; wider row gets fewer, bigger tiles
TARGET = 60                   # prefer scenes around this many objects per class

COLS = ["image", "pos_count", "neg_count", "category", "image_name"]
PTS = ["pos_points", "neg_points"]
MARGIN = 1.30   # cluster bbox expansion; keeps a little context around objects


def cluster_crop(im, pts):
    """Crop to the annotated cluster (plus margin) so tiles are filled with
    objects rather than empty table. Aspect ratio of the source is preserved
    loosely -- the row packer does the final horizontal trim."""
    if not pts:
        return im
    w, h = im.size
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    bw = max(48.0, (max(xs) - min(xs)) * MARGIN)
    bh = max(48.0, (max(ys) - min(ys)) * MARGIN)
    bw, bh = min(bw, float(w)), min(bh, float(h))
    left = min(max(0.0, cx - bw / 2), w - bw)
    top = min(max(0.0, cy - bh / 2), h - bh)
    return im.crop((int(left), int(top), int(left + bw), int(top + bh)))


def pick():
    """One candidate per class-pair, then spread across supercategories."""
    best = {}
    for f in sorted(glob.glob(f"{DATA}/*.parquet")):
        t = pq.read_table(f, columns=COLS + PTS).to_pydict()
        for i in range(len(t["pos_count"])):
            cat = t["category"][i]
            p, n = t["pos_count"][i], t["neg_count"][i]
            if not p or not n:
                continue
            cost = abs(p - TARGET) + abs(n - TARGET) + 2 * abs(p - n)
            cand = (cost, t["image_name"][i])
            if cat not in best or cand < best[cat][0]:
                pts = list(t["pos_points"][i] or []) + list(t["neg_points"][i] or [])
                best[cat] = (cand, (t["image"][i]["bytes"], pts))

    # interleave supercategories so neighbouring tiles look different
    by_super = {}
    for cat, (_, payload) in sorted(best.items()):
        by_super.setdefault(cat.split("_")[1], []).append((cat, payload))
    order, idx = [], 0
    supers = sorted(by_super)
    while len(order) < sum(PER_ROW):
        added = False
        for s in supers:
            if idx < len(by_super[s]):
                order.append(by_super[s][idx][1])
                added = True
                if len(order) == sum(PER_ROW):
                    break
        if not added:
            break
        idx += 1
    return order


def fill_row(items, row_w, row_h):
    """Scale each image to row_h, then trim total width to exactly row_w."""
    ims = []
    for b, pts in items:
        im = cluster_crop(Image.open(io.BytesIO(b)).convert("RGB"), pts)
        w, h = im.size
        nw = max(1, round(w * row_h / h))
        ims.append(im.resize((nw, row_h), Image.LANCZOS))

    total = sum(im.size[0] for im in ims)
    # rescale widths proportionally so they sum to row_w, cropping horizontally
    strip = Image.new("RGB", (row_w, row_h), (255, 255, 255))
    x = 0
    for k, im in enumerate(ims):
        share = im.size[0] / total
        tw = row_w - x if k == len(ims) - 1 else max(1, round(row_w * share))
        if im.size[0] >= tw:                       # centre-crop to target width
            left = (im.size[0] - tw) // 2
            tile = im.crop((left, 0, left + tw, row_h))
        else:                                      # rare: upscale to cover
            sc = tw / im.size[0]
            up = im.resize((tw, max(1, round(row_h * sc))), Image.LANCZOS)
            top = (up.size[1] - row_h) // 2
            tile = up.crop((0, top, tw, top + row_h))
        strip.paste(tile, (x, 0))
        x += tw
    return strip


def main():
    blobs = pick()  # list of (jpeg_bytes, points)
    print(f"selected {len(blobs)} images")
    banner = Image.new("RGB", (W, H), (255, 255, 255))
    y, used = 0, 0
    for r, frac in enumerate(ROWS):
        row_h = H - y if r == len(ROWS) - 1 else round(H * frac)
        chunk = blobs[used:used + PER_ROW[r]]
        used += PER_ROW[r]
        banner.paste(fill_row(chunk, W, row_h), (0, y))
        y += row_h

    banner.save(os.path.join(OUT, "cocount_teaser.png"))
    banner.save(os.path.join(OUT, "cocount_teaser.jpg"), "JPEG",
                quality=86, optimize=True, progressive=True)
    print("wrote cocount_teaser.{png,jpg}", banner.size)


if __name__ == "__main__":
    main()
