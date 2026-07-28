"""Render a publication-style CoCount showcase grid for the project page.

Picks one INTER and one INTRA example per supercategory, overlays positive dots
(green) and negative dots (red), and labels each tile with the two class names
and their counts. Deterministic: fixed candidate ordering, no RNG.
"""
import glob
import io
import os

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

DATA = "/data2/khanhnguyen/Rex-Omni/data/cocount/cocount_hf/data"
OUT = os.path.dirname(os.path.abspath(__file__))
SUPERS = ["FOO", "FUN", "HOU", "OFF", "OTR"]
TILE = 420          # tile side in px
PAD = 10
LABEL_H = 46
POS = (46, 204, 113)
NEG = (231, 76, 60)

COLS = ["image", "pos_caption", "neg_caption", "pos_count", "neg_count",
        "category", "image_name"]


TARGET = 55  # per-class count that reads clearly at tile resolution


def load_rows():
    """Return {(super, mode): row}, preferring balanced mid-density examples.

    Dense tiles (300+ objects) collapse into dot soup and hide the very thing
    the dataset is about: two visually near-identical classes in one scene. So
    score on closeness to TARGET per class plus pos/neg balance.
    """
    best = {}
    for f in sorted(glob.glob(f"{DATA}/*.parquet")):
        t = pq.read_table(f, columns=COLS + ["pos_points", "neg_points"]).to_pydict()
        for i in range(len(t["pos_count"])):
            cat = t["category"][i]
            mode, sc = cat.split("_")[0], cat.split("_")[1]
            if sc not in SUPERS:
                continue
            p, n = t["pos_count"][i], t["neg_count"][i]
            if not p or not n:
                continue
            # lower is better; ties broken by image_name for determinism
            cost = abs(p - TARGET) + abs(n - TARGET) + 2 * abs(p - n)
            cand = (cost, t["image_name"][i])
            key = (sc, mode)
            if key not in best or cand < best[key][0]:
                best[key] = (cand, {c: t[c][i] for c in COLS + ["pos_points", "neg_points"]})
    return {k: v[1] for k, v in best.items()}


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def square_crop(im, pts):
    """Square crop centred on the annotated cluster, so tiles zoom in on the
    objects instead of padding out empty table/floor."""
    w, h = im.size
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * 1.18
    side = max(64.0, min(side, float(min(w, h))))
    left = min(max(0.0, cx - side / 2), w - side)
    top = min(max(0.0, cy - side / 2), h - side)
    box = (int(left), int(top), int(left + side), int(top + side))
    return im.crop(box), box


def make_tile(row, badge=None):
    im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    pts_all = list(row["pos_points"] or []) + list(row["neg_points"] or [])
    if pts_all:
        im, box = square_crop(im, pts_all)
    else:
        box = (0, 0, im.size[0], im.size[1])
    cw, ch = im.size
    s = TILE / max(cw, ch)
    im = im.resize((max(1, int(cw * s)), max(1, int(ch * s))), Image.LANCZOS)
    canvas = Image.new("RGB", (TILE, TILE), (255, 255, 255))
    ox, oy = (TILE - im.size[0]) // 2, (TILE - im.size[1]) // 2
    canvas.paste(im, (ox, oy))

    d = ImageDraw.Draw(canvas)
    r = 5
    for pts, col in ((row["pos_points"], POS), (row["neg_points"], NEG)):
        for p in pts or []:
            # points are absolute pixel coords in the original image
            x, y = ox + (p[0] - box[0]) * s, oy + (p[1] - box[1]) * s
            d.ellipse([x - r, y - r, x + r, y + r], fill=col,
                      outline=(255, 255, 255), width=2)

    if badge:
        fb = font(15)
        bw = d.textlength(badge, font=fb) + 12
        d.rectangle([6, 6, 6 + bw, 30], fill=(20, 20, 20))
        d.text((12, 9), badge, fill=(255, 255, 255), font=fb)

    tile = Image.new("RGB", (TILE, TILE + LABEL_H), (255, 255, 255))
    tile.paste(canvas, (0, 0))
    d = ImageDraw.Draw(tile)
    f = font(15)
    d.text((4, TILE + 4), f"{row['pos_caption']}", fill=POS, font=f)
    d.text((4, TILE + 24), f"{row['neg_caption']}", fill=NEG, font=f)
    fc = font(14)
    pc, nc = str(row["pos_count"]), str(row["neg_count"])
    d.text((TILE - 4 - d.textlength(pc, font=fc), TILE + 4), pc, fill=POS, font=fc)
    d.text((TILE - 4 - d.textlength(nc, font=fc), TILE + 24), nc, fill=NEG, font=fc)
    return tile


def main():
    rows = load_rows()
    modes = ["INTER", "INTRA"]
    tw, th = TILE + PAD, TILE + LABEL_H + PAD
    hdr = 34
    W = PAD + len(SUPERS) * tw
    H = hdr + PAD + len(modes) * th
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    fh = font(22)
    for c, sc in enumerate(SUPERS):
        x = PAD + c * tw
        d.text((x, 6), sc, fill=(30, 30, 30), font=fh)
    for r, mode in enumerate(modes):
        for c, sc in enumerate(SUPERS):
            row = rows.get((sc, mode))
            if row is None:
                continue
            sheet.paste(make_tile(row, badge=mode),
                        (PAD + c * tw, hdr + PAD + r * th))
    for name in ("cocount_gallery.png",):
        sheet.save(os.path.join(OUT, name))
        print("wrote", os.path.join(OUT, name), sheet.size)


if __name__ == "__main__":
    main()
