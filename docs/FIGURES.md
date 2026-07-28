# Regenerating the project-page figures

Both figures are generated from the released `CoCount-train` parquet shards. Point
`DATA` at your local copy (or a `datasets` cache) and run:

```bash
python render_hero.py       # -> cocount_teaser.{png,jpg}   full-bleed banner
python render_gallery.py    # -> cocount_gallery.png        annotated INTER/INTRA grid
```

Then copy the JPEGs into `assets/`. Both scripts are deterministic: example
selection is driven by sorted keys and a target object count, with no RNG, so
re-running reproduces the same figures.

`render_hero.py` crops each tile to the annotated dot cluster so tiles are filled
with objects rather than empty table. `render_gallery.py` picks balanced
mid-density scenes (~55 objects per class) because dense ones collapse into dot
soup and hide the fine-grained distinction the dataset is about.
