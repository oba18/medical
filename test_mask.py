import pylidc as pl
import numpy as np

# test
ann = pl.query(pl.Annotation).filter(pl.Annotation.texture == 1)[1]
mask = ann.boolean_mask()
vol = ann.scan.to_volume()

# print (mask[363][343])
# print (mask.shape)
print (vol[363][343][0])
print (vol.shape)
