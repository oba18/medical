import pylidc as pl
import numpy as np

scans = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 1, pl.Scan.pixel_spacing <= 0.6)
print(scans.count())

pid = 'LIDC-IDRI-0001'
scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()

print(len(scan.annotations))

nods = scan.cluster_annotations()

print("%s has %d nodules." % (scan, len(nods)))

for i,nod in enumerate(nods):
    print("Nodule %d has %d annotations." % (i+1, len(nods[i])))

vol = scan.to_volume()
print(vol.shape)

print("%.2f, %.2f" % (vol.mean(), vol.std()))
