import pylidc as pl

# ann = pl.query(pl.Annotation).filter(pl.Annotation.texture==1)
# scan_ann = pl.query(pl.Scan).filter(pl.Scan.id == ann.first().scan_id)
# scan = pl.query(pl.Scan)[12]

ann = pl.query(pl.Annotation).first()
scan_ann = pl.query(pl.Scan).filter(pl.Scan.id == ann.scan_id)
scan = pl.query(pl.Scan).first()

# ann.visualize_in_scan()
scan.visualize(annotation_groups=scan.cluster_annotations())
