import pylidc as pl

# ann = pl.query(pl.Annotation).filter(pl.Annotation.texture==1)[0]
# ann = pl.query(pl.Scan).filter(pl.Scan.patient_id == 'LIDC-IDRI-0424')[0]
# scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == 'LIDC-IDRI-0002').first()
# print (ann)
# scan_ann = pl.query(pl.Scan).filter(pl.Scan.id == ann.first().scan_id)
# scan = pl.query(pl.Scan)[12]

# ann = pl.query(pl.Annotation)[0]
# scan_ann = pl.query(pl.Scan).filter(pl.Scan.id == ann.scan_id)
# scan = pl.query(pl.Scan).filter(pl.Scan.id == ann.scan_id)[0]
# print (scan)

# ann.visualize_in_scan()
# scan.visualize(annotation_groups=scan.cluster_annotations())
# ann_scan = ann.scan.to_volume().shape
# print (ann_scan)
# ann.scan.visualize(annotation_groups=scan.cluster_annotations())

ann = pl.query(pl.Annotation)\
                .filter(pl.Annotation.lobulation == 5).first()
ann.visualize_in_3d()
