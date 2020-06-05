from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import glob
import os
import pylidc as pl

def get_scan_data():
    for i in range(len(pl.query(pl.Scan).all())):
        scans = pl.query(pl.Scan)[i]
        return scans

# Annotationデータから欲しいデータのScanデータを取り出す
def get_scan_from_ann(ann):
    scan_ann = pl.query(pl.Scan).filter(pl.Scan.id == ann.scan_id)
    return scan_ann

# textureが1のもの
def get_ann_data():
    for i in range(len(pl.query(pl.Annotation).filter(pl.Annotation.texture == 1).all())):
        # textureが1のもののスキャンデータを取る
        anns = pl.query(pl.Annotation).filter(pl.Annotation.texture == 1)[i]
        print (anns.cluster_annotations())
        # print (get_scan_from_ann(anns).all())


# scanの中からアノテーションされたものを取り出す
def get_ann_from_scan():
    for i in range(len(pl.query(pl.Scan).all())):
        scan = pl.query(pl.Scan)[i]
        nodules = scan.cluster_annotations()
        print (nodules)
        return nodules

def get_path(dir_name):
    init_path = '/Volumes/masashi/workspace/0_KML/0_medical/0_data/LIDC-IDRI/'
    if len(glob.glob(init_path + dir_name + '/*')) >= 2:
        return 'a'
    else:
        return 'b'

def get_visual(i):
    ann = pl.query(pl.Annotation).filter(pl.Annotation.texture == 1)[i]
    scan = pl.query(pl.Scan).filter(pl.Scan.id == ann.scan_id)[i]
    scan.visualize(annotation_groups = scan.cluster_annotations())


ann_index = 0
scan_index = 0
for i, ann_index in enumerate(range(1012)):
    ann = get_texture(1)
    print ('ann', ann[ann_index])
    scan = get_scan(ann[ann_index])
    print ('scan', scan)
    patient_id = get_patient_id(scan[scan_index])
    print ('patient_id', patient_id)
    slices = get_slices(scan[scan_index])
    print ('slices', slices)
    path = get_path(patient_id)
    print (i+1, ':', path)
    print ('----------------------------------')
