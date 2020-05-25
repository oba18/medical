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

# textureが1のもの
def get_ann_data():
    for i in range(len(pl.query(pl.Annotation).filter(pl.Annotation.texture == 1).all())):
        anns = pl.query(pl.Annotation).filter(pl.Annotation.texture == 1)[i]
        # return anns
        print(anns)
        print(anns.scan)

# scanの中からアノテーションされたものを取り出す
def get_ann_from_scan():
    for i in range(len(pl.query(pl.Scan).all())):
        scan = pl.query(pl.Scan)[i]
        nodules = scan.cluster_annotations()
        print (nodules)
        return nodules

get_ann_data()
