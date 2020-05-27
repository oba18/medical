from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import glob
import os
import pylidc as pl

class Dataset(BaseDataset):
    def __init__(self,):
        
        self.data = [
                {
                    # img: 普通の画像,
                    # mask: マスク画像
                    }
                ]

    # 任意のtextureの値のアノテーションデータを全て取り出す
    def get_texture(self, texture_val):
        return pl.query(pl.Annotation).filter(pl.Annotation.texture <= texture_val).all()

    def get_scan(self, annotation):
        return (pl.query(pl.Scan).filter(pl.Scan.id == annotation.scan_id)).all()

    # これの返り値が参照するディレクトリの名前になる
    def get_patient_id(self, scan):
        return scan.patient_id

    # アノテーションされたスキャンデータからアノテーションされているスライス面のスライス番号を返す
    def get_slices(self, scan):
        slices = [np.array([a.centroid for a in gourp]).mean(0) for gourp in scan.cluster_annotations()]
        slice_list = []
        for a in range(len(slices)):
            a_slice = int(slices[a][2])
            slice_list.append(a_slice)
        return slice_list

    def get_path(self, dir_name):
        init_path = '/Volumes/masashi/workspace/0_KML/0_medical/0_data/LIDC-IDRI/'
        if len(glob.glob(init_path + dir_name + '/*')) >= 2:
            return 'a'
        else:
            return 'b'

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

d = Dataset()
d.ann_index = 0
scan_index = 0
for i, ann_index in enumerate(range(1012)):
    ann = d.get_texture(1)
    print ('ann', ann[ann_index])
    scan = d.get_scan(ann[ann_index])
    print ('scan', scan)
    patient_id = d.get_patient_id(scan[scan_index])
    print ('patient_id', patient_id)
    slices = d.get_slices(scan[scan_index])
    print ('slices', slices)
    path = d.get_path(patient_id)
    print (i+1, ':', path)
    print ('----------------------------------')
