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

    # 任意のtextureの値のあるアノテーションデータを全て取り出す
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

    # アノテーションされたスキャンデータのディレクトリを投げた時にpathを返す
    def get_path(self, dir_name):
        init_path = '/Volumes/masashi/workspace/0_KML/0_medical/0_data/LIDC-IDRI/'
        path = glob.glob(init_path + dir_name + '/*')
        return path, len(path)
    
    # ディレクトリが2つ以上の場合ファイル数が多い方を返す
    def are_there_many(self, path):
        if len(path) == 1:
            return False
        else:
            return True

    # ディレクトリが複数ある場合ファイル数の多い方のディレクトリを返す
    def get_many_file_dir(self, path):
        dict_target_path = {}
        if self.are_there_many(path):
            for tp in path:
                target_path = tp + '/**/*'
                len_target_path = len(glob.glob(target_path))
                buf_dict = {target_path: len_target_path}
                dict_target_path.update(buf_dict)
            max_file_dir = max(dict_target_path, key=dict_target_path.get)
            return glob.glob(max_file_dir.rstrip('/*') + '/**/')[0]

        else:
            target_path = path[0] + '/**/'
            return (glob.glob(target_path))[0]

    # ディレクトを与えた時にそのディレクトリの中のdicomファイルの数を返す
    def get_dicom_count(self, dir_name):
        target_dir_name = dir_name + '*.dcm'
        return len(glob.glob(target_dir_name))

    # 目的のスライスのdicomファイルを返す
    def get_dicom(self, slice_list, len_dicom_files):
        list_target_dicom = []
        for a_slice in slice_list:
            target_dicom = len_dicom_files - a_slice
            list_target_dicom.append(target_dicom)
        return list_target_dicom


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
    path, len_path = d.get_path(patient_id)
    print (i+1, ':', path, len_path)
    print (d.are_there_many(path))
    # print (d.get_many_file_dir(path))
    dir_name = d.get_many_file_dir(path)
    print (dir_name)
    dicom_count = d.get_dicom_count(dir_name)
    print (dicom_count)
    print (d.get_dicom(slices, dicom_count))
    print ('----------------------------------')
