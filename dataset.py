from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import glob
import os
import pylidc as pl
import pydicom

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

    def get_abs_path(self, scan):
        return scan[0].get_path_to_dicom_files() + '/'
    
    # アノテーションされたデータのアノテーション部分を取り出す
    def get_contour(self, annotation):
        return pl.query(pl.Contour).filter(pl.Contour.annotation_id == annotation.id)

    # アノテーションされたデータのスライス面のリストを返す
    def get_contour_slice_list(self, contour):
        list_contour_slice = []
        for index in range(len(contour.all())):
            contour_slice = contour[index].image_k_position
            list_contour_slice.append(contour_slice)
        return list_contour_slice

    # 目標のdicomファイルを取り出す
    def get_dicom_path(self, dir_name, contour_slice):
        all_dicom_count = len(glob.glob(dir_name + '*.dcm'))
        target_dicom_num = all_dicom_count - contour_slice # スライス面とdcmが反対から参照のためファイル総数からスライス面を引く
        target_dicom = glob.glob(dir_name + '*{}*.dcm'.format(str(target_dicom_num)))[0]
        return target_dicom_num, target_dicom

    # dicomのCT値を取り出す
    def get_ct_vol(self, dicom_path):
        ds = pydicom.dcmread(dicom_path)
        return ds.pixel_array

    # マスク画像を取り出す
    def get_mask(self, contour, index):
        return contour[index].to_matrix()

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

d = Dataset()
d.ann_index = 0
scan_index = 0
for i, ann_index in enumerate(range(1012)):
    annotation = (d.get_texture(1))[ann_index]
    # print (annotation)
    scan = d.get_scan(annotation)
    # print (scan)
    dir_name = d.get_abs_path(scan)
    # print (dir_name)
    contour_slice_list = d.get_contour_slice_list(d.get_contour(annotation))
    contour = d.get_contour(annotation).all()
    print (contour)
    # print (contour_slice_list)
    for contour_slice in contour_slice_list:
        print (contour_slice, type(contour_slice))
        target_dicom_num, target_dicom = d.get_dicom_path(dir_name, contour_slice)
        print (target_dicom_num, target_dicom)
        # print (d.get_ct_vol(target_dicom))
        for index in range(len(contour)):
            print (d.get_mask(contour, index))

