from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import glob
import os
import pylidc as pl
# import pydicom
# import dicom

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
        # mac
        # init_path = '/Volumes/masashi/workspace/0_KML/0_medical/0_data/LIDC-IDRI/'
        # wsl
        init_path = '/mnt/c/Users/masashi/workspace/1_KML/4_medical/LIDC-IDRI/'
        path = glob.glob(init_path + dir_name + '/*')
        return path, len(path)
    
    # ディレクトリが2つ以上かどうかを判断する
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

    # マスク画像を取り出す
    def get_mask(dicom_path):
        return 0


    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

d = Dataset()
d.ann_index = 0
scan_index = 0
for i, ann_index in enumerate(range(1012)):
    annotation = (d.get_texture(1))[ann_index]
    scan = d.get_scan(annotation)
    dir_name = d.get_abs_path(scan)
    contour_slice_list = d.get_contour_slice_list(d.get_contour(annotation))
    for contour_slice in contour_slice_list:
        target_dicom_num, target_dicom = d.get_dicom_path(dir_name, contour_slice)
        # print (target_dicom_num, target_dicom)
        print (target_dicom)
