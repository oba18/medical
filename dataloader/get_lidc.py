import numpy as np
import glob
import os
import pylidc as pl
import pydicom
import cv2
from tqdm import tqdm 

class Lidc():
    def __init__(self, is_develop):
        self.is_develop = is_develop
        # self.excute()
        self.img_list, self.mask_list = self.excute()

    def get_texture(self, texture_val):
        return pl.query(pl.Annotation).filter(pl.Annotation.texture <= texture_val).all()

    # アノテーション情報を入れると，そのアノテーションに対するスキャンデータを取り出す Annotation -> Scan
    def get_scan(self, annotation):
        return (pl.query(pl.Scan).filter(pl.Scan.id == annotation.scan_id)).all()

    # アノテーションされたデータのアノテーション部分を取り出す Annotation -> Contour
    def get_contour(self, annotation):
        return pl.query(pl.Contour).filter(pl.Contour.annotation_id == annotation.id).all()

    # スキャンデータに対するdicomのパスを取り出す
    def get_abs_path(self, scan):
        return scan[0].get_path_to_dicom_files() + '/'

    # HUを返す
    def get_pixels_hu(self, scan):
        a_slice = pydicom.dcmread(scan)

        # Convert to Hounsfield units (HU)
        intercept = a_slice.RescaleIntercept
        slope = a_slice.RescaleSlope
        a_slice = a_slice.pixel_array.astype(np.int16)

        if slope != 1:
            a_slice = slope * a_slice.astype(np.float64)
            a_slice = a_slice.astype(np.int16)
        a_slice += np.int16(intercept)

        return np.array(a_slice, dtype=np.int16)

    # アノテーションされたデータのスライス面のリストを返す
    def get_contour_slice_list(self, contour):
        contour_slice_list = []
        contour_matrix_list = []
        for index in range(len(contour)):
            contour_slice = contour[index].image_k_position
            contour_slice_list.append(contour_slice)
            contour_matrix = contour[index].to_matrix(False)
            contour_matrix_list.append(contour_matrix)
        return contour_slice_list, contour_matrix_list

    # 目標のdicomファイルを取り出す
    def get_dicom_path(self, dir_name, contour_slice):
        all_dicom_count = len(glob.glob(dir_name + '*.dcm'))
        # スライス面とdcmが反対から参照のためファイル総数からスライス面を引く
        target_dicom_num = all_dicom_count - contour_slice 
        target_dicom = glob.glob(dir_name + '*{}*.dcm'.format(str(target_dicom_num)))[0]
        return target_dicom_num, target_dicom

    # dicomのCT値を取り出す
    def get_ct_vol(self, dicom_path):
        ds = pydicom.dcmread(dicom_path)
        return ds.pixel_array

    # マスク画像の生成
    def to_mask(self, ct_vol, contour_matrix):
        mask = np.zeros(ct_vol.shape, dtype=np.int32)
        cv2.fillConvexPoly(mask, points=contour_matrix[:,::-1], color=(1))
        return mask

    def excute(self):
        ann_index = 0
        img_list = []
        mask_list = []
        annotation_size = len(self.get_texture(1)) if self.is_develop == False else 10
        for ann_index in tqdm(range(annotation_size)):
            annotation = (self.get_texture(1))[ann_index]
            scan = self.get_scan(annotation)
            dir_name = self.get_abs_path(scan)
            contour_slice_list, contour_matrix_list = self.get_contour_slice_list(self.get_contour(annotation))
            contour = self.get_contour(annotation)
            for contour_slice, contour_matrix in zip(contour_slice_list, contour_matrix_list):
                target_dicom_num, target_dicom = self.get_dicom_path(dir_name, contour_slice)
                # ct_vol = self.get_ct_vol(target_dicom)
                hu = self.get_pixels_hu(target_dicom)
                # mask = self.to_mask(ct_vol, contour_matrix)
                mask = self.to_mask(hu, contour_matrix)
                # img_list.append(ct_vol) 
                img_list.append(hu) 
                mask_list.append(mask)
                # target_dicom.append(dir_name)
        return np.array(img_list), np.array(mask_list)
