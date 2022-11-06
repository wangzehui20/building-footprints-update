"""
writed by wangzehui
"""

import cv2
import numpy as np
import shapely
import os
import os.path as osp
from shapely.geometry import Polygon
from tqdm import tqdm

def check_dir(dir):
    if not os.path.exists(dir): os.makedirs(dir)

def mask2cntrs(mask):
    contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # 出现area为空的情况
    if len(contours) == 0: return None
    contours_filter = []
    for j, contour in enumerate(contours):
        if len(contour)>=3:
            contours_filter.append(np.squeeze(contour))   # (count_num, 1, 2)
    return contours_filter

def cntr2poly(contour):
    return Polygon(contour).convex_hull

def get_cntrs(pred, mask_ref, iou_thred=0):
    pred_cntrs = mask2cntrs(pred)
    mask_ref_cntrs = mask2cntrs(mask_ref)
    cntrs = []
    if not pred_cntrs:
        return cntrs
    if not mask_ref_cntrs:
        return pred_cntrs

    for pred_cntr in pred_cntrs:
        poly = cntr2poly(pred_cntr)
        update_flag = False
        for mask_ref_cntr in mask_ref_cntrs:
            poly_ref = cntr2poly(mask_ref_cntr)
            # iou = cal_iou_sub(poly, poly_ref, poly_ref_flag=False)
            iou_ref = cal_iou_sub(poly, poly_ref)
            if iou_ref > iou_thred:
                # if iou > iou_ref:
                #     break
                cntrs.append(mask_ref_cntr)
                update_flag = True
                # break
        if not update_flag:
            # pred_cntr = cv2.approxPolyDP(pred_cntr)
            cntrs.append(pred_cntr)
    return cntrs

def cal_iou_sub(poly, poly_ref, poly_ref_flag=True):
    if not poly.intersects(poly_ref):
        return 0
    else:
        try:
            inter_area = poly.intersection(poly_ref).area
            union_area = poly_ref.area if poly_ref_flag else poly.area
            iou_sub = float(inter_area/union_area)
            return iou_sub
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            return 0

def update(pred_path, mask_ref_path, mask_update_path, im_size=256, iou_thred=0.5):
    pred = cv2.imread(pred_path, 0)
    mask_ref = cv2.imread(mask_ref_path, 0)
    cntrs = get_cntrs(pred, mask_ref, iou_thred)
    mask_update = np.zeros((im_size, im_size))
    for cntr in cntrs:
        cv2.fillPoly(mask_update, cntr[np.newaxis,:, :], 255)
    cv2.imwrite(mask_update_path, mask_update)

def update_dir(pred_dir, mask_ref_dir, mask_update_dir, iou_thred=0):
    """
    replace prediction to before mask when both are intersect most
    :param pred_dir: semantic segmentation prediction dir
    :param mask_ref_dir: historical mask dir
    :param mask_update_dir: update dir
    """
    pred_names = os.listdir(pred_dir)
    for name in tqdm(pred_names, total=len(pred_names)):
        pred_path = osp.join(pred_dir, name)
        # mask_ref_path = osp.join(mask_ref_dir, name.replace('2016_merge', '2012_merge'))   # replace name
        mask_ref_path = osp.join(mask_ref_dir, name.replace('2019_', '2018_'))   # replace name
        mask_update_path = osp.join(mask_update_dir, name)
        update(pred_path, mask_ref_path, mask_update_path, iou_thred=iou_thred)


if __name__ == '__main__':
    name = 'cyclegan'
    pred_dir = rf'/data/data/change_detection/models/{name}/unet/effb1_dicebce/pred'
    mask_ref_dir = r'/data/data/change_detection/merge/256_128/2012/mask'
    mask_update_dir = rf'/data/data/change_detection/models/{name}/unet/effb1_dicebce/pred_update/'

    update_dir(pred_dir, mask_ref_dir, mask_update_dir)
