

import sys
sys.path.append('.')
sys.path.append('..')
from utils.image import _palette
from utils.image import save_mask
import numpy as np
from skimage.morphology.binary import binary_dilation
from PIL import Image
import cv2
import importlib
import os
from time import time


color_palette = np.array(_palette).reshape(-1, 3)


def overlay(image, mask, colors=[255, 0, 0], cscale=1, alpha=0.4):
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask

        foreground = image * alpha + np.ones(
            image.shape) * (1 - alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        countours = binary_dilation(binary_mask) ^ binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)


seq = "dressage"

image_root = os.path.join("./datasets/long_videos/JPEGImages/480p", seq)
gt_root = os.path.join("./datasets/long_videos/Annotations/480p", seq)
gt_root = os.path.join("./results/long_videos/long_videos_val_pe_bs8_10wstep_R50_DeAOTL_PRE_YTB_DAV_ckpt_80000_ema_mem_1_7_drop_layer_0_focus_mov_mean_0.8_ucb_reset_0_plus_8_mul_1/Annotations/480p/", seq)
input1_root = os.path.join(
    "./results/long_videos/long_videos_val_gap_per_30_R50_DeAOTL_PRE_YTB_DAV_ckpt_unknown/Annotations/480p/", seq)
# input2_root = os.path.join(
#     "./results/long_videos/long_videos_val_pe_bs8_10wstep_R50_DeAOTL_PRE_YTB_DAV_ckpt_80000_ema_mem_1_7_drop_layer_0_focus_mov_mean_0.8_ucb_reset_0_plus_8_mul_1/Annotations/480p/dressage", seq)

output_root = os.path.join("./video_output", seq)
if not os.path.exists(output_root):
    os.makedirs(output_root)
gt_files = os.listdir(gt_root)
gt_files.sort()

fourcc = cv2.VideoWriter_fourcc(*'XVID')

gt_files = gt_files[gt_files.index("10818.png"):]

for i, gt_file in enumerate(gt_files):
    print(f"{gt_file = }")
    gt = Image.open(
        os.path.join(gt_root, gt_file)
    )
    gt = np.array(gt)

    image = Image.open(
        os.path.join(image_root, gt_file.replace('.png', '.jpg'))
    )
    input1 = Image.open(
        os.path.join(input1_root, gt_file)
    )
    overlayed_image1 = overlay(
        np.array(image, dtype=np.uint8),
        np.array(input1, dtype=np.uint8), color_palette)
    Image.fromarray(overlayed_image1).save(os.path.join(
        output_root,
        f"img1.{gt_file}"))
    # input2 = Image.open(
    #     os.path.join(input2_root, gt_file)
    # )
    # overlayed_image2 = overlay(
    #     np.array(image, dtype=np.uint8),
    #     np.array(input2, dtype=np.uint8), color_palette)
    # Image.fromarray(overlayed_image2).save(os.path.join(
    #     output_root,
    #     f"img2.{gt_file}"))
    overlayed_gt = overlay(
        np.array(image, dtype=np.uint8),
        np.array(gt, dtype=np.uint8), color_palette)
    Image.fromarray(overlayed_gt).save(os.path.join(
        output_root,
        f"gt.{gt_file}"))
    # print(f"{overlayed_image2.shape = }")
    # concat_image = np.concatenate([
    #     overlayed_image1, overlayed_image2
    # ], axis=1)
    # output_image_path = os.path.join(
    #     output_root,
    #     gt_file)
    # Image.fromarray(concat_image).save(output_image_path)
    if i == 0:
        videoWriter_image1 = cv2.VideoWriter(
            os.path.join(
                output_root,
                "img1.video.avi"), fourcc, 5,
            (int(overlayed_image1.shape[1]), int(overlayed_image1.shape[0])))
        # videoWriter_image2 = cv2.VideoWriter(
        #     os.path.join(
        #         output_root,
        #         "img2.video.avi"), fourcc, 5,
        #     (int(overlayed_image2.shape[1]), int(overlayed_image2.shape[0])))
        videoWriter_gt = cv2.VideoWriter(
            os.path.join(
                output_root,
                "gt.video.avi"), fourcc, 5,
            (int(overlayed_gt.shape[1]), int(overlayed_gt.shape[0])))
    videoWriter_image1.write(overlayed_image1[..., [2, 1, 0]])
    # videoWriter_image2.write(overlayed_image2[..., [2, 1, 0]])
    videoWriter_gt.write(overlayed_gt[..., [2, 1, 0]])
    # if gt_file == 'frame00420.png':
    # if gt_file == 'frame00498.png':
    # if gt_file == 'frame00520.png':
    # if gt_file == 'frame00384.png':
    # if gt_file == 'frame00330.png':
    if gt_file == '11097.png':
        break

videoWriter_image1.release()
# videoWriter_image2.release()
videoWriter_gt.release()
