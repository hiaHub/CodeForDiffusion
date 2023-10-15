import os
import torch
from skimage import io, img_as_float
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from PIL import Image

def compute_metrics(src_path, dst_path):
    src_images = [io.imread(os.path.join(src_path, img)) for img in os.listdir(src_path)]
    dst_images = [io.imread(os.path.join(dst_path, img)) for img in os.listdir(dst_path)]

    src_images = [img_as_float(img) for img in src_images]
    dst_images = [img_as_float(img) for img in dst_images]

    ssim_scores = [ssim(src_img, dst_img) for src_img, dst_img in zip(src_images, dst_images)]
    psnr_scores = [psnr(src_img, dst_img) for src_img, dst_img in zip(src_images, dst_images)]

    return ssim_scores, psnr_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, help='Ground truth images directory')
    parser.add_argument('-d', '--dst', type=str, help='Generate images directory')
   
    args = parser.parse_args()

    ssim_scores, psnr_scores = compute_metrics(args.src, args.dst)

    print('SSIM scores: {}'.format(ssim_scores))
    print('PSNR scores: {}'.format(psnr_scores))
