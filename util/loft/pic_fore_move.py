from pathlib import Path
from shutil import copyfile
from tqdm.auto import tqdm
import time

import cv2
import numpy as np
from PIL import Image
import multiprocessing

def align_image_1(fore, mask):
    fore_h, fore_w, _ = fore.shape
    mask_h, mask_w = mask.shape
    fore_1 = fore[:mask_h, :mask_w]
    return fore_1

def align_image_2(fore, mask):
    fore_h, fore_w, _ = fore.shape
    mask_h, mask_w = mask.shape
    y_offset = max(0, fore_h - mask_h)
    fore_2 = fore[y_offset:fore_h, :mask_w]
    return fore_2

def calc(comp_path, mask_path, fore_paths):
    # tqdm.write('\n'.join((str(comp_path), str(mask_path), str(len(fore_paths)), '')))
    tqdm.write(str(comp_path))
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    comp = cv2.imread(str(comp_path))
    comp_masked = cv2.bitwise_and(comp, comp, mask=mask)
    min_sum = 1e10
    true_fore_path = None
    true_fore = None
    for fore_path in fore_paths:
        fore = cv2.imread(str(fore_path))
        # print(fore.shape, mask.shape)
        fores = []
        
        border_size = 100
        fore_with_border = cv2.copyMakeBorder(fore, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
        result = cv2.matchTemplate(fore_with_border, comp, cv2.TM_CCOEFF_NORMED, mask=mask)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        h, w = comp_masked.shape[:2]
        aligned_fore = fore_with_border[max_loc[1]:max_loc[1]+h, max_loc[0]:max_loc[0]+w]
        fores.append(aligned_fore)

        if fore.shape[:2] != mask.shape:
            fores.append(align_image_1(fore, mask))
            fores.append(align_image_2(fore, mask))
        else:
            fores.append(fore)
        for fi in fores:
            fore_masked = cv2.bitwise_and(fi, fi, mask=mask)
            diff = cv2.absdiff(comp_masked, fore_masked)
            diff_sum = np.sum(diff)
            # print(str(comp_path.name), str(fore_path.name), diff_sum, sep='\t')
            if diff_sum < min_sum:
                min_sum = diff_sum
                true_fore_path = fore_path
                true_fore = fi
    # tqdm.write(true_fore_path.stem)
    output_fore_path = output_folder / (comp_path.name)
    cv2.imwrite(str(output_fore_path), true_fore)

    with open('fore10.txt', 'a') as f:
        composite_image = np.float32(np.array(Image.open(str(comp_path))))
        foreground_image = np.float32(np.array(Image.open(str(output_fore_path))))
        mask_image = np.float32(np.array(Image.open(str(mask_path)))) / 255
        if foreground_image.shape[0] != composite_image.shape[0] or foreground_image.shape[1] != composite_image.shape[1]:
            f.write(comp_path.name + " -1 -1 -1\n")
            return
            
        # mask = np.concatenate([mask[:,:,np.newaxis],mask[:,:,np.newaxis],mask[:,:,np.newaxis]],axis=-1)
        diff1 = mask_image * ((composite_image - foreground_image) ** 2)
        result1 = diff1.sum() / mask_image.sum()
        output_info = f'{comp_path.name} {true_fore_path.name} {result1}\n'
        f.write(output_info)
        tqdm.write(output_info)
    


def process_data(composite_images_folder, masks_folder, meta_folder):
    comp_paths = list(composite_images_folder.glob("*.jpg"))
    comp_paths.sort()
    pbar = tqdm(total=len(comp_paths), position=0, leave=True)
    with multiprocessing.Pool(processes=10) as pool:
        for _ in pool.imap_unordered(process_comp_path, comp_paths):
            pbar.update(1)

def process_comp_path(comp_path):
    # if not 'd90000009-20_1_6' in comp_path.name:
    #     return
    mask_path = masks_folder / (comp_path.stem.rsplit("_", 1)[0] + ".png")
    subfolder_name = str(int(comp_path.stem.split("-")[0][1:])).zfill(8)
    subfolder_path = meta_folder / subfolder_name
    fore_paths = list(subfolder_path.glob("*.jpg"))
    calc(comp_path, mask_path, fore_paths)

# Set the folder paths
dataset_folder = Path("/data/tanlinfeng/IHD/Hday2night")
composite_images_folder = dataset_folder / "composite_images"
masks_folder = dataset_folder / "masks"
meta_folder = Path("/data/smb_shared/@Dataset/Hday2night_meta/imageAlignedLD")
output_folder = Path('/data/smb_shared/@Dataset/IHD_FR/Hday2night/foreground_images_fix')

# Run the process_data function
process_data(composite_images_folder, masks_folder, meta_folder)