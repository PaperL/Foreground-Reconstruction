import numpy as np
from PIL import Image
from pathlib import Path
from termcolor import colored
from tqdm import tqdm
import csv
import multiprocessing

def get_first_file(path, rule):
    paths = list(path.glob(rule))
    return min(paths)

dataset_path = Path('/data/tanlinfeng/IHD/HCOCO')
output_path = Path('/data/smb_shared/@Dataset/IHD_FR/HCOCO/rerun')

def process_line(line):
    parts = line.split()
    name = parts[0]
    composite_image_path = get_first_file(dataset_path/'composite_images', name+'.*')
    foreground_image_path = get_first_file(output_path, name+'_2.*')
    real_image_path = get_first_file(dataset_path/'real_images', name.split('_')[0]+'.*')
    mask_path = get_first_file(dataset_path/'masks', '_'.join(name.split('_')[:2])+'.*')

    composite_image = np.float32(np.array(Image.open(str(composite_image_path))))
    foreground_image = np.float32(np.array(Image.open(str(foreground_image_path))))
    real_image = np.float32(np.array(Image.open(str(real_image_path))))
    mask = np.float32(np.array(Image.open(str(mask_path)))) / 255
    if foreground_image.shape[0] != composite_image.shape[0] or foreground_image.shape[1] != composite_image.shape[1]:
        f.write(name + " -1 -1 -1\n")
        return
        
    mask = np.concatenate([mask[:,:,np.newaxis],mask[:,:,np.newaxis],mask[:,:,np.newaxis]],axis=-1)
    
    diff1 = mask * ((composite_image - foreground_image) ** 2)
    result1 = diff1.sum() / mask.sum()

    diff2 = mask * ((composite_image - real_image) ** 2)
    result2 = diff2.sum() / mask.sum()
    diff3 = mask * ((foreground_image - real_image) ** 2)
    result3 = diff3.sum() / mask.sum()
    
    return [name, result1, result2, result3]

with open('HCOCO_rerun.txt', 'r') as f:
    lines = f.readlines()
    
    with open('HCOCO_fMSE.csv', mode='w', newline='') as table:
        writer = csv.writer(table)
        pbar = tqdm(total=len(lines))
        
        with multiprocessing.Pool(processes=20) as pool:
            results = []
            for result in pool.imap_unordered(process_line, lines):
                if result is not None:
                    results.append(result)
                    pbar.update()

            for result in results:
                writer.writerow(result)