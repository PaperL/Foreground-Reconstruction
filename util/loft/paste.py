from pathlib import Path
import cv2
from tqdm.auto import tqdm
import multiprocessing

src_folder = Path('/data/tanlinfeng/IHD/Hday2night')
dst_folder = Path('/data/smb_shared/@Dataset/IHD_FR/Hday2night')

composite_folder = src_folder / 'composite_images'
mask_folder = src_folder / 'masks'

foreground_folder = dst_folder / 'foreground_images'
composite_foreground_folder = dst_folder/'composite_foreground_images'
composite_foreground_folder.mkdir(exist_ok=True)

def process_comp_path(composite_path):
    mask_path = mask_folder / (composite_path.stem.rsplit('_', 1)[0] + '.png')
    foreground_path = foreground_folder / composite_path.name
    composite_foreground_path = composite_foreground_folder / composite_path.name

    composite_img = cv2.imread(str(composite_path))
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    foreground_img = cv2.imread(str(foreground_path))

    foreground_img[mask_img != 0] = composite_img[mask_img != 0]

    cv2.imwrite(str(composite_foreground_path), foreground_img)

comp_paths = list(composite_folder.glob("*.jpg"))
comp_paths.sort()
pbar = tqdm(total=len(comp_paths), position=0, leave=True)
with multiprocessing.Pool(processes=10) as pool:
    for _ in pool.imap_unordered(process_comp_path, comp_paths):
        pbar.update(1)