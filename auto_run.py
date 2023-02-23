import os
from pathlib import Path
from termcolor import colored

print(colored('=== AUTO RUN SCRIPT BEGIN', 'red'))

gt_folder_path = Path('/data/tanlinfeng/IHD/HFlickr/real_images')
assert gt_folder_path.exists and gt_folder_path.is_dir
gt_paths = list(gt_folder_path.glob('*'))
gt_names = [x.stem for x in gt_paths]
gt_names.sort()
print(colored(f'Collect {len(gt_names)} files.', 'red'))

cnt = 0
for gt_name in gt_names:
    print(colored('=== MODEL START', 'red'))
    os.system(f'python3 train.py --data_name={gt_name} --copy_data=True')
    cnt += 1
    if cnt == 1000:
        break

print(colored('=== FINISH', 'red'))
