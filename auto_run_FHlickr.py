import os
from pathlib import Path
from termcolor import colored
import sys

print(colored('=== AUTO RUN SCRIPT BEGIN', 'red'))

dataset_path = Path('/data/tanlinfeng/IHD/HFlickr')

names=[]
with open(dataset_path / str('HFlickr_test.txt'), "r") as f:
    names.extend([x.strip().split('.')[0] for x in f.readlines()])
with open(dataset_path / str('HFlickr_train.txt'), "r") as f:
    names.extend([x.strip().split('.')[0] for x in f.readlines()])
names.sort()
print(len(names))
print(colored(f'Collect {len(names)} files.', 'red'))
start_id = int(sys.argv[1])
end_id = int(sys.argv[2])
print(colored(f'From {start_id} to {end_id}.', 'red'))

for i in range(start_id, end_id):
    print(colored('=== MODEL START', 'red'))
    os.system(f'python3 train.py --data_name={names[i]} --copy_data=True')

print(colored(f'From {start_id} to {end_id}.', 'red'))
print(colored('=== FINISH', 'red'))
