from pathlib import Path
import shutil

root = Path('.')
old_fg = root / 'foreground_images_old'
rerun = root / 'rerun'
new_fg = root / 'foreground_images'

new_fg.mkdir(exist_ok=True)

for file in old_fg.glob('*'):
    shutil.copy(file, new_fg / file.name)

for file in rerun.glob('*'):
    shutil.copy(file, new_fg / file.name)

for file in new_fg.glob('*_2.*'):
    parts = file.name.rsplit('_2', 1)
    new_name = ''.join(parts)
    file.rename(new_fg / new_name)