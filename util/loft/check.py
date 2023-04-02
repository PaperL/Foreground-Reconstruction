from pathlib import Path

def compare_folders(folder1: str, folder2: str) -> bool:
    p1 = Path(folder1)
    p2 = Path(folder2)
    
    if not p1.is_dir() or not p2.is_dir():
        raise ValueError("Both inputs must be directories")
    
    files1 = set(p.name for p in p1.glob('*') if p.is_file())
    files2 = set(p.name for p in p2.glob('*') if p.is_file())
    for file2 in files2:
        if not file2 in files1:
            print(file2)
    return (files1 == files2), len(files1), len(files2)
    # for file1 in files1:
    #     if not file1 in files2:
    #         (p1 / file1).unlink()
    # files1 = set(p.name for p in p1.glob('*') if p.is_file())
    # files2 = set(p.name for p in p2.glob('*') if p.is_file())
    # return len(files1) == len(files2)

print(compare_folders(
    '/data/smb_shared/@Dataset/IHD_FR/HFlickr/foreground_images',
    '/data/tanlinfeng/IHD/HFlickr/composite_images',
    ))

print(compare_folders(
    '/data/smb_shared/@Dataset/IHD_FR/HFlickr/composite_foreground_images',
    '/data/tanlinfeng/IHD/HFlickr/composite_images',
    ))

print(compare_folders(
    '/data/smb_shared/@Dataset/IHD_FR/HCOCO/foreground_images',
    '/data/tanlinfeng/IHD/HCOCO/composite_images',
    ))

print(compare_folders(
    '/data/smb_shared/@Dataset/IHD_FR/HCOCO/composite_foreground_images',
    '/data/tanlinfeng/IHD/HCOCO/composite_images',
    ))

print(compare_folders(
    '/data/smb_shared/@Dataset/IHD_FR/Hday2night/foreground_images',
    '/data/tanlinfeng/IHD/Hday2night/composite_images',
    ))

print(compare_folders(
    '/data/smb_shared/@Dataset/IHD_FR/Hday2night/composite_foreground_images',
    '/data/tanlinfeng/IHD/Hday2night/composite_images',
    ))