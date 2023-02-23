from ccHarmony import polynomial_matching as pm
import numpy as np
from pathlib import Path
from IPython.utils import io
from termcolor import colored
from tqdm import tqdm
from multiprocessing import Pool

# Input
poly_folder_path = Path('testspace/polynomial_matching_coefficients')
assert poly_folder_path.is_dir and poly_folder_path.exists
# Output
lut_folder_path = Path('testspace/lut')
lut_folder_path.mkdir(parents=True, exist_ok=False)

poly_file_names = [x.name for x in list(poly_folder_path.glob('*.npy'))]
poly_file_names.sort()
assert len(poly_file_names) % 2 == 0

poly_names = []
for i in range(0, len(poly_file_names), 2):
    assert poly_file_names[i][:-4] == poly_file_names[i+1][:-12]
    poly_names.append(poly_file_names[i][:-4])

print('Collect ' + colored(str(len(poly_names)), 'yellow') + ' polynomial coefficient files.')

def poly_to_lut(io_path):
    (poly_path, lut_path) = io_path
    m = pm.polyCoeff()
    m.read_npy(str(poly_path))
    dim = 33
    step = 1.0 / (dim - 1)
    with open(str(lut_path), "w") as file:
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    image = np.zeros((1, 1, 3), np.uint8)
                    image[0][0][0] = step * k * 255
                    image[0][0][1] = step * j * 255
                    image[0][0][2] = step * i * 255
                    # output = image
                    with io.capture_output() as _: # suppress print output in m.transform()
                        output = m.transform(image)
                    # file.write('%d %d %d\t' % (output[0][0][0], output[0][0][1], output[0][0][2]))
                    file.write('%.6f %.6f %.6f\n' % (output[0][0][0]/255., output[0][0][1]/255., output[0][0][2]/255.));

input(colored('Press any key to start...', 'green'))

io_paths = []
data_names = []
for i, poly_name in enumerate(poly_names):
    data_names.append('%04d' % i)
    output_name = '%04d_0.txt' % i
    poly_path = poly_folder_path / (poly_name+'.npy')
    lut_path = lut_folder_path / output_name
    io_paths.append((poly_path, lut_path))
    r_output_name = '%04d_1.txt' % i
    r_poly_path = poly_folder_path / (poly_name+'_reverse.npy')
    r_lut_path = lut_folder_path / r_output_name
    io_paths.append((r_poly_path, r_lut_path))

with open(str(lut_folder_path / 'list.txt'), 'w') as f:
    for data_name in data_names:
        f.write(data_name+'\n')

# with Pool() as pool:
#     r = list(tqdm(pool.imap(poly_to_lut, io_paths), total = len(poly_names)))