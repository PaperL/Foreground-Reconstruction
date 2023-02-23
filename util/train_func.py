import yaml
import torch
import cv2
from torchvision.models import resnet18
from torchvision import transforms
import argparse
from pathlib import Path
import shutil
import numpy as np
from tqdm import tqdm
from termcolor import colored


from util.net import LUTFR

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, required=False)
    parser.add_argument('--checkpoint', type=str, required=False)
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--copy_data', type=bool, default=False, required=False)
    return parser.parse_args()


def init_exp():
    args = get_argument()
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
    checkpoint_path = Path(config['CHECKPOINT_PATH'])
    # checkpoint_path.mkdir(parents=False, exist_ok=False)
    device = torch.device(f'cuda:{args.gpu}')
    return args, config, checkpoint_path, device


def get_first_file(path, rule):
    paths = list(path.glob(rule))
    return min(paths)

def init_data(config, args):
    trans_tensor = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset_folder_path = Path(config['DATASET_PATH'])
    assert dataset_folder_path.is_dir and dataset_folder_path.exists

    name = args.data_name
    composite_path = get_first_file(dataset_folder_path / 'composite_images', name+'_*')
    mask_path = get_first_file(dataset_folder_path / 'masks', name+'_*')
    gt_path = get_first_file(dataset_folder_path / 'real_images', name+'.*')
    print(colored('Compoiste image, mask, ground truth:', 'yellow'))
    print(str(composite_path))
    print(str(mask_path))
    print(str(gt_path))
    if args.copy_data:
        result_path = Path(config['OUTPUT_PATH'])
        shutil.copyfile(str(composite_path), str(result_path/(name+'_2.jpg')))
        shutil.copyfile(str(mask_path), str(result_path/(name+'_1.png')))
        shutil.copyfile(str(gt_path), str(result_path/(name+'_0.jpg')))

    composite = cv2.imread(str(composite_path))
    composite = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
    composite = trans_tensor(composite)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = trans_tensor(mask)
    gt = cv2.imread(str(gt_path))
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    gt = trans_tensor(gt)

    lut_folder_path = Path(config['LUT_PATH'])
    assert lut_folder_path.is_dir and lut_folder_path.exists
    with open(str(lut_folder_path/'list.txt'), 'r') as f:
        lut_names = f.readlines()
    
    lutN = len(lut_names)
    lut = [[], []]
    dim = 33
    print(colored('Loading LUTs...', 'yellow'))
    for lut_name in tqdm(lut_names, leave=False):
        for t in range(2):
            with open(str(lut_folder_path/(lut_name[:-1]+'_%d.txt'%t)), 'r') as f:
                buffer = np.zeros((3,dim,dim,dim), dtype=np.float32)
                lines = f.readlines()
                for i in range(0,dim):
                    for j in range(0,dim):
                        for k in range(0,dim):
                            n = i * dim*dim + j * dim + k
                            x = lines[n].split()
                            buffer[0,i,j,k] = float(x[0])
                            buffer[1,i,j,k] = float(x[1])
                            buffer[2,i,j,k] = float(x[2])
                lut[t].append(buffer)
    lut = np.array(lut)
    print(lut.shape)
    return {'composite':composite, 'mask':mask, 'gt':gt, 'lut':lut, 'lutN': lutN}


# def calculate_fMSE(gt, pred, mask):
#     loss_fn = torch.nn.MSELoss()
#     fmse_loss = torch.nn.MSELoss(pred * mask, gt * mask)
#     return fmse_loss


def init_net(device, lut, lutN):
    model = LUTFR(device, lut, lutN)
    model = model.to(device)

    # print(colored('Parameters: ', 'red'), list(model.parameters()))
    # print(colored('Model: ', 'red'), model)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    return model, optimizer


def load_checkpoint(model, optimizer, checkpoint_path, checkpoint_name):
    checkpoint_file = checkpoint_path / (checkpoint_name + '.pt')
    assert checkpoint_file.exists()
    checkpoint_data = torch.load(str(checkpoint_file))
    model.load_state_dict(checkpoint_data['model_state_dict'])
    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    return checkpoint_data['epoch_id'] + 1

def save_checkpoint(checkpoint_path, model, optimizer, epoch_id, epoch_loss, data_name):
    statics_file = checkpoint_path / (data_name+'_statics.yaml')
    checkpoint_name = str(epoch_id).zfill(4)
    with open(str(statics_file), 'a+') as file:
        statics = yaml.full_load(file)
        statics = dict() if statics is None else statics
        statics[checkpoint_name] = {
            'epoch_id': epoch_id,
            'loss': epoch_loss
        }
        yaml.dump(statics, file)
    checkpoint_data = {
        'epoch_id': epoch_id,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss
    }
    torch.save(checkpoint_data,
                str(checkpoint_path / (data_name+'_checkpoint.pt')))
    # if epoch_id % 10 == 0:
    #     shutil.copy(str(checkpoint_path / ('last_epoch.pt')),
    #                 str(checkpoint_path / (checkpoint_name + '.pt')))