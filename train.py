import torch
from termcolor import colored
from tqdm import tqdm
import util.train_func as train_func
from PIL import Image
import math

from pathlib import Path
import os

if __name__ == '__main__':
    args, config, checkpoint_path, device \
        = train_func.init_exp()
    # names = train_func.get_data_names(config)
    div_part_id = args.part
    div_part_num = 4

    finished_names = []
    finished_folder = Path(config['OUTPUT_PATH'])
    for x in list(finished_folder.glob('*.jpg')):
        name = str(x.stem)
        finished_names.append(name)

    todo_names = []
    with open(config['TODO_LIST'], 'r') as f:
        lines = f.readlines()
        for line in lines:
            infos = line.strip().split(' ')
            if float(infos[1]) > 100.:
                # print(infos[0], infos[1])
                todo_names.append(infos[0].split('.')[0])
    todo_names.sort()
    # print(todo_names)
    total_n = len(todo_names)
    print(colored(f'Collect {total_n} data points.', 'red'))

    each_part_num = (total_n // div_part_num) + 1
    start_id = div_part_id * each_part_num
    end_id = min(total_n, (div_part_id+1)*each_part_num)
    print(f'On GPU {args.gpu}')
    print(f'From {start_id} to {end_id}')

    luts = train_func.init_luts(config)

    for t in range(start_id,end_id):
        print(f'{start_id} {end_id}, '+colored(f't: {t}.', 'green'))
        args.data_name = todo_names[t]
        name = todo_names[t]
        if(args.data_name in finished_names):
            print(colored(f'Skip {args.data_name}.', 'yellow'))
            continue
        data = train_func.init_data(config, args)
        model, optimizer \
            = train_func.init_net(device, luts['lut'], luts['lutN'])

        total_epoch = 1
        start_epoch = 0 if args.checkpoint == None \
            else train_func.load_checkpoint(
                model, optimizer, checkpoint_path, args.checkpoint)

        model.train()
        for epoch_id in tqdm(range(start_epoch, total_epoch), leave=False, disable=True):
            # print(colored(f'Epoch {epoch_id}', 'yellow'))
            loss_sum = 0.
            progress_bar = tqdm(range(config['LOOP_TIME_PER_EPOCH']), leave=False, disable=False, mininterval=1.)
            
            gt = data['gt'].to(device)
            mask = data['mask'].to(device)
            composite = data['composite'].to(device)
            for i in (progress_bar):
                optimizer.zero_grad()

                output = model(gt)

                # loss_fn = torch.nn.MSELoss()
                # loss = loss_fn(output, composite)
                # loss = loss_fn(output*mask, composite*mask)
                loss = train_func.calculate_fMSE(output, composite, mask)

                loss.backward()
                optimizer.step()

                assert not math.isnan(loss.item())
                # loss_sum += loss.item()
                # print(loss.item() / mask_cover_rate * mask.numel(), mask_cover_rate)
                loss_val = loss.item()
                if i%100 == 0:
                    progress_bar.set_description('%.2f' % (loss_val))
                # print(loss_val, mask_area, tot_area, mask_cover_rate, loss_val*tot_area, loss_val*mask_area, loss_val*tot_area/mask_cover_rate)
            # print(final_loss_value, final_loss_value * mask_area)
                if loss_val < 98.:
                    print(colored(f'Finish in {i}th loop.', 'red'))
                    break
            progress_bar.close()
            with open(config['RERUN_INFO_OUTPUT'], 'a') as f:
                f.write(f'{name} {loss_val} {i}\n')
            # epoch_loss = loss_sum
            # print(f'Epoch result:')
            # print(f'Loss: {colored(str(round(epoch_loss,2)), "green")}')
            # print('')
            
            # Save output image
            ndarr = output.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            # name_parts = args.data_name.split('_')
            # name_parts[1] = '2'
            # # name_parts[2] = str(int(name_parts[2])+2)
            # output_name = '_'.join(name_parts) + '.jpg'
            output_name = name + '.jpg'
            im.save(str(config['OUTPUT_PATH']+'/'+output_name), quality=100)

            torch.save(model.lc0, (config['LC_OUTPUT_PATH']+'/'+name+'_0.pt'))
            torch.save(model.lc1, (config['LC_OUTPUT_PATH']+'/'+name+'_1.pt'))

            # train_func.save_checkpoint(
            #     checkpoint_path, model, optimizer, epoch_id, epoch_loss, args.data_name)
