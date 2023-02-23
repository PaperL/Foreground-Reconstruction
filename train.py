import torch
from termcolor import colored
from tqdm import tqdm
import util.train_func as train_func

if __name__ == '__main__':
    args, config, checkpoint_path, device \
        = train_func.init_exp()
    data \
        = train_func.init_data(config)
    model, optimizer \
        = train_func.init_net(device, data['lut'])

    total_epoch = 100
    start_epoch = 0 if args.checkpoint == None \
        else train_func.load_checkpoint(
            model, optimizer, checkpoint_path, args.checkpoint)

    model.train()
    for epoch_id in range(start_epoch, total_epoch):
        print(colored(f'Epoch {epoch_id}', 'yellow'))
        loss_sum = 0.
        progress_bar = tqdm(range(config['LOOP_TIME_PER_EPOCH']))
        for i in enumerate(progress_bar):
            data['gt'].to(device)
            data['mask'].to(device)
            data['composite'].to(device)
            optimizer.zero_grad()

            output = model(data['gt'])

            loss = train_func.calculate_fMSE(data['composite'], output, data['mask'])
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        epoch_loss = loss_sum

        print(f'Epoch result:')
        print(f'Loss: {colored(str(round(epoch_loss,2)), "green")}')
        print('')

        train_func.save_checkpoint(
            checkpoint_path, model, optimizer, epoch_id, epoch_loss)
