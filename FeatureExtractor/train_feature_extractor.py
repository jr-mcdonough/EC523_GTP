import yaml
import os
import sys
import torch
import numpy as np
import argparse
import pickle

# parse command line arguments
parser = argparse.ArgumentParser()

# most parameters saved in config file;
# just need root directory if working in Colab, and path to config file
parser.add_argument('--root_dir', default='', help='for Colab: provide current working directory')
parser.add_argument('--yaml_file', required=True, help='path to .yaml config file')

args = parser.parse_args()

# get root directory and append path to current working directory if needed
root_dir = args.root_dir

if args.root_dir != '':
    sys.path.append(root_dir)


# when path to current directory is resolved, import functions for the feature extractor
from FeatureExtractor.data_transforms import DataSetWrapper
from FeatureExtractor.model import feature_extractor, NTXentLoss


# open the yaml config file and read the sub-dictionaries
config = yaml.load(open(os.path.join(root_dir, args.yaml_file), 'r'), Loader=yaml.FullLoader)
dataset_params = config['dataset']
model_params = config['model']
loss_params = config['loss']
checkpoints = config['checkpoints']


# load the data and prepare DataLoaders
data = DataSetWrapper(batch_size=config['batch_size'],
                      num_workers=dataset_params['num_workers'],
                      valid_size=dataset_params['valid_size'],
                      s=dataset_params['s'],
                      input_shape=dataset_params['input_shape'],
                      data_path = os.path.join(root_dir, dataset_params['train_patches'])
                      )

train_loader, val_loader = data.get_data_loaders()


# instantiate a ResNet model, loss function, optimizer & scheduler
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = feature_extractor(model_out=model_params['out_dim'])
model.to(device)
model.train()

loss = NTXentLoss(device=device,
                  batch_size=config['batch_size'],
                  temperature=loss_params['temperature'],
                  use_cosine_similarity=loss_params['use_cosine_similarity']
                  )

optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=eval(config['weight_decay']))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=0, last_epoch=-1)


# train the model
for epoch in range(config['epochs']):

    n_step = 1
    train_loss = []
    val_loss = []
    best_val = np.inf

    # training loop
    for isample, jsample in train_loader:

        optimizer.zero_grad()

        isample = isample.to(device)
        jsample = jsample.to(device)

        _, ioutput = model(isample)
        _, joutput = model(jsample)

        sample_loss = loss(ioutput, joutput)

        sample_loss.backward()
        optimizer.step()

        train_loss.append(sample_loss.item())

        if n_step % config['log_every_n_steps'] == 0:
            print('Epoch (%d/%d), step %d: training loss %3f' % ((epoch+1), config['epochs'], n_step, sample_loss.item()))

        n_step += 1

    # validation loop
    if epoch % config['eval_every_n_epochs'] == 0:

        with torch.no_grad():
            model.eval()

            for isample, jsample in val_loader:

                isample = isample.to(device)
                jsample = jsample.to(device)

                _, ioutput = model(isample)
                _, joutput = model(jsample)

                sample_loss = loss(ioutput, joutput)

                val_loss.append(sample_loss.item())

            model.train()

    # log loss, model for epoch
    train_loss = np.mean(np.array(train_loss))
    val_loss = np.mean(np.array(val_loss))

    if val_loss < best_val:
        best_val = val_loss
        best_val_epoch = epoch+1

    print('\nEpoch (%d/%d): training loss %3f\nEpoch (%d/%d): validation loss %3f \n' % ((epoch+1), config['epochs'], train_loss, (epoch+1), config['epochs'], val_loss))

    # save epoch data
    if (epoch+1) % 5 == 0:
        model_checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        
        torch.save(model_checkpoint, os.path.join(root_dir, checkpoints['model'], '%02d_checkpoint.pth' % (epoch+1)))

    loss_checkpoint = {
        'epoch': (epoch+1),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    
    loss_file = os.path.join(root_dir, checkpoints['loss'], '%02d_loss.pkl' % (epoch+1))
    with open(loss_file, 'wb') as file:
        pickle.dump(loss_checkpoint, file)

    # update the scheduler past 10 epochs
    if epoch >= 10:
        scheduler.step()

print(f'\nBest model validation at epoch {best_val_epoch} with validation loss {best_val}')