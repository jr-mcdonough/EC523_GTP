import yaml
import argparse
import os
import sys
import numpy as np
import pickle
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from torch.utils.data import random_split

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
from GraphTransformer.dataset import GraphDataset
from GraphTransformer.model import GraphTransformer


# open the yaml config file and read the sub-dictionaries
config = yaml.load(open(os.path.join(root_dir, args.yaml_file), 'r'), Loader=yaml.FullLoader)
model_params = config['model']
graph_data_params = config['graph_dataset']
graph_transformer_params = config['graph_transformer']
checkpoints = config['transformer_checkpoints']


# load the data and prepare DataLoaders
graph_dataset = GraphDataset(csv_file = os.path.join(root_dir, graph_data_params['graphs_csv']))

val_size=graph_data_params['val_size']
total_size = len(graph_dataset)
val_length = int(total_size * val_size)
train_length = total_size - val_length

train_dataset, val_dataset = random_split(graph_dataset, [train_length, val_length])

train_loader = DataLoader(train_dataset, batch_size=graph_data_params['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=graph_data_params['batch_size'], shuffle=False)


# instantiate a GraphTransformer model, loss function, optimizer & scheduler
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GraphTransformer(data_features=model_params['out_dim'],
                         embed_dim=graph_transformer_params['embed_dim'],
                         num_heads=graph_transformer_params['num_heads'],
                         mlp_dim=graph_transformer_params['mlp_dim'],
                         num_layers=graph_transformer_params['num_layers'],
                         num_classes=graph_transformer_params['num_classes'])
model = model.double()
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = eval(graph_transformer_params['learning_rate']), weight_decay = 5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=graph_transformer_params['lr_step'], gamma=0.1)


# train the model

best_val = np.inf

for epoch in range(graph_transformer_params['epochs']):

    train_loss = []
    val_loss = []

    # training loop
    for graph_batch, encoding_batch in train_loader:

        optimizer.zero_grad()

        graph_batch = graph_batch.to(device)
        encoding_batch = encoding_batch.to(device)

        out = model(x=graph_batch.x,
            edge_idx=graph_batch.edge_index,
            batch=graph_batch.batch,
            pos_enc=encoding_batch.x)

        loss = criterion(out, graph_batch.y)

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    # update the scheduler
    scheduler.step()

    # validation loop
    if epoch % graph_transformer_params['eval_every_n_epochs'] == 0:

        with torch.no_grad():
            model.eval()

            for graph_batch, encoding_batch in val_loader:

                graph_batch = graph_batch.to(device)
                encoding_batch = encoding_batch.to(device)

                out = model(x=graph_batch.x,
                    edge_idx=graph_batch.edge_index,
                    batch=graph_batch.batch,
                    pos_enc=encoding_batch.x)

                loss = criterion(out, graph_batch.y)

                val_loss.append(loss.item())

            model.train()

        val_loss = np.mean(np.array(val_loss))
        print('\nEpoch (%d/%d): validation loss %3f' % ((epoch+1), graph_transformer_params['epochs'], val_loss))

        # keep a record of the best model for validation loss
        if val_loss < best_val:
            best_val = val_loss
            best_val_epoch = epoch

            best_val_checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }

    
    # log loss, model for epoch
    train_loss = np.mean(np.array(train_loss))

    print('Epoch (%d/%d): training loss %3f \n' % ((epoch+1), graph_transformer_params['epochs'], train_loss))

    # save epoch data

    if epoch % graph_transformer_params['eval_every_n_epochs'] == 0:

        model_checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }

        torch.save(model_checkpoint, os.path.join(root_dir, checkpoints['model'], '%03d_checkpoint.pth' % (epoch+1)))

    loss_checkpoint = {
        'epoch': (epoch+1),
        'train_loss': train_loss,
        'val_loss': val_loss
    }

    loss_file = os.path.join(root_dir, checkpoints['loss'], '%03d_loss.pkl' % (epoch+1))
    with open(loss_file, 'wb') as file:
        pickle.dump(loss_checkpoint, file)

print(f'\nBest model validation at epoch {best_val_epoch+1} with validation loss {best_val}')

torch.save(best_val_checkpoint, os.path.join(root_dir, checkpoints['model'], '%03d_best_val_checkpoint.pth' % (best_val_epoch+1)))