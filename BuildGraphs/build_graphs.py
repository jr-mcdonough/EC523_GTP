import yaml
import os
import sys
import glob
import torch
import pandas as pd
import argparse

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


# when path to current directory is resolved, import functions for building graphs
from BuildGraphs.utils import get_edge_lists, get_feature_matrix, Resize
from FeatureExtractor.model import feature_extractor


# open the yaml config file and read the sub-dictionary
config = yaml.load(open(os.path.join(root_dir, args.yaml_file), 'r'), Loader=yaml.FullLoader)
model_params = config['model']
graph_params = config['build_graphs']
dataset_params = config['dataset']
transformer_params = config['graph_transformer']


# set the directory of patches and the csv file with patch data
patches_dir = os.path.join(root_dir, graph_params['patch_dir'])
csv_data = pd.read_csv(os.path.join(root_dir, graph_params['label_csv']))

# make dicts with the isup grades and Gleason scores as keys of their image ids
image_id_to_grade = csv_data.set_index('image_id')['isup_grade'].to_dict()  # image_id -> isup_grade
image_id_to_gleason = csv_data.set_index('image_id')['gleason_score'].to_dict()  # image_id -> gleason_score


# loop through each patch parent directory, saving its path and labels
# prepare lists to store data
image_paths = []
image_names = []
isup_grades = []
gleason_scores = []

# loop over just the top-level subdirectories
for entry in os.listdir(patches_dir):
    full_path = os.path.join(patches_dir, entry)

    if os.path.isdir(full_path):

        image_id = os.path.basename(entry)

        # add directory path to list and name of image
        image_paths.append(full_path)
        image_names.append(image_id)

        # add labels to lists
        isup_grades.append(image_id_to_grade.get(image_id, None))
        gleason_scores.append(image_id_to_gleason.get(image_id, None))


# load the model checkpoint dict
checkpoint = torch.load(os.path.join(root_dir, graph_params['ResNet_pth']))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# instantiate the ResNet18 feature extractor and load pretrained weights
model = feature_extractor(model_out=model_params['out_dim'])
model.load_state_dict(checkpoint['model_state_dict'])
model = model.double()
model = model.to(device)
model.eval()

# instantiate callable class to resize patches, as was done in ResNet training
patch_resize = Resize(dataset_params['input_shape'])


# for each parent image directory, generate data needed to make graph:
# features for each patch, patch edges, label

# loop over path to every folder of patches, with its two ids
for i, (path, name, isup) in enumerate(zip(image_paths, image_names, isup_grades)):

    # get a list of all patch files in the image folder
    patch_dir = os.path.join(path, '1.0')
    patches = glob.glob(os.path.join(patch_dir, '*.jpeg'))

    # get the feature matrix for each patch
    X = get_feature_matrix(model, device, patches, patch_resize)

    torch.cuda.empty_cache()

    # get the list of edges between neighboring patches and the positional encoding for each patch
    edge_idx, pos_enc = get_edge_lists(patches, transformer_params['embed_dim'])

    graph_dict = {'features': X,
                  'edges': edge_idx,
                  'label': isup,
                  'positional_encoding': pos_enc
                  }

    torch.save(graph_dict, os.path.join(root_dir, graph_params['save_dir'], '%s_graph_data.pth' % name))

    del X, edge_idx, pos_enc

    # track progress
    if i % 10 == 9:
        print(f'{i+1} total graphs prepared')