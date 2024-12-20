import re
import os
import math
import torch
import torchvision.transforms as transforms
from skimage.io import imread

'''
function definitions to apply positional encoding, generate feature matrices and edge lists for image graphs
'''

# generate a positional encoding term for each patch in an image
def sinusoidal_positional_encoding(grid_positions, embed_dim):

    assert embed_dim % 2 == 0, "Embedding dimension must be even"

    # use half of vector to encode row, half to encode column
    half_dim = embed_dim // 2
    div_term = torch.exp(torch.arange(0, half_dim, 2).float() * (-math.log(10000.0) / half_dim))

    # extract each index
    row_pos = grid_positions[:, 0].unsqueeze(1)  
    col_pos = grid_positions[:, 1].unsqueeze(1)  

    # compute row positional encoding
    row_encoding = torch.zeros((grid_positions.shape[0], half_dim))
    row_encoding[:, 0::2] = torch.sin(row_pos * div_term)
    row_encoding[:, 1::2] = torch.cos(row_pos * div_term)

    # compute column positional encoding
    col_encoding = torch.zeros((grid_positions.shape[0], half_dim))
    col_encoding[:, 0::2] = torch.sin(col_pos * div_term)
    col_encoding[:, 1::2] = torch.cos(col_pos * div_term)

    # concatenate row and column encodings
    positional_encoding = torch.cat([row_encoding, col_encoding], dim=1)

    return positional_encoding


# generate a pair of lists for edges in graphs
def get_edge_lists(path_to_patches, embed_dim):

    # parse the i, j coordinates from the filename of each patch
    pattern = r'(\d+)_(\d+)\.jpeg'

    patch_coords = []

    for file in path_to_patches:

        filepath = os.path.basename(file)
        match = re.match(pattern, filepath)

        if match:
            i, j = map(int, match.groups())
            patch_coords.append((i, j))

    # scan through the coordinates list and find neighbors for each patch
    source_list = []
    neighbor_list = []

    # loop over every node
    for i, loc in enumerate(patch_coords):

        # check if its 8 surrounding neighbors exist
        # add current node index, neighbor index to lists if so
        for row in [-1, 0, 1]:
            for col in [-1, 0, 1]:

                # if the neighbor node exists, add it to the list...
                try:
                    neighbor = patch_coords.index((loc[0] + row, loc[1] + col))

                    # (don't add self loop)
                    if not i == neighbor:
                        source_list.append(i)
                        neighbor_list.append(neighbor)

                # ... otherwise do nothing
                except ValueError:
                    pass

    # preprocess this data for torch_geometric.data.Data object later
    edge_idx = torch.tensor([source_list,
                             neighbor_list], dtype = torch.long)
                             
    # also get the positional encoding for each patch in the image with the (i, j) coordinate list
    pos_enc = sinusoidal_positional_encoding(torch.tensor(patch_coords), embed_dim)

    return edge_idx, pos_enc


# callable resize function to transform patches, as was done in feature extractor training
class Resize(object):
    def __init__(self, shape):
        self.shape = eval(shape)
        self.resize = transforms.Resize((self.shape[0], self.shape[1]))
        
    def __call__(self, img):
        return self.resize(img)


# generate feature vector for each patch in the image directory
def get_feature_matrix(net, device, path_to_patches, apply_reshape):

    # prepare list to collect all patches to batch process images
    patches = []

    for path in path_to_patches:

        # read the patch and convert to appropriate format for the ResNet
        patch = imread(path)
        patch = patch.astype(float)
        patch = torch.from_numpy(patch)
        patch.unsqueeze_(0)
        patch = patch.permute(0, 3, 1, 2)
        patch = apply_reshape(patch)
        patch = patch.double()/255

        patches.append(patch)

    # convert list of patches to batched torch tensor and mount on gpu
    patches = torch.cat(patches, dim=0).to(device)

    # forward pass through the model and save the feature matrix
    with torch.no_grad():
        features, _ = net(patches)

    features.cpu()

    return features
    