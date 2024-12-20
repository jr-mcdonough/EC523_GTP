import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

'''
Model architecture and loss function for the feature extractor

This code is copied of slightly modified from:
https://github.com/vkola-lab/tmi2022/tree/main/feature_extractor/data_aug

feature_extractor is implemented from scratch, following the original framework.
NTXentLoss is copied directly from source.
'''

class feature_extractor(nn.Module):
  def __init__(self, model_out):
    super(feature_extractor, self).__init__()

    self.resnet = models.resnet18(weights=None, norm_layer=nn.InstanceNorm2d)
    resnet_outfeats = self.resnet.fc.in_features
    self.resnet.fc = nn.Identity()

    self.MLP1 = nn.Linear(in_features=resnet_outfeats, out_features=resnet_outfeats)
    self.MLP2 = nn.Linear(in_features=resnet_outfeats, out_features=model_out)

  def forward(self, x):

    x = self.resnet(x)
    h = x.squeeze()

    x = self.MLP1(x)
    x = F.relu(x)
    x = self.MLP2(x)

    return h, x
  

class NTXentLoss(nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)