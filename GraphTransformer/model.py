from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn

'''
define the GraphTransformer model and forward pass
'''
class GraphTransformer(nn.Module):
    def __init__(self, data_features, embed_dim, num_heads, mlp_dim, num_layers, num_classes):
        super().__init__()

        # define the graph convolution layer
        self.graphconv = GCNConv(data_features, data_features)

        # create a learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, embed_dim))

        # define the vision transformer layers and output classification head
        self.embedding = nn.Linear(in_features=data_features, out_features=embed_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x, edge_idx, batch, pos_enc):
        
        # graph convolution layer
        x = self.graphconv(x, edge_idx)
        x = x.relu()

        # reformat graph data for the transformer and classification layer

        # add the positional encoding to the node features
        x = self.embedding(x)
        x += pos_enc

        # separate the individual graphs and add the CLS token
        num_graphs = batch.unique()
        graphs = [x[batch == g] for g in num_graphs]
        cls_graphs = [torch.cat([self.cls_token, g], dim=0) for g in graphs]

        # run each graph through the transformer layer and retrieve the output cls token
        cls_outputs = []

        for graph in cls_graphs:
            # add a batch dimension for the transformer
            graph = graph.unsqueeze(0)

            # output has shape (B=1, N, embed_dim)
            transformer_output = self.transformer_encoder(graph)
            cls_output = transformer_output[0, 0, :]
            cls_outputs.append(cls_output)

        # stack CLS outputs into a batched tensor
        cls_outputs = torch.stack(cls_outputs, dim=0)

        x = self.fc_out(cls_outputs)

        return x