from torch_geometric.data import Dataset, Data
import pandas as pd
import torch

'''
define a dataset that reads paths to graph data from a .csv file and prepares a Data object
'''
class GraphDataset(Dataset):
    def __init__(self, csv_file):
        super().__init__()

        self.graphs_list = pd.read_csv(csv_file)

    def len(self):
        return len(self.graphs_list)

    def get(self, idx):
        data_path = self.graphs_list.iat[idx, 0]
        graph_data = torch.load(data_path,
                                map_location = torch.device('cpu'))

        graph = Data(x = graph_data['features'],
                     edge_index = graph_data['edges'],
                     y = graph_data['label'])

        pos_enc = Data(x = graph_data['positional_encoding'])

        return graph, pos_enc