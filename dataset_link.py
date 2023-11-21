""" For link existence """
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

class LinkDataset(Dataset):
    def __init__(self, data_raw, train, task):
        super().__init__()
        self.train = train
        self.data_raw = data_raw
        self.task = task

        # Map atom symbols to int
        self.atoms = set()
        self.num_bond_types = 0
        for graph in self.data_raw:
            atoms = set([atom[0] for atom in graph['atoms']])
            self.atoms.update(atoms)
            self.num_bond_types = max(self.num_bond_types, 1 + max(bond[-1] for bond in graph['bonds']))
        self.num_atom_types = len(self.atoms)
        self.atom2idx = {}
        for i, atom in enumerate(self.atoms):
            self.atom2idx[atom] = i

        if self.task == 'node':
            # Node classification task
            self.data_indices = []
            for i, graph in enumerate(self.data_raw):
                for j in range(len(graph['atoms'])):
                    # For each example in node classification, we mask the information
                    # on one single node and predict this node, and thus we have 
                    # one example for every node in every graph.
                    # For now, ignoring the charge of each node, and working the atom first.
                    self.data_indices.append((i, j))

        elif self.task == 'link':
            self.data_indices = []
            for i, graph in enumerate(self.data_raw):
                for j in range(len(graph['atoms'])):
                    self.data_indices.append((i, j))

        else:
            raise NotImplementedError

    def len(self):
        return len(self.data_indices)

    def get(self, index):
        # Node_id is the id of the node to be predicted
        graph_id, node_id = self.data_indices[index]
        graph = self.data_raw[graph_id]
        # Node features of shape [num_nodes, num_node_features]
        x = torch.zeros(len(graph['atoms']), self.num_atom_types)

        for i, atom in enumerate(self.data_raw[graph_id]['atoms']):
            if i != node_id:
                # Not the one to be predicted
                # One-hot feature
                x[i][self.atom2idx[atom[0]]] = 1
            else:
                y = self.atom2idx[atom[0]]
        # Graph connectivity matrix of shape [2, num_edges]
        edge_index = torch.tensor([[bond[i] for bond in self.data_raw[graph_id]['bonds']] for i in range(2)])
        # Types of the links of shape [num_edges] 
        edge_types = torch.tensor([bond[2] for bond in self.data_raw[graph_id]['bonds']])
        edge_attr = F.one_hot(edge_types, self.num_bond_types)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def load_data(task):
    print('Loading data')
    with open('data_graphs.pkl', 'rb') as file:
        data = pickle.load(file)
    num_training = int(len(data) * 0.8)
    train_data = LinkDataset(data[:num_training], train=True, task=task)
    test_data = LinkDataset(data[num_training:], train=False, task=task)
    print(f'{len(train_data)} training examples and {len(test_data)} test examples')
    return train_data, test_data

def get_data_loader(data, batch_size, train):
    return DataLoader(data, batch_size=batch_size, shuffle=train)

if __name__ == '__main__':
    train_data, test_data = load_data('link')
    print(train_data.get(0))