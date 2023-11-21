import pickle
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

class MoleculeDataset(Dataset):
    def __init__(self, data_raw, train, task, num_atom_types, num_bond_types, 
                atom2idx, use_images, image_size=8):
        super().__init__()
        self.train = train
        self.data_raw = data_raw
        self.task = task
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.atom2idx = atom2idx
        self.use_images = use_images
        self.image_size = image_size
        
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
        elif self.task == "link":
            self.data_indices = []
            for i, graph in enumerate(self.data_raw):
                for j in range(len(graph['bonds'])):
                    self.data_indices.append((i,j))
        else:
            raise NotImplementedError

    def len(self):
        return len(self.data_indices)

    def get(self, index):
        if self.task == "node":
            # Node_id is the id of the node to be predicted
            graph_id, node_id = self.data_indices[index]
            graph = self.data_raw[graph_id]
            # Node features of shape [num_nodes, num_node_features]
            x = torch.zeros(len(graph['atoms']), self.num_atom_types)
            if self.use_images:
                x_image = torch.ones(len(graph['atoms']), self.image_size, self.image_size)
            for i, atom in enumerate(self.data_raw[graph_id]['atoms']):
                if i != node_id:
                    # Not the one to be predicted
                    # One-hot feature
                    x[i][self.atom2idx[atom[0]]] = 1
                else:
                    y = self.atom2idx[atom[0]]
                if self.use_images:
                    img = torchvision.io.read_image(f'data/images/{graph["id"]}_atom_{i}.png')
                    # Normalize to [0, 1]
                    img = img[0] / 255
                    x_image.data[i, :img.shape[0], :img.shape[1]] = img
            x = torch.cat([x, x_image.view(-1, self.image_size**2)], dim=-1)
            # Graph connectivity matrix of shape [2, num_edges]
            edge_index = torch.tensor([[bond[i] for bond in self.data_raw[graph_id]['bonds']] for i in range(2)])
            # Types of the links of shape [num_edges]
            edge_types = torch.tensor([bond[2] for bond in self.data_raw[graph_id]['bonds']])
            edge_attr = F.one_hot(edge_types, self.num_bond_types).float()
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        elif self.task == "link":
            graph_id, edge_id = self.data_indices[index]
            graph = self.data_raw[graph_id]
            x = torch.zeros(len(graph['atoms']), self.num_atom_types)
            for i,atom in enumerate(self.data_raw[graph_id]['atoms']):
                x[i][self.atom2idx[atom[0]]] = 1
            edge_index = torch.tensor([[bond[i] for bond in self.data_raw[graph_id]['bonds']] for i in range(2)])
            edge_types = torch.tensor([bond[2] for bond in self.data_raw[graph_id]['bonds']])
            edge_attr = F.one_hot(edge_types, self.num_bond_types).float()
            # Mask
            edge_attr[edge_id] = 0
            y = edge_types[edge_id]
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        else:
            raise NotImplementedError

def load_data(task, data_proportion=1.0, use_images=False, image_size=8):
    print('Loading data')
    with open('data/data_graphs.pkl', 'rb') as file:
        data = pickle.load(file)
    print(f'Using {data_proportion} of the data')
    data = data[:int(len(data) * data_proportion)]

    # Map atom symbols to int
    if True:
        # Fix the dictionary
        num_atom_types, num_bond_types = 12, 11
        atom2idx = {'B': 0, 'Br': 1, 'C': 2, 'Cl': 3, 'F': 4, 'H': 5, 'I': 6, 'N': 7, 'O': 8, 'P': 9, 'S': 10, 'Si': 11}
    else:
        # Recompute
        atoms = set()
        num_bond_types = 0
        for graph in data:
            atoms.update(set([atom[0] for atom in graph['atoms']]))
            num_bond_types = max(num_bond_types, 1 + max(bond[-1] for bond in graph['bonds']))
        num_atom_types = len(atoms)
        atom2idx = {}
        for i, atom in enumerate(sorted(atoms)):
            atom2idx[atom] = i
        print('atom2idx', atom2idx)
        print('num_atom_types', num_atom_types)
        print('num_bond_types', num_bond_types)

    num_training = int(len(data) * 0.8)
    if use_images:
        print('Using small data with images')
        data_training = data[:int(len(data)*0.08)]
        data_test = data[-int(len(data)*0.02):]
    else:
        data_training = data[:num_training]
        data_test = data[num_training:]
    train_data = MoleculeDataset(data_training, train=True, task=task, 
        num_atom_types=num_atom_types, num_bond_types=num_bond_types, 
        atom2idx=atom2idx, use_images=use_images, image_size=image_size)
    test_data = MoleculeDataset(data_test, train=False, task=task, 
        num_atom_types=num_atom_types, num_bond_types=num_bond_types, 
        atom2idx=atom2idx, use_images=use_images, image_size=image_size)
    print(f'{len(train_data)} training examples and {len(test_data)} test examples')
    return train_data, test_data

def get_data_loader(data, batch_size, train, num_workers=1):
    return DataLoader(data, batch_size=batch_size, shuffle=train, num_workers=num_workers)

if __name__ == '__main__':
    train_data, test_data = load_data('node', use_images=True)
    print(train_data.get(0))
