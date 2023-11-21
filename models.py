import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv

class GCN_large(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index).relu()
        x = self.linear(x).relu()
        return x 

class GAT_medium(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, heads=8):
        super().__init__()
        assert hidden_channels % heads == 0
        self.conv1 = GATConv(in_channels, hidden_channels//heads, heads, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_channels, hidden_channels//heads, heads, edge_dim=edge_dim)
        self.conv3 = GATConv(hidden_channels, hidden_channels, 1, edge_dim=edge_dim)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = self.conv3(x, edge_index, edge_attr).relu()
        x = self.linear(x).relu()
        return x

class GAT_large(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, heads=8):
        super().__init__()
        assert hidden_channels % heads == 0
        self.conv1 = GATConv(in_channels, hidden_channels//heads, heads, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_channels, hidden_channels//heads, heads, edge_dim=edge_dim)
        self.conv3 = GATConv(hidden_channels, hidden_channels//heads, heads, edge_dim=edge_dim)
        self.conv4 = GATConv(hidden_channels, hidden_channels//heads, heads, edge_dim=edge_dim)
        self.conv5 = GATConv(hidden_channels, hidden_channels, 1, edge_dim=edge_dim)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = self.conv3(x, edge_index, edge_attr).relu()
        x = self.conv4(x, edge_index, edge_attr).relu()
        x = self.conv5(x, edge_index, edge_attr).relu()
        x = self.linear(x).relu()
        return x     

class GAT_large_no_edge_attr(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, heads=8):
        super().__init__()
        assert hidden_channels % heads == 0
        self.conv1 = GATConv(in_channels, hidden_channels//heads, heads)
        self.conv2 = GATConv(hidden_channels, hidden_channels//heads, heads)
        self.conv3 = GATConv(hidden_channels, hidden_channels//heads, heads)
        self.conv4 = GATConv(hidden_channels, hidden_channels//heads, heads)
        self.conv5 = GATConv(hidden_channels, hidden_channels, 1)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index).relu()
        x = self.linear(x).relu()
        return x                   

class NodeClassifier(nn.Module):
    def __init__(self, encoder, dim_embeddings, num_classes):
        super().__init__()
        self.encoder = encoder
        self.dim_embeddings = dim_embeddings
        self.num_classes = num_classes
        self.cls = nn.Linear(self.dim_embeddings, num_classes)

    def forward(self, nodes, edge_index, edge_attr):
        embeddings = self.encoder(nodes, edge_index, edge_attr)
        y = self.cls(embeddings)
        return y

class cnn_5layer(nn.Module):
    def __init__(self, in_ch=1, width=64, linear_size=512, in_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((in_dim//2) * (in_dim//2) * 2 * width, linear_size),
        )

    def forward(self, x):
        return self.encoder(x)

class PureVisionNodeClassifier(nn.Module):
    def __init__(self, num_classes, dim_embeddings=512):
        super().__init__()
        self.encoder = cnn_5layer(linear_size=dim_embeddings)
        self.dim_embeddings = dim_embeddings
        self.num_classes = num_classes
        self.cls = nn.Linear(dim_embeddings, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), 1, 8, 8)
        embeddings = self.encoder(x)
        y = self.cls(embeddings)
        return y

class CombinedNodeClassifier(nn.Module):
    def __init__(self, model_graph, num_classes, dim_embeddings=512):
        super().__init__()
        self.encoder_graph = model_graph
        self.encoder_image = cnn_5layer(linear_size=dim_embeddings)
        self.dim_embeddings = dim_embeddings
        self.num_classes = num_classes
        self.cls = nn.Linear(dim_embeddings * 2, num_classes)

    def forward(self, nodes, edge_index, edge_attr, x_vision):
        embeddings_graph = self.encoder_graph(nodes, edge_index, edge_attr)
        embeddings_graph = embeddings_graph[nodes.sum(dim=-1) == 0]
        x_vision = x_vision.view(x_vision.size(0), 1, 8, 8)
        embeddings_image = self.encoder_image(x_vision)
        y = self.cls(torch.cat([embeddings_graph, embeddings_image], dim=-1))
        return y

class LinkClassification(nn.Module):
    def __init__(self, encoder, dim_embeddings, num_classes):
        super().__init__()
        self.encoder = encoder
        self.dim_embeddings = dim_embeddings
        self.num_classes = num_classes
        self.cls = nn.Linear(self.dim_embeddings*2, num_classes)

    def forward(self, nodes, edges, attrib):
        embeddings = self.encoder(nodes, edges, attrib)
        new_embeddings = torch.cat([embeddings[edges[0]], embeddings[edges[1]]], dim=-1)
        y = self.cls(new_embeddings)
        return y

class LinkClassifier(nn.Module):
    def __init__(self, encoder, dim_embeddings, num_classes):
        super().__init__()
        self.encoder = encoder
        self.dim_embeddings = dim_embeddings
        self.num_classes = num_classes
        # link existence, only contain true or false
        self.cls = nn.Linear(self.dim_embeddings*2, 2)

    def forward(self, nodes, edges):
        embeddings = self.encoder(nodes, edges)
        Nedges = edges.size(dim=0)
        Nnodes = nodes.size(dim=0)
        newembeddings = torch.zeros(Nnodes*(Nnodes-1)//2,embeddings.size(dim=1)*2)
        y_true = torch.zeros(Nnodes*(Nnodes-1)//2,1,dtype=torch.int8)
        count = 0
        # FIXME replace this for-loop as it is very slow (see `LinkClassification`).
        for source in range(Nnodes-1):
            for sink in range(source+1,Nnodes):
                newembeddings[count] = torch.cat((embeddings[source],embeddings[sink]),dim=0)
                if torch.any((edges[0]==source)&(edges[1]==sink)):
                    y_true[count] = 1
                count += 1
        y = self.cls(newembeddings)
        self.y_true = y_true.flatten().long()
        return y
