import numpy as np
import matplotlib.pyplot as plt
import torch


def load(dataset):
    data_path = "output/"+dataset+".pt"
    with open(data_path, 'rb') as file:
        data = torch.load(file)
    y_true = []
    y_pred = []
    for d in data:
        y_true.append(d["y"])
        y_pred.append(d["output"].argmax(-1))
    y_t = torch.cat((y_true),0)
    y_p = torch.cat((y_pred),0)
    graph(y_t, y_p)
    # print(y_t.size())
    # print(y_p.size())
    # print(max(y_t))
    # print(max(y_p))

def graph(y_true,y_pred):
    data_true = [0 for i in range(12)]
    data_pred = [0 for i in range(12)]
    for i in range(len(y_true)):
        data_true[y_true[i] ] += 1
        data_pred[y_pred[i] ] += 1

    fig_true = plt.figure(1)
    # ax = fig.add_axes([0,0,1,1])
    NodeType = ['B','Br','C','Cl','F','H','I','N','O','P','S','Si']
    # ax.bar(bondType,data)
    x_pos = np.arange(len(NodeType))

    # Create bars and choose color
    plt.bar(x_pos, data_true, color=(0.5, 0.1, 0.5, 0.6))

    # Add title and axis names
    plt.title('GAT True Nodes')
    plt.xlabel('Node Types')
    plt.ylabel('Node Numbers')

    # Create names on the x axis
    plt.xticks(x_pos, NodeType)

    fig_pred = plt.figure(2)

    # Create bars and choose color
    plt.bar(x_pos, data_pred, color=(0.5, 0.1, 0.5, 0.6))

    # Add title and axis names
    plt.title('GAT Predicted Nodes')
    plt.xlabel('Node Types')
    plt.ylabel('Node Numbers')

    # Create names on the x axis
    plt.xticks(x_pos, NodeType)
    # Show graph
    plt.show()
    return


load("node_GAT")
# load("node_GCN")






#link graphs
# def graph(y_true,y_pred):
#     data_true = [0 for i in range(10)]
#     data_pred = [0 for i in range(10)]
#     for i in range(len(y_true)):
#         data_true[y_true[i] - 1] += 1
#         data_pred[y_pred[i] - 1] += 1
#
#     fig_true = plt.figure(1)
#     # ax = fig.add_axes([0,0,1,1])
#     bondType = [1,2,3,4,5,6,7,8,9,10]
#     # ax.bar(bondType,data)
#     x_pos = np.arange(len(bondType))
#
#     # Create bars and choose color
#     plt.bar(x_pos, data_true, color=(0.5, 0.1, 0.5, 0.6))
#
#     # Add title and axis names
#     plt.title('GCN True Link')
#     plt.xlabel('Bond Types')
#     plt.ylabel('Bond Numbers')
#
#     # Create names on the x axis
#     plt.xticks(x_pos, bondType)
#
#     fig_pred = plt.figure(2)
#
#     # Create bars and choose color
#     plt.bar(x_pos, data_pred, color=(0.5, 0.1, 0.5, 0.6))
#
#     # Add title and axis names
#     plt.title('GCN Predicted Link ')
#     plt.xlabel('Bond Types')
#     plt.ylabel('Bond Numbers')
#
#     # Create names on the x axis
#     plt.xticks(x_pos, bondType)
#     # Show graph
#     plt.show()
#     return

# load("link_GAT")
# load("link_GCN")