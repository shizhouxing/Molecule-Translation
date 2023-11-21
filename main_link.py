""" For link existence """
import argparse
import torch
import torch.nn as nn
from dataset_link import load_data, get_data_loader
from models import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='link', choices=['node','link'])
parser.add_argument('--hidden-dim', type=int, default=128)
parser.add_argument('--num-epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()

def run_epoch(model, data_loader, opt=None):
    loss_func = nn.CrossEntropyLoss()
    meter = MultiAverageMeter()
    for i, data in enumerate(data_loader):
        y = model(data.x, data.edge_index)
        ytrue = model.y_true
        # print(ytrue.size(dim=0))
        loss = loss_func(y, ytrue)
        pred = y.argmax(dim=-1)
        acc = (pred == ytrue).float().mean()
        # TODO: add other metrics
        size = y.shape[0]
        meter.update('loss', loss, size)
        meter.update('acc', acc, size)
        if opt:
            loss.backward()
            opt.step()
            opt.zero_grad()
        if (i + 1) % args.log_interval == 0:
            print(f'Step {i+1}/{len(data_loader)}: {meter}')

if __name__ == '__main__':
    train_data, test_data = load_data(task=args.task)
    train_data_loader = get_data_loader(train_data, batch_size=args.batch_size, train=True)
    test_data_loader = get_data_loader(test_data, batch_size=args.batch_size, train=False)
    
    if args.task == 'node':
        input_dim = num_classes = train_data.num_atom_types
    elif args.task == 'link':
        input_dim = num_classes = train_data.num_atom_types
    else:
        raise NotImplementedError

    model_graph = GCN(input_dim, args.hidden_dim, args.hidden_dim)
    model = LinkClassifier(model_graph, args.hidden_dim, num_classes)
    print('Model:', model)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for t in range(args.num_epochs):
        print(f'Training epoch {t + 1}')
        run_epoch(model, train_data_loader, opt)
        print('Testing')
        run_epoch(model, train_data_loader, opt)
