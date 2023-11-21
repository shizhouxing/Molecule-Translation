import argparse
import os
import torch
import torch.nn as nn
from dataset import load_data, get_data_loader
from models import *
from utils import *
from tqdm import tqdm
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='node', choices=['node', 'link'])
parser.add_argument('--model-graph', type=str, default='GCN_tiny', help='Encoder for the graph')
parser.add_argument('--hidden-dim', type=int, default=512)
parser.add_argument('--num-epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--log-interval', type=int, default=1000)
parser.add_argument('--image-size', type=int, default=8)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--data-proportion', type=float, default=1.0, help='Optionally we may only use part of the data')
parser.add_argument('--num-dl-workers', type=int, default=8, help='Number of workers for the data loader')
parser.add_argument('--load', type=str, help='Load a trained checkpoint')
parser.add_argument('--infer', action='store_true')
parser.add_argument('--save-dir', type=str, default='output', help='Directory for saving trained models')
parser.add_argument('--save-infer', type=str, help='Path for saving inference results')
parser.add_argument('--use-images', action='store_true')
parser.add_argument('--vision-model', type=str, default=None, choices=[None, 'pure', 'combined'])

args = parser.parse_args()

def run_epoch(model, data_loader, opt=None):
    loss_func = nn.CrossEntropyLoss()
    meter = MultiAverageMeter()

    saved_results = [] 
    y_true_all, y_pred_all = [], []
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        if args.device == 'cuda':
            data.x, data.y = data.x.to(args.device), data.y.to(args.device)
            data.edge_index = data.edge_index.to(args.device)
            data.edge_attr = data.edge_attr.to(args.device)
        if args.task == "node":
            if args.use_images:
                x_graph = data.x[:, :-args.image_size**2]
                x_vision = data.x[x_graph.sum(dim=-1) == 0][:, -args.image_size**2:]
            else:
                x_graph = data.x
            if args.vision_model is None:
                y = model(x_graph, data.edge_index, data.edge_attr)[x_graph.sum(dim=-1) == 0]
            elif args.vision_model == 'pure':
                assert args.use_images
                y = model(x_vision)
            elif args.vision_model == 'combined':
                assert args.use_images
                y = model(x_graph, data.edge_index, data.edge_attr, x_vision)#[x_graph.sum(dim=-1) == 0]
            loss = loss_func(y, data.y)
            pred = y.argmax(dim=-1)
            acc = (pred == data.y).float().mean()
            y_pred_all += pred.tolist()
            y_true_all += data.y.tolist()
        elif args.task == "link":
            y = model(data.x, data.edge_index, data.edge_attr)[data.edge_attr.sum(dim=-1)==0]
            loss = loss_func(y, data.y)
            pred = y.argmax(dim=-1)
            acc = (pred == data.y).float().mean()
        size = y.shape[0]
        meter.update('loss', loss, size)
        meter.update('acc', acc, size)
        if opt:
            loss.backward()
            opt.step()
            opt.zero_grad()
        if (i + 1) % args.log_interval == 0:
            print(f'Step {i+1}/{len(data_loader)}: {meter}')
        if args.save_infer:
            saved_results.append({
                'x': data.x.cpu(),
                'y': data.y.cpu(),
                'edge_index': data.edge_index.cpu(),
                'edge_attr': data.edge_attr.cpu(),
                'output': y.cpu(),
            })

    if args.task == "node":
        meter.update('f1_macro', f1_score(y_true_all, y_pred_all, average='macro'))
        meter.update('f1_micro', f1_score(y_true_all, y_pred_all, average='micro'))

    print(meter)

    if args.save_infer:
        torch.save(saved_results, args.save_infer)
        print(f'Results saved to {args.save_infer}')


if __name__ == '__main__':
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_data, test_data = load_data(task=args.task, 
        data_proportion=args.data_proportion, use_images=args.use_images, 
        image_size=args.image_size)
    train_data_loader = get_data_loader(train_data, batch_size=args.batch_size, train=True, num_workers=args.num_dl_workers)
    test_data_loader = get_data_loader(test_data, batch_size=args.batch_size, train=False, num_workers=args.num_dl_workers)
    
    if args.task == 'node':
        input_dim = num_classes = train_data.num_atom_types
        if 'GAT' in args.model_graph:
            model_graph = eval(args.model_graph)(input_dim, args.hidden_dim, args.hidden_dim, 
                train_data.num_bond_types)
        else:
            model_graph = eval(args.model_graph)(input_dim, args.hidden_dim, args.hidden_dim)        
        if args.vision_model == 'pure':
            model = PureVisionNodeClassifier(num_classes)
        elif args.vision_model == 'combined':
            model = CombinedNodeClassifier(model_graph, num_classes)
        else:
            model = NodeClassifier(model_graph, args.hidden_dim, num_classes)
        print('Node classification Model:', model)
    elif args.task == 'link':
        input_dim = train_data.num_atom_types
        if 'GAT' in args.model_graph:
            model_graph = eval(args.model_graph)(input_dim, args.hidden_dim, args.hidden_dim, train_data.num_bond_types)
        else:
            model_graph = eval(args.model_graph)(input_dim, args.hidden_dim, args.hidden_dim)
            # raise Exception("Only GAT model works for Link Classificaiton")
        model = LinkClassification(model_graph, args.hidden_dim, train_data.num_bond_types)
        print('Link classification Model:', model)
    else:
        raise NotImplementedError

    model = model.to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if args.load:
        print(f'Loading checkpoint {args.load}')
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint)
    
    if args.infer:
        print('Running inference')
        with torch.no_grad():
            run_epoch(model, test_data_loader)
    else:
        for t in range(args.num_epochs):
            print(f'Training epoch {t + 1}')
            run_epoch(model, train_data_loader, opt)
            print('Testing')
            with torch.no_grad():
                run_epoch(model, test_data_loader)
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'ckpt_{t+1}'))
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'ckpt_final'))
