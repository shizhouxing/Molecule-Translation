# Relational Learning for Molecule Translation from Noisy Images

UCLA CS 267 (Spring 2022) project by Zhouxing Shi, Boyang Fu, Yifei Xu, Yuxin Wu.

We propose to improve [molecule translation](https://www.kaggle.com/competitions/bms-molecular-translation) by leveraging the graphical structure information of molecules with Graph Neural Networks.

## Data

Download generated data from [Google Drive](https://drive.google.com/drive/folders/1Rykr7ZxcArQ7mIdV4WGeAdnAGS25g1b-?usp=sharing), and extract the data to the `data/` folder.
It is recommended to use the following commands wtih `gdown` which can be installed by `pip`:
```bash
cd data
gdown --id 1_dvhLNvONUhIAWIsfnFKQlyQEqRXIknA
gdown --id 1Ke3aaCh_HuygDMlduxtN1xOZ-h3-cFKM
gdown --id 1tF8X4bo2n_CFuj6Jd2xhwgosXUutEWW6
gdown --id 1QFMMRqkzbrNX6WIPUhFEes5CuBbOM75r
gdown --id 1zgBBqgvv9cUVWHK8NErd-0_DixSZyARN
mkdir pretrained
cd pretrained
gdown --id 1u8MoXGBnvkuqSIbYka4C1Fg_cz2mjxuQ
gdown --id 1Xj5iooTYrXAn8Hi9RFRYB03Vv746_G_w
gdown --id 1kCmp9ZNc-S-XHIVZBWS-oJa7uKpxHzd3
unzip image_data_training.zip
unzip image_data_test.zip
mv data_subset image_training
mv data_test image_test
```

## Dependencies

Installl [PyG](https://github.com/pyg-team/pytorch_geometric) first.
Some [wheels](https://pytorch-geometric.com/whl/) provided by PyG may be helpful if is too slow to build some dependencies locally.

Installing PyG by conda:
```bash
conda create --name pyg python=3.8
conda activate pyg
pip install torch torchvision torchaudio
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
```

CUDA version for pytorch 1.8:
```bash
pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_scatter-2.0.8-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_sparse-0.6.9-cp38-cp38-linux_x86_64.whl
pip install torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.8.0%2Bcu111.html
```

Install other Python libraries:
```bash
pip install -r requirements.txt
```

## Experiments

Set a `DIR` environment variable to specify a prefix of directories for saving
trained models. 

### Node Classification

```bash
python main.py --device cuda  --save-dir $DIR\_node_GCN --model-graph GCN_large --task node 
python main.py --device cuda  --save-dir $DIR\_node_GAT --model-graph GAT_large --task node
python main.py --device cuda  --save-dir $DIR\_node_GAT_no_edge_attr --model-graph GAT_large_no_edge_attr --task node
```

### Link Classification

```bash
python main.py --device cuda  --save-dir $DIR\_link_GCN --model-graph GCN_large --task link
python main.py --device cuda  --save-dir $DIR\_link_GAT --model-graph GAT_large --task link
python main.py --device cuda  --save-dir $DIR\_link_GAT_no_edge_attr --model-graph GAT_large_no_edge_attr --task link
```

### Running Other Commands

Run `python main.py --help` to see available options.

## Pre-Trained Models

* [Graph models](https://drive.google.com/file/d/1sW_6JSMGwFI4QHh_OqtrWmLQ_PbWrgl1/view?usp=sharing)
