from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from graph_image_builder import graph_pic_gen
import torch.nn as nn 
import torch
from torchvision import transforms
from torchvision import datasets, transforms
import random



def create_data_loaders(batch_size, train_split=0.8):

    dataset = []
    for i in range(11000):
        graph = graph_pic_gen()
        dataset.append(graph)

    # Adjust split sizes based on your dataset
    train_size = int(train_split * len(dataset))
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    print(f"Total dataset size: {len(dataset)}")
    print(f"Training dataset size: {train_size}")
    print(f"Validation dataset size: {valid_size}")
    print("======")
    
    return train_dataset, valid_dataset

