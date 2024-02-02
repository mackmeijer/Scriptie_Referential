from dgl.nn import GATConv
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import dgl
import torch
import dgl.data
import matplotlib.pyplot as plt
import cv2
import random
from torch_geometric.data import Data
from torchvision.utils import save_image
from torchvision import transforms
from torchvision import datasets, transforms
import os
def graph_pic_gen():
    templates = [[0, 1], [2, 0], [2, 1], [2, 2], [3, 1], [2, 3]] # 2 x 1 templates

    images = [[r".\img\fles.png", r".\img\lama.png", r".\img\tulp.png", r".\img\cake.png", r".\img\doll.png"], 
              [r".\img\pilaar.png", r".\img\tafel.png", r".\img\stoel.png", r".\img\picknick.png", r".\img\poef.png"], 
              [r".\img\sterren.png", r".\img\schilderij.png", r".\img\art.png", r".\img\leeg.png", r".\img\heartart.png"], 
              [r".\img\gitaar.png", r".\img\tree.png", r".\img\tv.png", r".\img\cactus.png", r".\img\house.png"]]
    
    #select 2 random 2x1 templates
    seed = random.randint(0, 5)
    seed2 = random.randint(0, 5)

    #define grid
    images_list = [templates[seed][0], templates[seed][1], templates[seed2][0], templates[seed2][1]]


    x = [[x] for x in images_list]

    #generate random numbers to define the object 
    random_list = [random.randint(0, 4), random.randint(0, 4), random.randint(0, 4), random.randint(0, 4)]

    #generate list of the object file names
    images_list = [images[images_list[0]][random_list[0]], images[images_list[1]][random_list[1]],
                    images[images_list[2]][random_list[2]], images[images_list[3]][random_list[3]]]
    
    #generate image
    images = [Image.open(x) for x in images_list]
    widths, heights = images[0].size
    total_width = widths*2
    max_height = heights*2
    new_im = Image.new('RGB', (total_width, max_height))
    xy = [[0, 0], [0, 1], [1, 0], [1, 1]]
    offset = 0

    for im in images:
        x_offset, y_offset = im.size[0]*xy[offset][0], im.size[0]*xy[offset][1]
        new_im.paste(im, (x_offset, y_offset))
        offset+=1
    

    new_im.save('test_lower.png') 
    new_im = cv2.imread('test_lower.png', cv2.IMREAD_GRAYSCALE) #turn into grayscald

    #grids telling the proper relation between two objects

    # 0 = sits on top, 1 = supports, 2 = hanging above, 3 = hanging under, 4 = standing under

    # small objects, supporting objects, hanging objects, big objects

    matrix_bo_on = [[None, 0,    None, None], 
                    [None, None, None, None],
                    [2,    2,    2,       2], 
                    [None, 0,    None, None]]
    
    matrix_on_bo = [[None, None, 4, None],
                    [1,    None, 4,    1],
                    [None, None, 3, None], 
                    [None, None, 4, None]]
  
    #create edge indexes (same over all graphs)
    edge_index = torch.tensor(([0, 1,  2,  3,  0,  1, 2,  3],
                             [1, 0,  3,  2,  2,  3, 0,  1]), dtype=torch.long).t().contiguous()
    
    #preset for node object classification
    nodes = torch.zeros(4, 20)

    #classify each node 
    for i in range(0, 4):
        object_number = x[i][0]*5 + random_list[i]
        nodes[i][object_number] = 1 
    
    #initialize edge atribute
    edge_atr = torch.zeros(8, 7)

    #left_of, right_of relations always the same
    edge_atr[4][5] = 1
    edge_atr[5][5] = 1
    edge_atr[6][6] = 1
    edge_atr[7][6] = 1

    #look up edge relation in grid and classify the edge features accordingly
    edge_atr[0][matrix_bo_on[x[0][0]][x[1][0]]] = 1
    edge_atr[1][matrix_bo_on[x[2][0]][x[3][0]]] = 1
    edge_atr[2][matrix_on_bo[x[1][0]][x[0][0]]] = 1
    edge_atr[3][matrix_on_bo[x[3][0]][x[2][0]]] = 1

    #combine all data into graph
    data = Data(x=nodes, edge_index=edge_index.t().contiguous(), edge_attr=edge_atr, image=torch.from_numpy(new_im).float() / 255.)


    ##### this code  below can be uncommented, with the above line being commented, to get test the setup using mnist autobase
    ##### note that in the image architecture, 2 lines have to be changed since mnist pictures are 28 x 28 instead of 30 x 30
    ##### x = torch.stack(embs).flatten().view(num_graphs, 1,  30, 30) => x = torch.stack(embs).flatten().view(num_graphs, 1,  28, 28)
    ##### self.lin1 = nn.Linear(900, 250) =>  self.lin1 = nn.Linear(784, 250)

    # transform = transforms.ToTensor()
    # dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    # data = Data(x=nodes, edge_index=edge_index.t().contiguous(), edge_attr=edge_atr, image=random.choice(dataset)[0].float())

    return data

