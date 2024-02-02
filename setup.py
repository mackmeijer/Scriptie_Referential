import argparse
from torch_geometric.nn import GATv2Conv
import torch
import torch.nn.functional as F
import egg.core as core
import torch.nn as nn
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter
from dataloaders import create_data_loaders
from typing import Any, List, Optional, Sequence, Union
from torchvision import models
from PIL import Image
from torchvision.utils import save_image
import torch.utils.data
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.data.on_disk_dataset import OnDiskDataset
from torchvision import datasets, transforms
import json
from egg.core.callbacks import ConsoleLogger
import matplotlib.pyplot as plt
import os 

class Collater:
    def __init__(
        self,
        game_size: int,  # the number of graphs for a game
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None
    ):
        self.game_size = game_size
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, BaseData):
            batch = batch[:((len(batch) // self.game_size) * self.game_size)]  # we throw away the last batch_size % game_size
            batch = Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
            # we return a tuple (sender_input, labels, receiver_input, aux_input)
            # we use aux_input to store minibatch of graphs

            # we concat the node object classes together for the topsim input
            topsim_input = batch.x.view(len(batch) // self.game_size, 20 * 4 * self.game_size)

            return (
                 topsim_input,  # we use the input to store the topsim input
                torch.zeros(len(batch) // self.game_size).long(),  # the target is aways the first graph among game_size graphs
                None,  # we don't care about receiver_input
                batch  # this is a compact data for batch_size graphs 
            )

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")

    def collate_fn(self, batch: List[Any]) -> Any:
        if isinstance(self.dataset, OnDiskDataset):
            return self(self.dataset.multi_get(batch))
        return self(batch)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        game_size: int,  # the number of graphs for a game
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        self.game_size = game_size
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.collator = Collater(game_size, dataset, follow_batch, exclude_keys)

        if isinstance(dataset, OnDiskDataset):
            dataset = range(len(dataset))

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collator.collate_fn,
            **kwargs,
        )


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, n_hidden, game_size, input_type):
        super().__init__()

        self.input_type = input_type 

        if input_type == "both": 
            if n_hidden % 2 != 0:
                raise("should be even number")
            else:
               n_hidden = n_hidden // 2 #split n_hidden in 2 to make sure the ending output is still size n_hidden

        self.head_size = 2
        self.game_size = game_size
        feature_size = 1600

        self.gr_conv1 = GATv2Conv(num_node_features, n_hidden, edge_dim = 7,  num_heads = self.head_size)
        self.gr_conv2 = GATv2Conv(n_hidden, n_hidden, edge_dim = 7, num_heads = self.head_size)
        self.emb_lin1 = nn.Linear(feature_size, 50, bias=False)

        self.lin1 = nn.Linear(900, n_hidden)
        
    def forward(self, data):
        x, edge_index, edge_feat, image = data.x, data.edge_index, data.edge_attr, data.image  # check number of graphs via: data.num_graphs

        if self.input_type != "image":            
            x = self.gr_conv1(x, edge_index, edge_feat)
            x = F.relu(x)
            x = scatter(x, data.batch, dim=0, reduce='mean')  # size: data.num_graph,  n_hidden
            
        if self.input_type != "graph":
            #adjust tensor size for the network:
            emb = self.return_embeddings(image, data.num_graphs) # num_graphs, 1, 1, embedding_size

            a = emb.squeeze(dim = 1).flatten(1, 2)

            a = self.lin1(a)   #size: data.num_graphs, n_hidden
            a = F.leaky_relu(a)



        if self.input_type == "both":

            h = torch.cat((a, x), dim = 1) #concat graph and image representation together; size: data.num_graphs, n_hidden

            return h 
    
        elif self.input_type == "image":
            return a

        elif self.input_type == "graph":
            return x
        
        else:
            raise("invalid input option: choose from {both, graph, image}")

    def return_embeddings(self, x, num_graphs):

        embs = []
        h = x.chunk(num_graphs)

        for i in range(num_graphs):
            b = h[i]
            embs.append(b)

        x = torch.stack(embs).flatten().view(num_graphs, 1,  30, 30)
        return x
   
class Sender(nn.Module):
    def __init__(self, num_node_features, n_hidden, game_size, input_type):
        super().__init__()
        self.game_size = game_size
        self.gcn = GCN(num_node_features, n_hidden, game_size, input_type)
        self.fc1 = nn.Linear(n_hidden, n_hidden)

    def forward(self, x, _aux_input):
        # _aux_input is a minibatch of n_games x game_size graphs
        data = _aux_input
        assert data.num_graphs % self.game_size == 0
        x = self.gcn(data)[::self.game_size]  # we just need the target graph, hence we take only the first graph of every game_size graphs
        return self.fc1(x)  # size: n_games * n_hidden  (note: n_games = batch_size // game_size)


class Receiver(nn.Module):
    def __init__(self, num_node_features, n_hidden, game_size, input_type):
        super().__init__()
        self.game_size = game_size

        self.gcn = GCN(num_node_features, n_hidden, game_size, input_type)

    def forward(self, x, _input, _aux_input):
        # x is the tensor of shape n_games * n_hidden -- each row is an embedding decoded from the message sent by the sender
        cands = self.gcn(_aux_input)  # graph embeddings for all batch_size graphs; size: batch_size * n_hidden
        cands = cands.view(cands.shape[0] // self.game_size, self.game_size, -1)  # size: n_games * game_size * n_hidden 
        dots = torch.matmul(cands, torch.unsqueeze(x, dim=-1))  # size: n_games * game_size * 1
        return dots.squeeze() # size: n_games * game_size: each row is a list of scores for a game (each score tells how good the corresponding candidate is) 


def get_params(params):
    parser = argparse.ArgumentParser()

    # arguments concerning the training method
    parser.add_argument(
        "--mode",
        type=str,
        default="gs",
        help="Selects whether Reinforce or Gumbel-Softmax relaxation is used for training {rf, gs} (default: rf)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender, only relevant in Gumbel-Softmax (gs) mode (default: 1.0)",
    )
    # arguments concerning the agent architectures
    parser.add_argument(
        "--sender_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)",
    )

    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=8,
        help="Size of the hidden layer of Sender; should be an even number(default: 10)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=8,
        help="Size of the hidden layer of Receiver; should be an even number (default: 10)",
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=5,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=5,
        help="Output dimensionality of the layer that embeds the message symbols for Receiver (default: 10)",
    )
    # arguments controlling the script output
    parser.add_argument(
        "--print_validation_events",
        default=False,
        action="store_true",
        help="If this flag is passed, at the end of training the script prints the input validation data, the corresponding messages produced by the Sender, and the output probabilities produced by the Receiver (default: do not print)",
    )
    parser.add_argument(
        "--game_size",
        type=int,
        default=3,
        help="The number of graphs in a game (including a target and distractors) (default: 4)",
    )
    parser.add_argument(
        "--batch_per_epoch", 
        type = int,
        default = 1

    )
    parser.add_argument(
        "--input_type_sender", 
        type = str,
        default = "both",
        help="type of input used for the sender agent in the game {graph, image, both} (default: both)",
    )
    
    parser.add_argument(
        "--input_type_receiver", 
        type = str,
        default = "both",
        help="type of input used for the receiver agent in the game {graph, image, both} (default: both)",
    )
    

    args = core.init(parser, params)
    return args

class ResultsCollector(ConsoleLogger):
    def __init__(self , dump: list, results: list, print_to_console: bool, print_train_loss=True, as_json=True):
        super().__init__(as_json=as_json, print_train_loss=print_train_loss)
        self.results = results
        self.print_to_console = print_to_console
        self.dump = dump
    # adapted from egg.core.callbacks.ConsoleLogger
        
    def aggregate_print(self, loss: float, logs, mode: str, epoch: int):
        dump = dict(loss=loss)
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        dump.update(aggregated_metrics)
        dump.update(dict(mode=mode, epoch=epoch))

        results = json.dumps(dump)
        self.dump.append(dump['acc'])
        self.results.append(results)

        if self.print_to_console:
            output_message = ", ".join(sorted([f"{k}={v}" for k, v in dump.items()]))
            output_message = f"{mode}: epoch {epoch}, loss {loss}, " + output_message

def main(params):
    opts = get_params(params)
    print(opts, flush=True)
    game_size = opts.game_size

    # we care about the communication success: the accuracy that the receiver can distinguish the target from distractors
    def loss(
        _sender_input,
        _message,
        _receiver_input,
        receiver_output,
        labels,
        _aux_input,
    ):

        acc = (receiver_output.argmax(dim=1) == labels).detach().float()
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        return loss, {"acc": acc}

    # we create dataset and dataloader
    dataset1, dataset2 = create_data_loaders(1)
    train_loader = DataLoader(game_size, dataset1, batch_size=opts.batch_size,  shuffle=True)
    val_loader = DataLoader(game_size, dataset2, batch_size=opts.batch_size, shuffle=True)

    # we create the two agents
    receiver = Receiver(20, n_hidden=opts.receiver_hidden, game_size=game_size, input_type = opts.input_type_receiver) 
    sender = Sender(20, n_hidden=opts.sender_hidden, game_size=game_size, input_type = opts.input_type_sender)

    sender = core.RnnSenderGS(sender, vocab_size=opts.vocab_size, embed_dim=10, hidden_size=opts.sender_hidden, max_len=opts.max_len, temperature=1.0, cell='rnn')
    receiver = core.RnnReceiverGS(receiver, vocab_size=opts.vocab_size, embed_dim=10, hidden_size=opts.receiver_hidden, cell='rnn')
    game = core.SenderReceiverRnnGS(sender, receiver, loss)

    callbacks = []
    results = []
    dump = []
  
    optimizer = core.build_optimizer(game.parameters())
    
    topographic_similarity = core.TopographicSimilarity(
        sender_input_distance_fn="cosine", 
        message_distance_fn="euclidean",
        compute_topsim_test_set=True,
        compute_topsim_train_set=False, 
        is_gumbel=True
    )

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=val_loader,
        callbacks=callbacks
        + [core.ConsoleLogger(print_train_loss=True, as_json=True), topographic_similarity, ResultsCollector(results = results,dump = dump, print_to_console=True)],
    )

    trainer.train(n_epochs=opts.n_epochs)

    core.close()

    ### used to generate plots ###

    # plt.plot(range(len(dump) // 2), dump[0::2], label = "training")
    # plt.plot(range(len(dump) // 2), dump[1::2], label = "testing") 
    # plt.axhline(y = 1 / opts.game_size, color = 'b', linestyle = '-.',  label = "chance level") 
    # plt.xlabel('x - axis')
    # plt.ylabel('')
    # plt.title('accuracy')
    # plt.ylim(0, 1)
    # plt.legend(loc='best')
    # plt.savefig('plot'+str(opts.input_type) + str(opts.game_size) + 'png')
    # plt.clf()

    ### used to generate plots ###

main(["--vocab_size", "30", "--max_len", "3","--n_epochs", "100", "--game_size", "2", "--lr", "1e-3", "--batch_size", "10",  "--sender_hidden", "20", "--receiver_hidden", "20",  "--input_type_sender", "image", "--input_type_receiver", "graph"])

