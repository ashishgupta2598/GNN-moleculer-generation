import torch
from collections import namedtuple
from graphinvent import gnn


class MLP(torch.nn.Module):
    def __init__(self, in_features : int, hidden_layer_sizes : list, out_features : int,
                 dropout_p : float) -> None:
        super().__init__()

        activation_function = torch.nn.SELU
        # create list of all layer feature sizes
        fs = [in_features, *hidden_layer_sizes, out_features]
        # create list of linear_blocks
        layers = [self._linear_block(in_f, out_f,
                                     activation_function,
                                     dropout_p)
                  for in_f, out_f in zip(fs, fs[1:])]
        # concatenate modules in all sequentials in layers list
        layers = [module for sq in layers for module in sq.children()]

        # add modules to sequential container
        self.seq = torch.nn.Sequential(*layers)

    def _linear_block(self, in_f : int, out_f : int, activation : torch.nn.Module,
                      dropout_p : float) -> torch.nn.Sequential:
        
        # bias must be used in most MLPs in our models to learn from empty graphs
        linear = torch.nn.Linear(in_f, out_f, bias=True)
        torch.nn.init.xavier_uniform_(linear.weight)
        return torch.nn.Sequential(linear, activation(), torch.nn.AlphaDropout(dropout_p))

    def forward(self, layers_input : torch.nn.Sequential) -> torch.nn.Sequential:
        return self.seq(layers_input)

class GlobalReadout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = MLP(in_features=constants.message_size,
                  hidden_layer_sizes=[constants.mlp1_hidden_dim]*constants.mlp1_depth,
                  out_features=constants.message_size,
                  dropout_p=0.0)
        self.mlp2 = MLP(in_features=constants.message_size,
                  hidden_layer_sizes=[constants.mlp2_hidden_dim]*constants.mlp2_depth,
                  out_features=constants.message_size,
                  dropout_p=0.0)
        
        self.mlp3 = MLP(in_features=constants.message_size,
                  hidden_layer_sizes=[constants.mlp1_hidden_dim]*constants.mlp1_depth,
                  out_features=constants.message_size,
                  dropout_p=0.0)
        self.mlp4 = MLP(in_features=2*constants.message_size,
                  hidden_layer_sizes=[constants.mlp2_hidden_dim]*constants.mlp2_depth,
                  out_features=constants.message_size,
                  dropout_p=0.0)
        self.mlpt = MLP(in_features=constants.message_size,
                  hidden_layer_sizes=[constants.mlp1_hidden_dim]*constants.mlp1_depth,
                  out_features=constants.message_size,
                  dropout_p=0.0)
        self.final_l = MLP(in_features=2900,#23*100 #(14+14+1)100
                  hidden_layer_sizes=[500]*1,
                  out_features=989,
                  dropout_p=0.0)
          
    def forward(self,features):
        g= torch.sum(features,dim=1)
        g = g.view(1,1,100)
        #g = torch.broadcast_to(g, (1,10, 100))
        print("api is ",g.shape)
        fadd1 = self.mlp1(features)
        fconn1 = self.mlp2(features)  

        print("dims fadd ip",fconn1.shape,g.shape,torch.cat([fadd1,g],dim=1).shape)
        fadd = self.mlp3(torch.cat([fadd1,g],dim=1)).unsqueeze(dim=1)
        fconn = self.mlp3(torch.cat([fconn1,g],dim=1)).unsqueeze(dim=1)

        fterm = self.mlpt(g)
        print("global readout shapes ",fadd.shape,fconn.shape,fterm.shape)
        cat = torch.cat((fadd.squeeze(dim=1), fconn.squeeze(dim=1), fterm), dim=1)
        cat = torch.flatten(cat)
        cat = self.final_l(cat)
        print("final shape ",cat.shape)

        return cat
        #apd = self.Softmax()....from original code its removed



class SummationMPNN(torch.nn.Module):
    """
    Abstract `SummationMPNN` class. Specific models using this class are
    defined in `mpnn.py`; these are MNN, S2V, and GGNN.
    """
    def __init__(self, constants : namedtuple):

        super().__init__()

        self.hidden_node_features = constants.hidden_node_features
        self.edge_features        = constants.n_edge_features
        self.message_size         = constants.message_size
        self.message_passes       = constants.message_passes
        self.constants            = constants

    def forward(self, nodes : torch.Tensor, edges : torch.Tensor) -> None:
        adjacency = torch.sum(edges, dim=3)

        # **note: "idc" == "indices", "nghb{s}" == "neighbour(s)"
        (edge_batch_batch_idc,
         edge_batch_node_idc,
         edge_batch_nghb_idc) = adjacency.nonzero(as_tuple=True)
        print("sizes are edge_batch_node_idc ",edge_batch_node_idc)

        (node_batch_batch_idc, node_batch_node_idc) = adjacency.sum(-1).nonzero(as_tuple=True)

        same_batch = node_batch_batch_idc.view(-1, 1) == edge_batch_batch_idc
        same_node  = node_batch_node_idc.view(-1, 1) == edge_batch_node_idc

        # element ij of `message_summation_matrix` is 1 if `edge_batch_edges[j]`
        # is connected with `node_batch_nodes[i]`, else 0
        message_summation_matrix = (same_batch * same_node).float()

        edge_batch_edges = edges[edge_batch_batch_idc, edge_batch_node_idc, edge_batch_nghb_idc, :]

        # pad up the hidden nodes
        hidden_nodes = torch.zeros(nodes.shape[0],
                                   nodes.shape[1],
                                   self.hidden_node_features,
                                   device=self.constants.device)
        hidden_nodes[:nodes.shape[0], :nodes.shape[1], :nodes.shape[2]] = nodes.clone()
        node_batch_nodes = hidden_nodes[node_batch_batch_idc, node_batch_node_idc, :]
        

        for _ in range(self.message_passes):
            edge_batch_nodes = hidden_nodes[edge_batch_batch_idc, edge_batch_node_idc, :]

            edge_batch_nghbs = hidden_nodes[edge_batch_batch_idc, edge_batch_nghb_idc, :]

            print("hello ji ",edge_batch_nghbs.shape,edge_batch_nodes.shape,hidden_nodes.shape)

            message_terms = self.message_terms(edge_batch_nodes,
                                                  edge_batch_nghbs,
                                                  edge_batch_edges)

            if len(message_terms.size()) == 1:  # if a single graph in batch
                message_terms = message_terms.unsqueeze(0)

            # the summation in eq. 1 of the NMPQC paper happens here
            messages = torch.matmul(message_summation_matrix, message_terms)

            node_batch_nodes = self.update(node_batch_nodes, messages)
            hidden_nodes[node_batch_batch_idc, node_batch_node_idc, :] = node_batch_nodes.clone() #updated the hidden states

        node_mask = adjacency.sum(-1) != 0
        output    = self.readout(hidden_nodes, nodes, node_mask)

        return output    
# from graphinvent.gnn.modules import GlobalReadout 
class MNN(SummationMPNN):
    def __init__(self,constants) -> None:
        super().__init__(constants)
        self.constants       = constants
        print(self.constants.message_size,self.constants.hidden_node_features,4)
        message_weights      = torch.Tensor(self.constants.message_size,
                                            self.constants.hidden_node_features,
                                            4)#edge features
        print(message_weights.shape)
        if False:#"cuda" == "cuda":
            message_weights = message_weights.to("cuda", non_blocking=True)
        
    
        self.message_weights = torch.nn.Parameter(message_weights)

        self.gru             = torch.nn.GRUCell(
            input_size=self.constants.message_size,
            hidden_size=self.constants.hidden_node_features,
            bias=True
        )
        
        self.APDReadout = GlobalReadout()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        import math
        stdev = 1.0 / math.sqrt(self.message_weights.size(1))
        self.message_weights.data.uniform_(-stdev, stdev)

    def message_terms(self, nodes : torch.Tensor, node_neighbours : torch.Tensor,
                        edges : torch.Tensor) -> torch.Tensor:
        
        edges_view            = edges.view(-1, 1, 1, self.constants.n_edge_features)
        #print("edges ",edges_view.shape)
        weights_for_each_edge = (edges_view * self.message_weights.unsqueeze(0)).sum(3)
        return torch.matmul(weights_for_each_edge,
                            node_neighbours.unsqueeze(-1)).squeeze()
    #torch.broadcast_to(x, (3, 3))
    def update(self, nodes : torch.Tensor, messages : torch.Tensor) -> torch.Tensor:
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes : torch.Tensor, input_nodes : torch.Tensor,
                node_mask : torch.Tensor) -> torch.Tensor:
        # graph_embeddings = torch.sum(hidden_nodes, dim=1)
        print("input to network ",hidden_nodes.shape)
        output           = self.APDReadout(hidden_nodes)
        return output




hyperparameters = {
        "mlp1_depth"          : 4,
        "mlp1_dropout_p"      : 0.0,
        "mlp1_hidden_dim"     : 100,
        "mlp2_depth"          : 4,
        "mlp2_dropout_p"      : 0.0,
        "mlp2_hidden_dim"     : 100,
        "hidden_node_features": 100,
        "message_passes"      : 3,
        "message_size"        : 100,
        "n_edge_features"     :4
    }
import json
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
constants = dotdict(hyperparameters)

network = MNN(constants)
print(network)