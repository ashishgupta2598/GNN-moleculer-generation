from rdkit.Chem.rdmolfiles import SmilesMolSupplier
import rdkit 
from typing import Union, Tuple
import numpy as np
import torch
import random
from copy import deepcopy
import rdkit.Chem as Chem
def load_molecule(path):
    with open(path) as smi_file:
        first_line = smi_file.readline()
        has_header = bool("SMILES" in first_line)
        print(has_header)
    smi_file.close()

    molecule_set = SmilesMolSupplier('data\pre-training\gdb13_1K\Train.smi',
                                     sanitize=True,
                                     nameColumn=-1,
                                    )
    return molecule_set

params = {
    "atom_types"   : ["C", "N", "O", "S", "Cl",'I'],
    "n_atom_types":6,
    "formal_charge": [-1, 0, +1],
    "max_n_nodes"  : 13,
    "restart"      : 1,
    "model"        : "GGNN",
    "sample_every" : 2,
    "init_lr"      : 1e-4,
    "epochs"       : 100,
    "batch_size"   : 50,
    "block_size"   : 1000,
    "device"       : 'cuda',
    "n_samples"    : 100,}

from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdchem import BondType
class MolecularGraph:
    def __init__(self,mol):
        self.data_feat=[]
        self.data_ed=[]
        self.n_nodes = self.n_atoms=0
        self.bondtype_to_int = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2,BondType.AROMATIC:3}
        self.n_edge_features = len(self.bondtype_to_int)
        self.mol_to_graph(mol)
        self.int_to_bondtype=dict(map(reversed, self.bondtype_to_int.items()))

        
    def mol_to_graph(self,mol):
        n_edge_features= len(self.bondtype_to_int)
        n_atoms = 0 #len(params['atom_types'])
        #for mol in molecule_set:
        self.n_nodes = self.n_atoms=len(mol.GetAtoms())
        atoms = list(map(mol.GetAtomWithIdx,range(self.n_atoms)))    
        self.node_features = np.array(self.get_feat(mol,atoms))#we have to make features for the atoms and then return them
        self.edge_features=np.zeros([self.n_atoms,self.n_atoms,n_edge_features],dtype=np.int32) #this edge features will be kind of adj matrix, first two as adj and last as features info
        for bond in mol.GetBonds():
            i=bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = self.bondtype_to_int[bond.GetBondType()]
            self.edge_features[i,j,bond_type] = 1
            self.edge_features[j,i,bond_type] = 1
        #return self.node_features,self.edge_features
    def node_remap(self):#edge features of that particuler mol
        self.n_nodes = len(self.edge_features)
        self.atom_ranking = list(range(self.n_nodes))
        random.shuffle(self.atom_ranking)
        node_ordering = self.bfs(self.atom_ranking,self.atom_ranking[0])
        return node_ordering
    def get_feat(self,mol,atoms):
        feat=[]
        def one_of_encoding(m):
            features=[]
            s=str(m.GetSymbol())
            for x in params["atom_types"]:
                #print(int(x==s))
                features.append(int(x==s))
            s=m.GetFormalCharge()
            for x in params["formal_charge"]:
                features.append(int(x==s))
            return features
        for a in atoms:
            feat.append(one_of_encoding(a))
        return feat

    def bfs(self,node_ranking,node_init):
        edge_features = self.edge_features
        self.nodes_visited = [node_init]
        last_nodes_visited = [node_init]
        n_nodes = len(node_ranking)
        # print("node ranking ",node_ranking)
        # print("node init ",node_init)
        while len(self.nodes_visited) < n_nodes:
            neighboring_nodes = []
            for node in last_nodes_visited:
                neighbor_nodes = []
                for bond_type in range(len(self.bondtype_to_int)):
                    neighbor_nodes.extend(list(
                        np.nonzero(edge_features[node, :, bond_type])[0]
                    ))
                new_neighbor_nodes = list(
                    set(neighbor_nodes) - (set(neighbor_nodes) & set(self.nodes_visited))
                )
                node_importance  = [node_ranking[neighbor_node] for
                                      neighbor_node in new_neighbor_nodes]

                # check all neighboring nodes and sort in order of importance
                while sum(node_importance) != -len(node_importance):
                    next_node = node_importance.index(max(node_importance))
                    neighboring_nodes.append(new_neighbor_nodes[next_node])
                    node_importance[next_node] = -1
            self.nodes_visited.extend(set(neighboring_nodes))
            last_nodes_visited = set(neighboring_nodes)
        return self.nodes_visited
    def get_decoding_route_length(self):
        return int((np.array(self.edge_features).sum()//2)+2)
    
    def truncate_graph(self):
        last_atom_idx = self.n_nodes-1
        if self.n_nodes == 1:
            self.node_features[last_atom_idx,:]=0
            self.n_nodes-=1
        else:
            bond_idc = []
            for bond_type in range(self.n_edge_features):
                bond_idc.extend(
                    list(
                        np.nonzero(self.edge_features[:,last_atom_idx,bond_type]) #chossing out that begin node index which connects to last atom and we are doing it for all 4 bond types
                    )
                )
            bond_idc = [i for i in bond_idc if len(i)!=0]
            degree = len(bond_idc)
            print("degree is ",degree," ",bond_idc)
            if degree==1:
                self.node_features[last_atom_idx,:]=0
                self.n_nodes-=1
            else:
                print("this we we have empty nodes at the last")
                bond_idc=bond_idc[-1] #Here we are removing bond_idc...the last bond
            print("here is the value ",bond_idc)
            self.edge_features[bond_idc,last_atom_idx,:]=0 #-->bond_idc gives u index of non zero element--> 
            self.edge_features[last_atom_idx,bond_idc,:]=0#--->selecting last atom,bond-->Set to zero
            #These are set to zero because padding is their so it will be handled in padding
            #2 times as this is undirected graph....
    def get_graph_state(self):
        return self.node_features,self.edge_features
    
    def get_decoding_route_state(self,subgraph_idx:int)->\
                                Tuple[list,np.ndarray]:
        molecular_graph = deepcopy(self)
        if subgraph_idx!=0:
            # for _ in range(1,subgraph_idx):
            #     molecular_graph.truncate_graph()
            decoding_APD=molecular_graph.get_decoding_APD()
            molecular_graph.truncate_graph()
            X,E = molecular_graph.get_graph_state()
        elif subgraph_idx==0:
            decoding_APD = molecular_graph.get_final_decoding_APD()
            X,E = molecular_graph.get_graph_state()
        else:
            raise ValueError("subgraph_idx not a valid value")
        decoding_graph=[X,E]
        return decoding_graph,decoding_APD
    def explicit_graph_setting(self,node_feat,edge_feat):
        self.node_features,self.edge_features = node_feat,edge_feat
        self.n_nodes = self.n_atoms = len(self.node_features)
    def get_nonzero_feature_indices(self, node_idx : int) -> list:
        
        fv_idc = np.cumsum([6,3]).tolist()
        print(self.node_features[node_idx])
        idc    = np.nonzero(self.node_features[node_idx])[0]

        # correct for the concatenation of the different segments of each node
        # feature vector
        print(idc,"node index is as  ",node_idx)
        segment_idc = [idc[0]]
        for idx, value in enumerate(idc[1:]):
            segment_idc.append(value - fv_idc[idx])

        return segment_idc
    def get_decoding_APD(self):
        self.dim_f_add = (13,6,3,1,4)
        self.dim_f_conn = (13,4)
        last_atom_idx = last_node_idx = self.n_atoms-1
        fv_nonzero_idc = self.get_nonzero_feature_indices(node_idx=last_node_idx)

        f_add = np.zeros(self.dim_f_add,dtype=np.int32)
        f_conn=np.zeros(self.dim_f_conn,dtype=np.int32)

        bonded_nodes = []
        for bond_type in range(self.n_edge_features):
            bonded_nodes.extend(list(
                np.nonzero(self.edge_features[:, last_node_idx, bond_type])[0]
            ))
        if bonded_nodes:
            degree = len(bonded_nodes)
            v_idx = bonded_nodes[-1]
            bond_type_forming = int(np.nonzero(self.edge_features[v_idx,last_node_idx,:])[0]
            )
            if degree>1:
                f_conn[v_idx, bond_type_forming] = 1

            else:
                f_add[tuple([v_idx]+fv_nonzero_idc+[bond_type_forming])] = 1
        
        else:
            f_add[tuple([0] + fv_nonzero_idc + [0])] = 1
        print("shapes of various ",f_add.shape," ",f_conn.shape," ")
        apd = np.concatenate((f_add.ravel(), f_conn.ravel(), np.array([0])))
        return apd
    
    def features_to_atom(self, node_idx : int) -> rdkit.Chem.Atom:
        
        # determine the nonzero indices of the feature vector
        feature_vector = self.node_features[node_idx]
        try:  # if `feature_vector` is a `torch.Tensor`
            nonzero_idc = torch.nonzero(feature_vector)
        except TypeError:  # if `feature_vector` is a `numpy.ndarray`
            nonzero_idc = np.nonzero(feature_vector)[0]

        # determine atom symbol
        print("non zero idc ",nonzero_idc)
        atom_idx  = nonzero_idc[0]
        atom_type = params["atom_types"][atom_idx]
        new_atom  = rdkit.Chem.Atom(atom_type)

        # determine formal charge
        fc_idx        = nonzero_idc[1] - params["n_atom_types"]
        formal_charge = params["formal_charge"][fc_idx]

        new_atom.SetFormalCharge(formal_charge)

        return new_atom
    
    def graph_to_mol1(self) -> rdkit.Chem.Mol:
        # create empty editable `rdkit.Chem.Mol` object
        molecule    = rdkit.Chem.RWMol()

        # add atoms to `rdkit.Chem.Mol` and keep track of idx
        node_to_idx = {}

        for v in range(0, self.n_nodes):
            atom_to_add    = self.features_to_atom(node_idx=v)
            molecule_idx   = molecule.AddAtom(atom_to_add)
            node_to_idx[v] = molecule_idx

        # add bonds between adjacent atoms
        for bond_type in range(self.n_edge_features):
            # `self.edge_features[:, :, bond_type]` is an adjacency matrix
            #  for that specific `bond_type`
            for bond_idx1, row in enumerate(
                self.edge_features[:self.n_nodes, :self.n_nodes, bond_type]
                ):
                # traverse only half adjacency matrix to not duplicate bonds
                for bond_idx2 in range(bond_idx1):
                    bond = row[bond_idx2]
                    if bond:  # if `bond_idx1` and `bond_idx2` are bonded
                        try:  # try adding the bond to `rdkit.Chem.Mol` object
                            molecule.AddBond(
                                node_to_idx[bond_idx1],
                                node_to_idx[bond_idx2],
                                self.int_to_bondtype[bond_type]
                            )
                        except (TypeError, RuntimeError, AttributeError):
                            # errors occur if the above `AddBond()` action tries
                            # to add multiple bonds to a node pair (should not
                            # happen, but kept here as a safety)
                            raise ValueError("MolecularGraphError: Multiple "
                                             "edges connecting a single pair "
                                             "of nodes in graph.")

        try:  # convert from `rdkit.Chem.RWMol` to Mol object
            molecule.GetMol()
        except AttributeError:  # raised if molecules is `None`
            pass
        return molecule
    def neighs_find(self,d_values):#dvalues should be 2d array.....
        d = np.sum(d_values,axis=1)
        y = np.nonzero(d)[0]
        return y#zip(list(x),list(y))
    def picking(self):
        edge_feat = self.edge_features
        node_feat = self.node_features
        apd_nodes = np.zeros((13,6,3,1,4))
        edges = np.zeros((13,4))
        node_ranking = np.arange(self.n_nodes)
        node_picks = node_ranking #check these should not contains some unneccery removed nodes
        
        for i in range(0,1):#len(node_picks)):#picking nodes 
            node = node_picks[i]
            neighs = self.neighs_find(edge_feat[node])
            print("neighs are ",neighs)
            if len(neighs)>1:
                #put code that does remove bonds
                #finding suitable bond
                #lets pick at least any fcking bond
                print("we are deleting an edge")
                random.shuffle(neighs)
                j_node = neighs[0]
                print("choosing one neighs ",j_node)
                finding_bond_type = np.nonzero(edge_feat[node][j_node])#it will give u location at where its
                #This finding bond type check at last one should be non zero  
                edges[j_node][finding_bond_type] = 1 #bond type filling is done
                #getting apd--->
                #Node deletion and edge_deletion(just make that bond to zero)------------
                #node_feat[:node] = 0
                edge_feat[node][j_node]=0
                edge_feat[j_node][node]=0
                print("nodes that we are checking out ",node,neighs[0])

            elif len(neighs)==1:
                print("we are deleting an node")
                finding_bond_type = np.nonzero(edge_feat[node][neighs[0]])
                vals = np.nonzero(node_feat[node])[0]#only one value should come
                print("atom and formal charges are such as ",vals)
                if len(vals)==0:
                    raise Exception("atom_type and formal charge is empty!!!!Alert situation")
                atom_type,formal_charge = vals
                formal_charge1 =8-formal_charge#8 becz we are counting from 0  
                apd_nodes[node][atom_type][formal_charge1][0][finding_bond_type] = 1
                print("nodes that we are checking out last ",node,neighs[0])
                edge_feat = np.delete(edge_feat,(node),axis=0) #node jinse connected hy
                edge_feat = np.delete(edge_feat,(node),axis=1) #jo node se connected se
                #edge_feat[node][neighs[0]]=0
                # edge_feat[neighs[0]][node]=0#becz we have alreay deleted this part in y axis...it will auda dena
                node_feat = np.delete(node_feat,(node),axis=0)
                self.n_nodes = self.n_nodes-1
            else:
                return None,None

            #Nodes deletion left
            self.node_features ,self.edge_features = node_feat,edge_feat
            node_feat,edge_feat = self.pad()
            apdtt = [apd_nodes,edges]
            graph = [node_feat,edge_feat]
            return graph, apdtt

    def pad(self):
        node_features_padded_f = np.zeros((13,9))
        edge_features_padded_f = np.zeros((13,13,4))
        n = self.n_nodes
        node_features_padded_f[:n,:] = self.node_features
        edge_features_padded_f[:n,:n,:] = self.edge_features

        return node_features_padded_f,edge_features_padded_f



"""
print(molecule.edge_features.shape,molecule.node_features.shape, molecule.n_nodes)
# molecule.pad()

#1 molecule lets break into some n-1 pieces and store them into a numpy array for now due to different sizes..then we will chose
"""
#code for defaulting breaking down 
path = "data\pre-training\gdb13_1K\Train.smi"
molecule_set = load_molecule(path)
print(molecule_set)
dict1 = {}
for j,mol in enumerate(molecule_set):
    molecule = MolecularGraph(mol)
    routes = molecule.get_decoding_route_length()
    l = j*10
    for i in range(routes):
        if i>8:
            
            continue
        gh,adp = molecule.picking()
        if gh==None:
            i-=1
            continue
        dict1[l+i]=[gh,adp]
        print(i)
import pickle
a_file = open("data_1000_train.pkl", "wb")
pickle.dump(dict1, a_file)