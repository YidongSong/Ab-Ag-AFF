# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,AutoModel,RoFormerModel
from torch_geometric.nn import TransformerConv
from torch_scatter import scatter_mean,scatter_add
import numpy as np



def _positional_embeddings(edge_index, num_embeddings=16):
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=edge_index.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        PE = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return PE

def _get_angle(X, eps=1e-7):
    # psi, omega, phi
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    D = F.pad(D, [1, 2]) # This scheme will remove phi[0], psi[-1], omega[-1]
    D = torch.reshape(D, [-1, 3])
    dihedral = torch.cat([torch.cos(D), torch.sin(D)], 1)

    # alpha, beta, gamma
    cosD = (u_2 * u_1).sum(-1) # alpha_{i}, gamma_{i}, beta_{i+1}
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.acos(cosD)
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    bond_angles = torch.cat((torch.cos(D), torch.sin(D)), 1)

    node_angles = torch.cat((dihedral, bond_angles), 1)
    return node_angles # dim = 12

def _rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def _get_distance(X, edge_index):
    atom_N = X[:,0]  # [L, 3]
    atom_Ca = X[:,1]
    atom_C = X[:,2]
    atom_O = X[:,3]
    atom_R = X[:,4]
    node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C', 'R-N', 'R-Ca', "R-C", 'R-O']
    node_dist = []
    for pair in node_list:
        atom1, atom2 = pair.split('-')
        E_vectors = vars()['atom_' + atom1] - vars()['atom_' + atom2]
        rbf = _rbf(E_vectors.norm(dim=-1))
        node_dist.append(rbf)
    node_dist = torch.cat(node_dist, dim=-1) # dim = [N, 10 * 16]

    atom_list = ["N", "Ca", "C", "O", "R"]
    edge_dist = []
    for atom1 in atom_list:
        for atom2 in atom_list:
            E_vectors = vars()['atom_' + atom1][edge_index[0]] - vars()['atom_' + atom2][edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1))
            edge_dist.append(rbf)
    edge_dist = torch.cat(edge_dist, dim=-1) # dim = [E, 25 * 16]

    return node_dist, edge_dist

def _quaternions(R):
    """ Convert a batch of 3D rotations [R] to quaternions [Q]
        R [N,3,3]
        Q [N,4]
    """
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
          Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:,i,j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)

    return Q

def _get_direction_orientation(X, edge_index): # N, CA, C, O, R
    X_N = X[:,0]  # [L, 3]
    X_Ca = X[:,1]
    X_C = X[:,2]
    u = F.normalize(X_Ca - X_N, dim=-1)
    v = F.normalize(X_C - X_Ca, dim=-1)
    b = F.normalize(u - v, dim=-1)
    n = F.normalize(torch.cross(u, v), dim=-1)
    local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1) # [L, 3, 3] (3 column vectors)

    node_j, node_i = edge_index
    t = F.normalize(X[:, [0,2,3,4]] - X_Ca.unsqueeze(1), dim=-1) # [L, 4, 3]
    try:
        node_direction = torch.matmul(t, local_frame).reshape(t.shape[0], -1) # [L, 4 * 3]
    except Exception as ex:
        print(t.size())
        print(local_frame.size())
        print('except')
        print(ex)

    t = F.normalize(X[node_j] - X_Ca[node_i].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ji = torch.matmul(t, local_frame[node_i]).reshape(t.shape[0], -1) # [E, 5 * 3]
    t = F.normalize(X[node_i] - X_Ca[node_j].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ij = torch.matmul(t, local_frame[node_j]).reshape(t.shape[0], -1) # [E, 5 * 3]
    edge_direction = torch.cat([edge_direction_ji, edge_direction_ij], dim = -1) # [E, 2 * 5 * 3]

    r = torch.matmul(local_frame[node_i].transpose(-1,-2), local_frame[node_j]) # [E, 3, 3]
    edge_orientation = _quaternions(r) # [E, 4]

    return node_direction, edge_direction, edge_orientation

def get_geo_feat(X, edge_index):
    pos_embeddings = _positional_embeddings(edge_index)
    node_angles = _get_angle(X)
    node_dist, edge_dist = _get_distance(X, edge_index)
    node_direction, edge_direction, edge_orientation = _get_direction_orientation(X, edge_index)

    geo_node_feat = torch.cat([node_angles, node_dist, node_direction], dim=-1)
    geo_edge_feat = torch.cat([pos_embeddings, edge_orientation, edge_dist, edge_direction], dim=-1)

    return geo_node_feat, geo_edge_feat

def padding_ver1(x, batch_id, feature_dim):
        batch_size = max(batch_id) + 1
        max_len= max(torch.unique(batch_id,return_counts=True)[1])
        batch_data = torch.zeros([batch_size,max_len,feature_dim])
        mask = torch.zeros([batch_size,max_len])
        len_0 = 0
        len_1 = 0
        for i in range(batch_size):
            len_1 = len_0 + torch.unique(batch_id,return_counts=True)[1][i]
            batch_data[i][:torch.unique(batch_id,return_counts=True)[1][i]] = x[len_0:len_1]
            mask[i][:torch.unique(batch_id,return_counts=True)[1][i]] = 1
            len_0 += torch.unique(batch_id,return_counts=True)[1][i]
        return batch_data, mask



class Context(nn.Module):
    def __init__(self, num_hidden):
        super(Context, self).__init__()

        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.Sigmoid()
                                )

    def forward(self, h_V, batch_id):
        # c_V = scatter_add(h_V, batch_id, dim=0)
        c_V = scatter_mean(h_V, batch_id, dim=0)
        h_V = h_V * self.V_MLP_g(c_V[batch_id])
        return h_V

class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, dropout=0.2):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(3*num_hidden, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, edge_index, h_E):
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W12(self.act(self.W11(h_EV)))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E
    
class GNNLayer(nn.Module):
    def __init__(self, num_hidden, dropout=0.2, num_heads=4):
        super(GNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])

        self.attention = TransformerConv(in_channels=num_hidden, out_channels=int(num_hidden / num_heads), heads=num_heads, dropout = dropout, edge_dim = num_hidden, root_weight=False)
        self.PositionWiseFeedForward = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden)

    def forward(self, h_V, edge_index, h_E, batch_id):
        dh = self.attention(h_V, edge_index, h_E)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.PositionWiseFeedForward(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        # update edge
        h_E = self.edge_update(h_V, edge_index, h_E)

        # context node update
        h_V = self.context(h_V, batch_id)

        return h_V, h_E
    
class Graph_encoder(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim,
                 seq_in=False, num_layers=4, drop_rate=0.2):
        super(Graph_encoder, self).__init__()

        self.seq_in = seq_in
        if self.seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim += 20
        
        self.node_embedding = nn.Linear(node_in_dim, hidden_dim, bias=True)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim, bias=True)
        self.norm_nodes = nn.BatchNorm1d(hidden_dim)
        self.norm_edges = nn.BatchNorm1d(hidden_dim)
        
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.layers = nn.ModuleList(
                GNNLayer(num_hidden=hidden_dim, dropout=drop_rate, num_heads=4)
            for _ in range(num_layers))


    def forward(self, h_V, edge_index, h_E, seq, batch_id):
        if self.seq_in and seq is not None:
            seq = self.W_s(seq)
            h_V = torch.cat([h_V, seq], dim=-1)
        
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_E = self.W_e(self.norm_edges(self.edge_embedding(h_E)))

        for layer in self.layers:
            h_V, h_E = layer(h_V, edge_index, h_E, batch_id)
        
        return h_V
    
class Ab_Ag_AFF(nn.Module):
    def __init__(self, heavy_dir,light_dir, antigen_dir, emb_dim=256):
        super().__init__()
        self.HeavyModel = AutoModel.from_pretrained(heavy_dir, output_hidden_states=True, return_dict=True)
        self.LightModel = AutoModel.from_pretrained(light_dir, output_hidden_states=True, return_dict=True)
        self.AntigenModel = AutoModel.from_pretrained(antigen_dir, output_hidden_states=True, return_dict=True,cache_dir = "./esm2/ESM_models")


        self.cnn1 = MF_CNN(in_channel= 256)
        self.cnn2 = MF_CNN(in_channel = 256)
        self.cnn3 = MF_CNN(in_channel = 700,hidden_size=76)

        self.cnn1_esm2 = MF_CNN(in_channel= 256, hidden_size=76)
        self.cnn2_esm2 = MF_CNN(in_channel= 256, hidden_size=76)

        self.binding_predict = nn.Sequential(
            nn.Linear(in_features=128 * 3, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=1)
        )

    def forward(self, heavy, light, antigen, device):
        heavy_esm2 = {}
        light_esm2 = {}
        heavy_esm2['input_ids'] = heavy['input_ids']
        heavy_esm2['attention_mask'] = heavy['attention_mask']
        light_esm2['input_ids'] = light['input_ids']
        light_esm2['attention_mask'] = light['attention_mask']
        heavy_encoded = self.AntigenModel(**heavy_esm2).last_hidden_state 
        light_encoded = self.AntigenModel(**light_esm2).last_hidden_state 
        antigen_encoded = self.AntigenModel(**antigen).last_hidden_state 
        heavy_cls = self.cnn1_esm2(heavy_encoded)
        light_cls = self.cnn2_esm2(light_encoded)
        antigen_cls = self.cnn3(antigen_encoded)

        concated_encoded = torch.concat((heavy_cls,light_cls,antigen_cls) , dim = 1)

        output = self.binding_predict(concated_encoded)

        return output

class MF_CNN(nn.Module):
    def __init__(self, in_channel=118,emb_size = 20,hidden_size = 92):#189):
        super(MF_CNN, self).__init__()
        
        # self.emb = nn.Embedding(emb_size,128)  # 20*128
        self.conv1 = cnn_liu(in_channel = in_channel,hidden_channel = 64)   # 118*64
        self.conv2 = cnn_liu(in_channel = 64,hidden_channel = 32) # 64*32

        self.conv3 = cnn_liu(in_channel = 32,hidden_channel = 32)

        self.fc1 = nn.Linear(32*hidden_size , 128) # 32*29*512
        self.fc2 = nn.Linear(128 , 128)

        self.fc3 = nn.Linear(128 , 128)

    def forward(self, x):
        #x = x
        # x = self.emb(x)
        
        x = self.conv1(x)
        
        x = self.conv2(x)

        x = self.conv3(x)
        
        x = x.view(x.shape[0] ,-1)
        
        x = nn.ReLU()(self.fc1(x))
        sk = x
        x = self.fc2(x)

        x = self.fc3(x)
        return x +sk

class cnn_liu(nn.Module):
    def __init__(self, in_channel=2, hidden_channel=2, out_channel=2):
        super(cnn_liu, self).__init__()
        
        self.cnn = nn.Conv1d(in_channel , hidden_channel , kernel_size = 5 , stride = 1) # bs * 64*60
        self.max_pool = nn.MaxPool1d(kernel_size = 2 , stride=2)# bs * 32*30
                               
        self.relu = nn.ReLU()
    
    def forward(self, x):
        
        #x = self.emb(x)
        x = self.cnn(x)
        x = self.max_pool(x)
        x = self.relu(x)
        return x