import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv, GINEConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
batch_size = 128
learning_rate = 1e-3
num_node_features = 73 # one-hot(20) + position encoding(20) + esm-per_residue_embedding(33)
num_classes = 11


class GraphDataset(Dataset):

    def __init__(self, graph_list):
        '''
        graph_list is a numpy array of tuples defined as [(node_features, edge_index, class), (node_features, edge_index, class)]
        '''
        super().__init__()
        self.graph_list = graph_list 
        
    def len(self):
        return self.graph_list.shape[0]

    def get(self, idx):
       
        node_features_np, edge_index_np, labels_np = self.graph_list[idx]
        #node_features_np = node_features_np[:, 0:-2]
        
        labels_np = labels_np.reshape(-1, num_classes)
        #phanns_feats_np = phanns_feats_np.reshape(-1, phanns_feats_np.shape[0])

        t_node_features = torch.tensor(node_features_np, dtype=torch.float32)
        t_edge_index = torch.tensor(edge_index_np, dtype=torch.int64)
        #t_edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)
        #t_phanns_feats = torch.tensor(phanns_feats_np, dtype=torch.float32)
        t_labels = torch.tensor(labels_np, dtype=torch.float32)

        data = Data(x=t_node_features, edge_index=t_edge_index, y=t_labels)
        return data


class GINnet(nn.Module):

    def __init__(self, dim_h):
        super(GINnet, self).__init__()
        
        # GIN definition
        self.conv1 = GINConv(
            Sequential(Linear(num_node_features, dim_h), BatchNorm1d(dim_h), ReLU(), 
                       Linear(dim_h, dim_h), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(), 
                       Linear(dim_h, dim_h), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),  
                       Linear(dim_h, dim_h), ReLU()))
        
        '''
        self.conv4 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),  
                       Linear(dim_h, dim_h), ReLU()))
        '''

        # classifier definition
        self.lin1 = Linear(dim_h*3, 50)
        self.lin2 = Linear(50, 20)
        self.lin3 = Linear(20, num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        #h4 = self.conv3(h3, edge_index)

        # Graph-level readout
        #h0 = global_add_pool(x, batch)
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)
        #h4 = global_add_pool(h4, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.lin2(h)
        h = h.relu()
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.lin3(h)

        return h, F.log_softmax(h, dim=1)


def train(model, which_file_idx_as_val, test_set_bin):

    # include 1D, 7D, 8D, 9D always in training set
    training_dataset = []

    training_np = np.load("/home/muhitemon/graph_data_bin_wise_wo_edge_features/1D/1_graphData.npy", allow_pickle = True)
    training_dataset.append(training_np)

    training_np = np.load("/home/muhitemon/graph_data_bin_wise_wo_edge_features/7D/7_graphData.npy", allow_pickle = True)
    training_dataset.append(training_np)

    training_np = np.load("/home/muhitemon/graph_data_bin_wise_wo_edge_features/8D/8_graphData.npy", allow_pickle = True)
    training_dataset.append(training_np)

    training_np = np.load("/home/muhitemon/graph_data_bin_wise_wo_edge_features/9D/9_graphData.npy", allow_pickle = True)
    training_dataset.append(training_np)

    files = ["/home/muhitemon/graph_data_bin_wise_wo_edge_features/3D/3_graphData.npy", "/home/muhitemon/graph_data_bin_wise_wo_edge_features/4D/4_graphData.npy", "/home/muhitemon/graph_data_bin_wise_wo_edge_features/5D/5_graphData.npy", 
             "/home/muhitemon/graph_data_bin_wise_wo_edge_features/6D/6_graphData.npy", "/home/muhitemon/graph_data_bin_wise_wo_edge_features/10D/10_graphData.npy", "/home/muhitemon/graph_data_bin_wise_wo_edge_features/11D/11_graphData.npy"]

    for i in range(len(files)):
        if i != which_file_idx_as_val:
            training_np = np.load(files[i], allow_pickle = True)
            training_dataset.append(training_np)

    training_dataset = np.vstack(training_dataset)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    train_graphs = GraphDataset(training_dataset)
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)

    epochs = 3
    best_val_loss = float('inf')
    for epoch in range(epochs+1):
        total_loss = 0
        #acc = 0
        correct = 0

        # Train on batches
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            _, out = model(data.x, data.edge_index, data.batch)
            #print(out.shape)
            #print(data.y.shape)
            #print(data.batch.shape)
            #loss = criterion(out, data.y)
            loss = F.nll_loss(out, data.y.argmax(dim=1))
            total_loss += loss.item() * len(data)
            #acc += accuracy(out.argmax(dim=1), data.y.argmax(dim=1)) / len(loader)
            correct += (out.argmax(dim=1)==data.y.argmax(dim=1)).sum().item()
            loss.backward()
            optimizer.step()

        acc = correct / len(train_loader.dataset)
        total_loss = total_loss / len(train_loader.dataset)

        # Validation
        graphs_val = np.load(files[which_file_idx_as_val], allow_pickle=True)
        val_dataset = GraphDataset(graphs_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        val_loss, val_acc = test(model, val_loader)

        val_set_name = files[which_file_idx_as_val].split('/')[-1].split('_')[0]
        test_set_name = test_set_bin

        best_model_name = "val_" + val_set_name + "_"+ test_set_name + ".pth"
        best_model_path = "/home/muhitemon/saved_models_gin_wo_edge_features/"+ "test_" + test_set_name + "/" + best_model_name
        '''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
        '''
    # Print metrics every 10 epochs
    #if(epoch % 10 == 0):
        print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
            f'| Train Acc: {acc*100:>5.2f}% '
            f'| Val Loss: {val_loss:.4f} '
            f'| Val Acc: {val_acc*100:.2f}%')
        
    #test_loss, test_acc = test(model, test_loader)
    #print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%')
    del training_dataset
    del train_graphs
    del train_loader
    del training_np

    return model

def test(model, loader):
    #criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0
    acc = 0
    correct = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            _, out = model(data.x, data.edge_index, data.batch)
        
        loss = F.nll_loss(out, data.y.argmax(dim=1))
        total_loss += loss.item() * len(data)
        correct += (out.argmax(dim=1)==data.y.argmax(dim=1)).sum().item()

    total_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return total_loss, acc


'''
fixed_graphs_val = np.load("/home/muhitemon/graph_data_bin_wise_wo_edge_features/10D/10_graphData.npy", allow_pickle=True)
val_dataset = GraphDataset(fixed_graphs_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
'''

#trained_gine_model = train(gin_model, 0)
j = 0
while j<=5:

    gin_model = GINnet(32)
    gin_model = gin_model.to(device)
    train(gin_model, j, "2D")
    del gin_model
    torch.cuda.empty_cache()
    j+=1
