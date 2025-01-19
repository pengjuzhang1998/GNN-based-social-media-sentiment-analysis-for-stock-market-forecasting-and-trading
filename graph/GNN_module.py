import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
#from coomat import set_coomat
import pickle
from set_graph import set_graph
import os

def min_max_normalize(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    return 100 * (arr - arr_min) / (arr_max - arr_min)

dir_path = './data/'
#rel_matrix , node_dict = set_coomat(dir_path)
rel_matrix, node_dict = set_graph(dir_path = dir_path, end_time = '2022-12-31', stock_name = 'XOM', time_period = 365)

rel_matrix = rel_matrix.coalesce()
with open('node_XOM.pkl', 'wb') as f:
    pickle.dump(node_dict, f)
print('finish read')
node_names = list(node_dict.keys())
node_features = torch.tensor([node_dict[node] for node in node_names], dtype=torch.float32)
node_features = node_features.view(-1, 1)

# Construct edge_index from node_dict
edge_index = []
for node1 in node_names:
    for node2 in node_names:
        if node1 != node2:
            if [node_names.index(node1), node_names.index(node2)] not in edge_index:
                edge_index.append([node_names.index(node1), node_names.index(node2)])
edge_index = torch.tensor(edge_index).t()

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.relu = nn.ReLU()
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x

num_nodes = len(node_dict)

# Set the output feature dimension for the GCN.
out_features = num_nodes
#print(node_features.shape)
# Create the GCN model.
model = GCN(num_features = 1, hidden_dim=16, num_classes=out_features) #only 1 feature firstly
target_nodes = list(range(num_nodes))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
node_features = node_features.to(device)
edge_index = edge_index.to(device)

# Define the custom loss function.
def mse_loss(influence_rates_pred, influence_rates_actual):
    influence_rates_actual = influence_rates_actual.expand_as(influence_rates_pred)
    return F.mse_loss(influence_rates_pred, influence_rates_actual)

affecttive_rate = 0.1 # Influence factor of the influence factor of the initiating account of the action on the influence factor of the receiving account of the action
print('start training')
# Set the optimizer.
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
num_epochs = 100
# Train the GCN model.
for epoch in range(num_epochs):
    # Forward pass through the GCN model.
    output = model(node_features, edge_index)

    # Compute the influence rates for all nodes.
    influence_rates_pred = output.squeeze()

    # Compute the actual influence rates from the relative matrix.
    influence_rates_actual = torch.zeros(num_nodes)
    for i, j in rel_matrix.indices().t():
        v = rel_matrix[i,j]
        influence_rates_actual[i] += v * (1+influence_rates_actual[j]*affecttive_rate)
    influence_rates_actual = influence_rates_actual.to(device)
    # Compute the loss.
    loss = mse_loss(influence_rates_pred, influence_rates_actual.detach())

    # Backward pass and optimization step.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Print the loss for this epoch.
    print('Epoch {}: Loss = {}'.format(epoch, loss.item()))


# Compute the influence rates for the target nodes.
influence_rates = model(node_features.to(device), edge_index.to(device))[target_nodes].squeeze().detach().cpu().numpy()


influence_rates_normalized = min_max_normalize(influence_rates)

with open('influence_rate_JNJ.pkl', "wb") as f:
    pickle.dump(influence_rates_normalized, f) 
print(influence_rates_normalized)
