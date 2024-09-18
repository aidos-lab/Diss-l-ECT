import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, Reddit, WebKB, WikipediaNetwork, Amazon
from baselines.models import GCN, GAT
from torch_geometric.nn.norm import LayerNorm
import numpy as np

def main():
    # Load the dataset
    dataset = Amazon(root='/tmp/Photo',name='Photo')
    data = dataset[0]

    try:
        if len(data.train_mask.shape)>1:
            train_mask = data.train_mask[:,0]
            test_mask = data.test_mask[:, 0]
        else:
            train_mask = data.train_mask
            test_mask = data.test_mask
    except AttributeError:
        bool_list = [True,False]
        p = [.75,.25]
        train_mask = np.random.choice(bool_list,len(data.x),p=p)
        test_mask = [not x for x in train_mask]

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(dataset=dataset).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    def test():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        acc = (pred[test_mask] == data.y[test_mask]).sum().item() / sum(test_mask)
        return acc

    for epoch in range(1, 1001):
        loss = train()
        acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')


if __name__ == '__main__':
    main()
