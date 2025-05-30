import torch
import torch.nn as nn
from layers.config import EctConfig
from torch_geometric.data import Data

from typing import Protocol
from dataclasses import dataclass


def compute_ecc_derivative(nh, index, lin,out):
    ecc = torch.nn.functional.sigmoid(50 * torch.sub(lin, nh))* (1- torch.nn.functional.sigmoid(50 * torch.sub(lin, nh)))
    return torch.index_add(out,1, index, ecc).movedim(0, 1)


def compute_ect_points_derivative(data, v, lin, out):
    nh = data.x @ v
    return compute_ecc_derivative(nh, data.batch, lin, out)


def compute_ecc(nh, index, lin,out):
    ecc = torch.nn.functional.sigmoid(500*torch.sub(lin, nh))
    return torch.index_add(out,1, index, ecc).movedim(0, 1)


def compute_ect_points(data, v, lin, out):
    nh = data.x @ v
    return compute_ecc(nh, data.batch, lin, out)


def compute_ect_edges(data, v, lin, out):
    nh = data.x @ v
    eh, _ = nh[data.edge_index].max(dim=0)
    return compute_ecc(nh, data.batch, lin,out) - compute_ecc(
        eh, data.batch[data.edge_index[0]], lin, out
    )


def compute_ect_faces(data, v, lin, out):
    nh = data.x @ v
    eh, _ = nh[data.edge_index].max(dim=0)
    fh, _ = nh[data.face].max(dim=0)
    return (
        compute_ecc(nh, data.batch, lin, out)
        - compute_ecc(eh, data.batch[data.edge_index[0]], lin,out)
        + compute_ecc(fh, data.batch[data.face[0]], lin ,out)
    )


class  EctLayer(nn.Module):
    """docstring for EctLayer."""

    def __init__(self, config: EctConfig, v=None):
        super().__init__()
        self.config = config
        self.lin = (
            torch.linspace(-config.R, config.R, config.bump_steps)
            .view(-1, 1, 1)
            .to(config.device)
        )
        if torch.is_tensor(v):
            self.v = v
        else:
            self.v = (torch.rand(size=(config.num_features, config.num_thetas)) - 0.5).T.to(config.device)
            self.v /= self.v.pow(2).sum(axis=1).sqrt().unsqueeze(1)
            self.v = self.v.T 

        if config.ect_type == "points":
            self.compute_ect = compute_ect_points
        elif config.ect_type == "edges":
            self.compute_ect = compute_ect_edges
        elif config.ect_type == "faces":
            self.compute_ect = compute_ect_faces
        elif config.ect_type == "points_derivative":
            self.compute_ect = compute_ect_points_derivative

    def forward(self, data):
        out = torch.zeros(size=(self.config.bump_steps,data.batch.max().item()+1,self.config.num_thetas), device=self.config.device)
        ect = self.compute_ect(data, self.v, self.lin,out)
        if self.config.normalized:
            return ect / torch.amax(ect,dim=(1,2)).unsqueeze(1).unsqueeze(1)
        else:
            return ect

