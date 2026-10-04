"""Corrected periodic message-passing building blocks, not a trained model.

CGCNN-inspired, not an exact reproduction of the published architecture.
Old checkpoints and metrics are incompatible with this implementation.
"""
import numpy as np
import torch
from torch import nn


def periodic_edges(structure, cutoff=6.0, rbf_count=8):
    if not structure.is_ordered:
        raise ValueError('Disordered sites require an explicit occupancy model.')
    src, dst, images, distances = structure.get_neighbor_list(cutoff, exclude_self=True)
    # Keep all periodic images, including nonzero-image self neighbors.
    if len(src) == 0:
        raise ValueError('No physical neighbors within cutoff; no synthetic edge is inserted.')
    centers = np.linspace(0,cutoff,rbf_count)
    edge = np.exp(-.5*((distances[:,None]-centers[None,:])/.5)**2).astype(np.float32)
    return src.astype(np.int64),dst.astype(np.int64),images,distances,edge


class PeriodicGatedConv(nn.Module):
    def __init__(self, width=32, edge_dim=8):
        super().__init__()
        self.message=nn.Linear(2*width+edge_dim,2*width)

    def forward(self, nodes, edges, src, dst):
        # Messages depend on the center, the neighboring atom, and the bond.
        gate,core=self.message(torch.cat([nodes[src],nodes[dst],edges],dim=1)).chunk(2,dim=1)
        msg=torch.sigmoid(gate)*torch.nn.functional.softplus(core)
        aggregate=torch.zeros_like(nodes)
        aggregate.index_add_(0,src,msg)
        return torch.nn.functional.softplus(nodes+aggregate)


def mean_pool(nodes, counts):
    # One atom contributes once. Element fractions would double-weight abundance.
    return torch.stack([part.mean(0) for part in torch.split(nodes,[int(n) for n in counts])])
