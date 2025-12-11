from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import Linear as Lin
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import degree

from .mlp import MLP


class GINEN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers,
        batch_norm=False,
        cat=True,
        lin=True,
    ):
        super(GINEN, self).__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.cat = cat
        self.lin = lin

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP(in_channels, out_channels, 2, batch_norm, dropout=0.0)
            self.convs.append(GINENConv(mlp, train_eps=True, edge_dim=1))
            in_channels = out_channels

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.final = Lin(in_channels, out_channels)
        else:
            self.out_channels = in_channels

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """"""
        xs = [x]

        for conv in self.convs:
            xs += [conv(xs[-1], edge_index, edge_attr)]

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = self.final(x) if self.lin else x
        return x

    def __repr__(self):
        return ("{}({}, {}, num_layers={}, batch_norm={}, cat={}, lin={})").format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.num_layers,
            self.batch_norm,
            self.cat,
            self.lin,
        )


class GINENConv(MessagePassing):
    def __init__(
        self,
        nn: torch.nn.Module,
        eps: float = 0.0,
        train_eps: bool = False,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer("eps", torch.empty(1))
        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, "in_features"):
                in_channels = nn.in_features
            elif hasattr(nn, "in_channels"):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            self.lin = Linear(edge_dim, in_channels)

        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        # Calculate target node degrees for normalization
        # Handle both edge_index tensor and SparseTensor formats
        if hasattr(edge_index, "__getitem__"):
            # Regular edge_index tensor
            deg_tgt = degree(edge_index[1], num_nodes=x[1].size(0), dtype=torch.float)
        else:
            # SparseTensor format
            deg_tgt = edge_index.storage.colcount().to(torch.float)

        deg_tgt = deg_tgt.view(-1, 1)  # Reshape for proper broadcasting

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        # Normalize by target node degree
        out = out / deg_tgt

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError(
                "Node and edge feature dimensionalities do not "
                "match. Consider setting the 'edge_dim' "
                "attribute of 'GINEConv'"
            )

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"
