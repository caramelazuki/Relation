# Relation

A PyTorch layer that shares global context across a sequence without constructing an attention matrix.

## Usage

```python
import torch
from relation import Relation

inputs = torch.randn(32, 512, 64)
mask = torch.ones(32, 512, dtype=torch.bool)

relation = Relation(in_features=64)
outputs = relation(inputs, mask)
```

Relation is a standard `torch.nn.Module` designed to replace a self-Attention layer: it maps one token sequence to a contextualized sequence of the same length without computing an attention matrix.

`inputs` and `outputs` have shape `(batch, sequence, features)`. The mask is optional; `True` marks a valid token. Set `out_features` when the output feature size should differ from the input feature size.

## Installation

```bash
python -m pip install "git+https://github.com/caramelazuki/Relation.git"
```

## Reference

Yamada KD, Baladram MS, Lin F, [Relation is an Option for Processing Context Information](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2022.924688/), Frontiers in Artificial Intelligence, 2022
