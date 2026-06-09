"""
Load a pack (tree + measure registry) from a YAML file. A pack file has two
top-level keys:

    measures:            # the vocabulary the tree may reference
      gmv:    {kind: sum, column: gmv_usd}
      orders: {kind: count}
      aov:    {kind: ratio, numerator: gmv, denominator: orders}
    tree:                # the frozen hypothesis tree
      id: gmv
      ...

This is what makes the YAML real instead of decorative: load_pack() parses it
into the same Node / Measure objects the engine walks, so trees/*.yaml and the
Python constants in tree.py are interchangeable.
"""
from __future__ import annotations
from dataclasses import dataclass

import yaml

from .tree import Node
from .measures import Measure, registry_from_dict


@dataclass
class Pack:
    tree: Node
    measures: dict[str, Measure]


def _node(d: dict) -> Node:
    return Node(
        id=d["id"],
        label=d.get("label", d["id"]),
        measure=d["measure"],
        relation=d.get("relation", "root"),
        segment_dims=list(d.get("segment_dims", [])),
        sign=int(d.get("sign", 1)),
        note=d.get("note", ""),
        children=[_node(c) for c in d.get("children", [])],
    )


def load_pack(path: str) -> Pack:
    with open(path) as fh:
        doc = yaml.safe_load(fh)
    if "tree" not in doc or "measures" not in doc:
        raise ValueError(f"{path}: pack needs top-level 'measures' and 'tree' keys")
    return Pack(tree=_node(doc["tree"]),
                measures=registry_from_dict(doc["measures"]))
