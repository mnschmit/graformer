from typing import Iterable, List, Dict, Tuple, Optional


def index_strings(strings: Iterable[str], sp) -> List[List[int]]:
    res = []
    for s in strings:
        res.append(sp.EncodeAsIds(s))
    return res


def index_node_labels(node_labels: Iterable[str], sp) -> List[List[int]]:
    nodes: List[List[int]] = []
    for n in node_labels:
        # remove <s> and </s> for node labels
        nodes.append(sp.EncodeAsIds(n)[1:-1])
    return nodes


def tokenize_entities(
        entity_labels: Iterable[str], sp,
        start_node_id=0, sample_alpha: Optional[float] = None
) -> Tuple[Dict[str, List[int]], List[int], List[int]]:
    ent2nodes: Dict[str, List[int]] = {}
    nodes: List[int] = []
    pos: List[int] = []
    node_id = start_node_id
    for el in entity_labels:
        # this removes <s> and </s>
        token_indices = sp.encode(
            el, enable_sampling=sample_alpha is not None, alpha=sample_alpha)
        nodes.extend(token_indices)
        pos.extend(range(len(token_indices)))
        ent2nodes[el] = [idx + node_id for idx in range(len(token_indices))]
        node_id += len(token_indices)
    return ent2nodes, nodes, pos
