from typing import Tuple, List
import torch


def pad(matrix: List[List[int]], pad_elem: int, max_length=None) -> int:
    if max_length is None:
        max_length = max(len(row) for row in matrix)

    for row in matrix:
        for _ in range(max_length - len(row)):
            row.append(pad_elem)

    return max_length


def pad_square_matrices(list_of_squares: List[List[List[int]]], pad_elem: int) -> int:
    max_length = max(len(matrix) for matrix in list_of_squares)
    for square in list_of_squares:
        pad(square, pad_elem, max_length=max_length)
        for _ in range(max_length - len(square)):
            square.append([pad_elem] * max_length)

    return max_length


def pad_batch_of_multiple_references(texts: List[List[List[int]]], pad_elem: int) -> int:
    max_num_references = 0
    max_ref_length = 0
    for ref_list in texts:
        for ref in ref_list:
            if len(ref) > max_ref_length:
                max_ref_length = len(ref)
        if len(ref_list) > max_num_references:
            max_num_references = len(ref_list)

    for ref_list in texts:
        for _ in range(max_num_references - len(ref_list)):
            ref_list.append([])
        pad(ref_list, pad_elem, max_length=max_ref_length)

    return max_num_references, max_ref_length


def collate_webnlg(
        samples: List[Tuple[List[List[int]], List[bool], List[int],
                            List[int], List[int]]],
        pad_elem: int = 0
) -> Tuple[torch.LongTensor, torch.BoolTensor, torch.LongTensor,
           torch.LongTensor, torch.LongTensor]:
    # The input `samples` is a list of tuples (dm, is_entity, positions, node labels, text).
    dms, is_entity, pos, node_labels, texts = map(list, zip(*samples))

    max_length = pad_square_matrices(dms, pad_elem)
    pad(is_entity, False, max_length=max_length)
    pad(pos, pad_elem, max_length=max_length)

    pad(node_labels, pad_elem, max_length)

    if isinstance(texts[0][0], list):
        pad_batch_of_multiple_references(texts, pad_elem)
    else:
        pad(texts, pad_elem)

    dm_tensor = torch.LongTensor(dms)
    is_ent_tensor = torch.BoolTensor(is_entity)
    pos_tensor = torch.LongTensor(pos)
    nl_tensor = torch.LongTensor(node_labels)
    text_tensor = torch.LongTensor(texts)

    return dm_tensor, is_ent_tensor, pos_tensor, nl_tensor, text_tensor
