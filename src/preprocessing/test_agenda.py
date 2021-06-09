from .prepare_agenda import generate_graph
import matplotlib.pyplot as plt
import networkx as nx
import json


def show_graph(word2idx, graph):
    idx2word = {v: k for k, v in word2idx.items()}

    for i, label in enumerate(graph.ndata['label']):
        print(i, ':', [idx2word[l.item()] for l in label])

    print('# Edges #')
    for i, label in enumerate(graph.edata['label']):
        n1 = graph.edges()[0][i].item()
        n2 = graph.edges()[1][i].item()
        print(n1, '->', n2, ':', idx2word[label.item()])

    nx.draw(graph.to_networkx(), with_labels=True)
    plt.show()


def test_file(filename):
    with open(filename) as f:
        val_data = json.load(f)
    sample = val_data[0]

    word2idx = {'<pad>': 0}
    graph = generate_graph(sample, word2idx)

    show_graph(word2idx, graph)


def test_val():
    test_file('data/raw/agenda/unprocessed.val.json')


def test_train():
    test_file('data/raw/agenda/unprocessed.train.json')


if __name__ == '__main__':
    test_train()
