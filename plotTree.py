#!/usr/bin/python3


import sys
import pydot
import codecs
import requests
import numpy as np
from bs4 import BeautifulSoup

from collections import namedtuple as nt, defaultdict as dfd


def adjacencyMatrixplot(nodes):
    """Builds an adjacency matrix of a dependency graph"""
    adMat = np.zeros((len(nodes), len(nodes)), int)
    for node in nodes:
        if (node.id == 0):continue
        parent, child = node.parent, node.id # -1 -> tally with list indices
        adMat[parent, child] = 1
    return adMat

def dependencyGraph(conll_text):
    leaf = nt('leaf', ['id','form','lemma','ctag','tag','features','parent','drel'] )
    for node in [0] + conll_text + [-1]:
        if node in [0, -1]:
            yield leaf._make([0, 'ROOT_F', 'ROOT_L', 'ROOT_C', 'ROOT_P', dfd(str), -1, '_'])
        else:
            id_,form,lemma,ctag,tag,features,parent,drel = node.strip().split("\t")[:8]
            yield leaf._make([int(id_),form,lemma,ctag,tag,features,int(parent),drel])

def BFSPlot(data, adjMatrix, root):
    """Breadth-first search over labelled arcs to generate a dependency tree for visualisation via pydot."""
    graph = pydot.Dot(graph_type='graph', splines='polyline')
    pNode = pydot.Node(str(root), label='ROOT', shape='plaintext')
    graph.add_node(pNode)
    queue = [(root, pNode)]
    while queue:
        node,pNode = queue.pop(0)
        adjList = np.nonzero(adjMatrix[node])[0]
        for child in adjList:
            form = data[child].form
            cNode = pydot.Node(str(child), label='"%s"'%form, shape='plaintext')
            graph.add_node(cNode)
            if data[child].drel == "root":
                edge = pydot.Edge(pNode, cNode, fontsize="8.0", fontcolor='blue')
            else:
                edge = pydot.Edge(pNode, cNode, label='"%s"'%data[child].drel, fontsize="8.0", fontcolor='blue')
            graph.add_edge(edge)
            queue.append((child, cNode))

    return graph
