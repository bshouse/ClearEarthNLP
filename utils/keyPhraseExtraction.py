#!/usr/local/python/bin/python

"""
From this paper: https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf

External dependencies: nltk, numpy, networkx

Based on https://gist.github.com/voidfiles/1646117
"""

import io
import os
import sys
import nltk
import string
import itertools
from operator import itemgetter
from collections import Counter

import networkx as nx
from utils.wordAssociationScore import get_npmi


#apply syntactic filters based on POS tags
def filter_for_tags(tagged, tags=['NN', 'NNP']):
    return [item for item in tagged if item[1] in tags]

def normalize(tagged):
    return [(item[0].replace('.', ''), item[1]) for item in tagged]

def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    if key is None:
        #for element in itertools.ifilterfalse(seen.__contains__, iterable):
        for element in filter(lambda e: e not in seen, iterable):
            seen.add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen.add(k)
                yield element

def extract_discont_ngrams(text):
    ngramCounter = Counter()
    for tokens in text:
        [ngramCounter.update([" ".join(ngram)]) for ngram in nltk.ngrams(tokens, 2) ]
        #[ngramCounter.update([" ".join(ngram)]) for ngram in nltk.ngrams(tokens, 3) ]
    return ngramCounter

def lDistance(firstString, secondString):
    """Function to find the Levenshtein distance between two words/sentences - 
    http://rosettacode.org/wiki/Levenshtein_distance#Python"""

    if len(firstString) > len(secondString):
        firstString, secondString = secondString, firstString
    distances = range(len(firstString) + 1)
    for index2, char2 in enumerate(secondString):
        newDistances = [index2 + 1]
        for index1, char1 in enumerate(firstString):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1])))
        distances = newDistances
    return distances[-1]

def buildGraph(nodes, ngrams=None):
    """nodes - list of hashables that represents the nodes of the graph"""
    
    gr = nx.Graph() #initialize an undirected graph
    gr.add_nodes_from(nodes)
    nodePairs = list(itertools.combinations(nodes, 2))
    
    #add edges to the graph (weighted by Levenshtein distance)
    for pair in nodePairs:
        firstString = pair[0]
        secondString = pair[1]
        if ngrams:
            distance = ngrams["%s %s" % (firstString,secondString)]
        else:
            distance = lDistance(firstString, secondString)
        if distance > 0.0:
            gr.add_edge(firstString, secondString, weight=distance+1)
        
    return gr

def extractKeyphrases(text):
    tokenizedText = [sentence.split() for sentence in text]
    bigramCounts, npmiScores = get_npmi(tokenizedText)

    #assign POS tags to the words in the text
    tagged = list()
    filteredText = list()
    for sentTokens in tokenizedText: #List of sentences with each sentence inturn a list of tokens 
        tagged_d = nltk.pos_tag(sentTokens)
        tagged += tagged_d
        filteredText.append([ptok[0].lower() for ptok in filter_for_tags(tagged_d) if ptok[0] not in string.punctuation])
    textlist = [x[0].lower() for x in tagged]
    
    ngrams = extract_discont_ngrams(filteredText) 
    tagged = filter_for_tags(tagged)
    #tagged = normalize(tagged)

    unique_word_set = unique_everseen([x[0].lower() for x in tagged if x[0] not in string.punctuation])
    word_set_list = list(unique_word_set)

   #this will be used to determine adjacent words in order to construct keyphrases with two words

    graph = buildGraph(word_set_list,ngrams)

    #pageRank - initial value of 1.0, error tolerance of 0,0001, 
    calculated_page_rank = nx.pagerank(graph, weight='weight')

    #most important words in ascending order of importance
    keyphrases = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)
    keyphrases = [kph for kph in keyphrases if len(kph) > 2]
    #the number of keyphrases returned will be relative to the size of the text (a third of the number of vertices)
    kTerms = round(len(word_set_list) / 1.4)
    keyphrases = keyphrases[0:int(kTerms)+1]

    #take keyphrases with multiple words into consideration as done in the paper - 
    #if two words are adjacent in the text and are selected as keywords, join them together
    modifiedKeyphrases = set([])
    dealtWith = set([]) #keeps track of individual keywords that have been joined to form a keyphrase
    i = 0
    j = 1
    k = 2
    textlist = textlist+['@#%DuMmY%#@']
    while k < len(textlist):
        firstWord = textlist[i]
        secondWord = textlist[j]
        thirdWord = textlist[k]
        if firstWord in keyphrases and secondWord in keyphrases:
            bi_keyphrase = "%s %s" % (firstWord, secondWord)
            if thirdWord in keyphrases:
                tri_keyphrase = "%s %s %s" % (firstWord, secondWord, thirdWord)
                if ngrams[tri_keyphrase] > 0: modifiedKeyphrases.add(tri_keyphrase)
                dealtWith.add(bi_keyphrase)
            else:
                #if ngrams[bi_keyphrase] > 0: 
                if (npmiScores.get(bi_keyphrase, 0.0) > 0.5) and (bigramCounts.get(bi_keyphrase, 0) > 2):
                        modifiedKeyphrases.add(bi_keyphrase)
                        dealtWith.update([firstWord, secondWord])
                        
        i = i + 1
        j = j + 1
        k = k + 1
    modifiedKeyphrases = modifiedKeyphrases - dealtWith
    #return list(modifiedKeyphrases) + list(set(keyphrases) - dealtWith)
    return list(modifiedKeyphrases) + keyphrases

def generatePairs(text):
    glossary = [ep for ep in extractKeyphrases(text) if len(ep) > 2]
    #for kp_i in phrases:
    while glossary:
        kp_i = glossary.pop(0)
        for kp_j in glossary:
            if kp_i == kp_j: continue
            if kp_i in kp_j: continue
            yield (kp_i, kp_j)


if __name__ == "__main__":
    import io
    with io.open(sys.argv[1]) as inp:
        terms = extractKeyphrases(inp.read())
        print (terms)
