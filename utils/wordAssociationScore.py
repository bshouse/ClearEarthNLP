#!/usr/local/python/bin/pyhton2.7

import string
from math import log

import nltk
from collections import Counter

def joint_probability(ngrams, prior, ntype):
    ngramJointProb = dict()
    for ngram in ngrams:
        ngramSplits = ngram.split()
        if ntype == 2:
            ngramJointProb[ngram] = ngrams[ngram] * prior[ngramSplits[0]]
        elif ntype == 3:
            bigram = " ".join(ngramSplits[:-1]) 
            ngramJointProb[ngram] = ngrams[ngram] * prior[bigram]
        else:
            return dict()
    return ngramJointProb
    
def conditional_probability(ngrams, given, ntype):
    ngramcondProb = dict()
    for ngram in ngrams:
        ngramSplits = ngram.split()
        if ntype == 1:
            ngramcondProb[ngram] = ngrams[ngram] / given
        elif ntype == 2:
            ngramcondProb[ngram] = ngrams[ngram] / float(given[ngramSplits[0]])
        elif ntype == 3:
            bigram = " ".join(ngramSplits[:-1])
            ngramcondProb[ngram] = ngrams[ngram] / float(given[bigram])
        else:
            return dict()
    return ngramcondProb


def npmi(ngrams, unigramProbs):
    npmiScores = dict()
    for ngram in ngrams:
        first, second = ngram.split()
        pmi = ngrams[ngram] / (unigramProbs[first] * unigramProbs[second])
        npmiScores[ngram] = log(pmi)/-log(ngrams[ngram])
    return npmiScores
            
def extract_ngrams(text):
    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    for sentTokens in text:
        sentTokens = [token for token in sentTokens if token not in string.punctuation]
        [unigrams.update([" ".join(ngram)]) for ngram in nltk.ngrams(sentTokens, 1) ]
        [bigrams.update([" ".join(ngram)]) for ngram in nltk.ngrams(sentTokens, 2) ]
        [trigrams.update([" ".join(ngram)]) for ngram in nltk.ngrams(sentTokens, 3) ]
    return unigrams, bigrams, trigrams

def get_npmi(text):
    unigramCounts, bigramCounts, trigramCounts = extract_ngrams(text)
    vocab = float(sum(unigramCounts.values()))
    
    unigramProbs = conditional_probability(unigramCounts, vocab, 1)
    bigramCondProbs = conditional_probability(bigramCounts, unigramCounts, 2)
    #trigramCondProbs = conditional_probability(trigramCounts, bigramCounts, 3)

    bigramJointProbs = joint_probability(bigramCondProbs, unigramProbs, 2)
    #trigramJointProbs = joint_probability(trigramCondProbs, bigramJointProbs, 3)
    npmiScores = npmi(bigramJointProbs, unigramProbs)
    return bigramCounts, npmiScores

if __name__ == "__main__":
    import io
    import sys
    from isc_tokenizer import Tokenizer
    tok = Tokenizer(split_sen=True, lang="eng")
    with io.open(sys.argv[1]) as inp:
        tokenizedText = tok.tokenize(inp.read())
    npmiscores = get_npmi(tokenizedText)
    for term, score in npmiscores.iteritems():
        print ([term, score])

