#!/usr/local/python/bin/python -*- coding: utf-8 -*-

import io
import re
import os
import sys
import copy
import socket
import timeit
import pickle

import argparse
import numpy as np
from collections import namedtuple as nt, defaultdict as dfd, Counter

from scipy.spatial import distance

import dynet as dy
#from nltk.corpus import stopwords
from gensim.models.word2vec import *
from nltk.stem.wordnet import WordNetLemmatizer

np.random.seed(100)
_MAX_BUFFER_SIZE_ = 102400

class Meta:
    def __init__(self):
        self.c_dim = 32
        self.add_words = 3
        self.n_hidden = 128
        self.lstm_word_dim = 64
        self.lstm_char_dim = 32
        self.tdmaps = {"Hypernym": 0, "Unrelated": 1}

class SubsumptionLearning(object):
    def __init__(self, model=None, meta=None):
        self.model = dy.Model()
        if model:
            self.meta = pickle.load(open('%s.meta' %model, 'rb'))
        else:
            self.meta = meta
    
        # ndims: hidden x input::[for eachevent in (event1, event2) 100 dims farwordlstmvec, 100 dims backwardlstmvec]
        self.pW1 = self.model.add_parameters((self.meta.n_hidden, self.meta.lstm_word_dim*2)) #2 for pair of events; 2 for forward and backward
        self.pb1 = self.model.add_parameters(self.meta.n_hidden) # ndims: hidden units
        self.pW2 = self.model.add_parameters((self.meta.n_out, self.meta.n_hidden)) #ndims: output x hidden
        self.pb2 = self.model.add_parameters(self.meta.n_out) # ndims: output
    
        #self.fwdRNN = dy.LSTMBuilder(1, self.w_dim, self.lstm_word_dim, self.model) # layers, in-dim, out-dim, model
        #self.bwdRNN = dy.LSTMBuilder(1, self.w_dim, self.lstm_word_dim, self.model) 
        #self.cFwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.lstm_char_dim, self.model)
        #self.cBwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.lstm_char_dim, self.model)
    
        #self.pword2lstm = self.model.add_parameters((self.lstm_word_dim*2, self.w_dim + self.nlexdims))
        #self.pword2lstmbias = self.model.add_parameters(self.lstm_word_dim*2)
    
        self.WORDS_LOOKUP = self.model.add_lookup_parameters((self.meta.n_words+self.meta.add_words, self.meta.w_dim))
        self.CHARS_LOOKUP = self.model.add_lookup_parameters((self.meta.n_chars, self.meta.c_dim))
    
        if not model:
            for word, V in wvm.vocab.iteritems():
                self.WORDS_LOOKUP.init_row(V.index, wvm.syn0[V.index])
        if model:
            self.model.populate('%s.dy' %model)

        self.lmtzr = WordNetLemmatizer()
        #self.stop = set(stopwords.words('english'))
        sfile = open("misc/stopwords.txt")
        self.stop = set([sword.strip() for sword in sfile])

    def initialize_graph_nodes(self, train=False):
        #if not train:
        #    self.fwdRNN.disable_dropout()
        #    self.bwdRNN.disable_dropout()
        #else:
        #    self.fwdRNN.set_dropout(0.3)
        #    self.bwdRNN.set_dropout(0.3)

        self.W1 = dy.parameter(self.pW1)#parameter(self.model["W1"])
        self.b1 = dy.parameter(self.pb1)#parameter(self.model["b1"])
        self.W2 = dy.parameter(self.pW2)#parameter(self.model["W2"])
        self.b2 = dy.parameter(self.pb2)#parameter(self.model["b2"])
    
        #self.cf_init = self.cFwdRNN.initial_state()
        #self.cb_init = self.cBwdRNN.initial_state()

        #self.f_init = self.fwdRNN.initial_state()
        #self.b_init = self.bwdRNN.initial_state()

    def get_linear_embd(self, sequence):
        embs = list()
        flag = True
        for node in sequence:
            try:
                index = self.meta.w2i[node]
                if node == sequence[-1]: flag = True 
                embs.append(self.WORDS_LOOKUP[index])
            except KeyError:
                flag = False
                embs.append(self.WORDS_LOOKUP[self.meta.n_words])
                #pad_char = self.meta.c2i["<*>"]
                #char_ids = [pad_char] + [self.meta.c2i[c] if self.meta.cc[c]>5 else self.meta.c2i['_UNK_'] for c in node] + [pad_char]
                #char_embs = [self.CHARS_LOOKUP[cid] for cid in char_ids]
                #fw_exps = self.cf_init.transduce(char_embs)
                #bw_exps = self.cb_init.transduce(reversed(char_embs))
                #embs.append(dy.concatenate([ fw_exps[-1], bw_exps[-1] ]))
                ##yield self.WORDS_LOOKUP[self.n_words]
        if embs:
            embs[-1] *= 0.6
            embs = [emb*(0.4/len(embs[:-1])) for emb in embs[:-1]] + [embs[-1]]
        return embs, flag

    def predict_hyp(self, subtype, supertype):
        dy.renew_cg()
        self.initialize_graph_nodes()
    
        subtype = [self.lmtzr.lemmatize(sb) for sb in subtype.split() if sb not in self.stop]
        supertype = [self.lmtzr.lemmatize(sp) for sp in supertype.split() if sp not in self.stop]
        if subtype == supertype:return
        # context insensitive embeddings or local embeddings
        fembs, DSTATUS_X = self.get_linear_embd(subtype)
        sembs, DSTATUS_Y = self.get_linear_embd(supertype)
    
        if len(fembs) < 1 or len(sembs) < 1: return
        if (DSTATUS_X is False) or (DSTATUS_Y is False): return
        fembs = fembs[0] if len(fembs) == 1 else dy.average(fembs)
        sembs = sembs[0] if len(sembs) == 1 else dy.average(sembs)
    
        x = dy.concatenate([fembs,sembs])
        
        #e_dist = dy.squared_distance(fembs, sembs)
        #e_dist = distance.euclidean(fembs.npvalue(), sembs.npvalue())
        e_dist = 1 - distance.cosine(fembs.npvalue(), sembs.npvalue())
        #weighted_x = x * e_dist
        output = dy.softmax(self.W2*(dy.rectify(self.W1*x) + self.b1) + self.b2)
        prediction = np.argmax(output.npvalue())
        confidence = np.max(output.npvalue())
        return self.meta.rmaps[prediction], confidence, e_dist

def Train(instances, itercount):
    dy.renew_cg()
    ontoparser.initialize_graph_nodes(train=True)

    loss = []
    errors = 0.0
    for instance in instances:
        fexpr, sexpr, groundtruth = instance
        # context insensitive embeddings or local embeddings
        subtype = [sb.lower() for sb in fexpr.split()] #if sb.lower() not in stop]
        supertype = [sp.lower() for sp in sexpr.split()] #if sp.lower() not in stop]
        fembs, DSTATUS_X = ontoparser.get_linear_embd(subtype)
        sembs, DSTATUS_Y = ontoparser.get_linear_embd(supertype)

        #if (DSTATUS_X is False) or (DSTATUS_Y is False): continue
        fembs = fembs[0] if len(fembs) == 1 else dy.average(fembs)
        sembs = sembs[0] if len(sembs) == 1 else dy.average(sembs)

        x = dy.concatenate([fembs,sembs])

        #e_dist = dy.squared_distance(fembs, sembs)
        e_dist = 1 - distance.cosine(fembs.npvalue(), sembs.npvalue())
        #weighted_x = x * e_dist
        output = ontoparser.W2*(dy.rectify(ontoparser.W1*x) + ontoparser.b1) + ontoparser.b2
        
        prediction = np.argmax(output.npvalue())
        loss.append(dy.pickneglogsoftmax(output, ontoparser.meta.tdmaps[groundtruth]))
        #if ((ontoparser.meta.rmaps[prediction] == "Hypernym") and ("Hypernym" != groundtruth)) and (e_dist < 0.5):
        #    loss[-1] += -log(0.6)
        errors += 0 if groundtruth == ontoparser.meta.rmaps[prediction] else 1
    return loss, errors

def nntraining(dataset):
    sys.stderr.write("Started training ...\n")
    sys.stderr.write("Training Examples: %s Classes: %s\n" % (len(dataset), len(ontoparser.meta.tdmaps)))
    for epoch in range(15):#args.epochs):
        np.random.shuffle(dataset)
        errors = 0
        batchSize = 30
        for instance in range(0,len(dataset), 30):
            loss, error = Train(dataset[instance:instance+batchSize], epoch+1)
            if not loss: continue
            errors += error
            cumulativeloss = dy.esum(loss)
            _ = cumulativeloss.scalar_value()
            cumulativeloss.backward()
            trainer.update()
        
        accuracy = Test(inputGenDev)
        ontoparser.model.save('%s.dy' %args.save_model)
    sys.stderr.write("Epoch:: %s Loss:: %s Test Accuracy:: %s\n" % (epoch+1, 100.*errors/len(dataset), accuracy))


def Test(dataset):
    sys.stderr.write("Started testing ...\n")
    inst = {"Hypernym":0., "Unrelated":0.}
    hits = {"Hypernym":[0.,0.], "Unrelated":[0.,0.]}
    for idx, instance in enumerate(dataset):
        subtype, supertype, groundtruth = instance.strip().split("\t")
        groundtruth = "Hypernym" if groundtruth == "True" else "Unrelated"
        inst[groundtruth] += 1
        try:
            prediction, confidence, edist = predict_hyp(subtype, supertype)
            #print subtype, supertype, prediction, groundtruth, edist
        except TypeError:
            continue
        if groundtruth == prediction: 
            hits[groundtruth][0] += 1
        else:
            hits[groundtruth][-1] += 1
    #return [(hit, 100.*hits[hit]/inst[hit]) for hit in hits if inst[hit]]
    total = sum(hits['Hypernym']+hits['Unrelated'])
    return [(hit, 100.*hits[hit][0]/sum(hits[hit])) for hit in hits]

def read(inputGenTrain):
    global train_sents, meta, plabels, tdlabels
    for instance in inputGenTrain:
        firstW, secondW, rel = instance.strip().split("\t")
        train_sents.append((firstW, secondW, "Hypernym" if rel == "True" else "Unrelated"))
        meta.cc.update(firstW+secondW)

def processInput(ifp, ofp):
    #dy.renew_cg()
    #ontoparser.initialize_graph_nodes()
    for line in ifp:
        if not line.strip():continue
        subtype, supertype = line.strip().split('\t')
        predictions = list()
        for pair in [(subtype, supertype), (supertype, subtype)]:
            try:
                prediction, confidence, distance = predict_hyp(pair[0], pair[1])
                predictions.append((subtype, supertype, prediction, confidence, distance))
            except:
                continue
        if len(predictions) < 2:continue
        if predictions[0][3] >= predictions[1][3]:
            subtype, supertype, prediction, confidence, distance = predictions[0]
            ofp.write("%s\t%s\t%s\t%s\t%s\n" % (subtype, supertype, prediction, str(confidence), str(distance)))
        else:
            subtype, supertype, prediction, confidence, distance = predictions[1]
            ofp.write("%s\t%s\t%s\t%s\t%s\n" % (subtype, supertype, prediction, str(confidence), str(distance)))

def run_client(ip, port, clientsocket):
    data = clientsock.recv(_MAX_BUFFER_SIZE_)
    fakeInputFile = StringIO.StringIO(data)
    fakeOutputFile = StringIO.StringIO("")
    processInput(fakeInputFile, fakeOutputFile)
    fakeInputFile.close()
    clientsocket.send(fakeOutputFile.getvalue())
    fakeOutputFile.close()
    clientsocket.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Neural Network Ontology Extractor.", description="MLP Onto Extractor")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('--dynet-mem')
    parser.add_argument('--dynet-seed', dest='seed', type=int)
    parser.add_argument('--trainer', help='NN Optimizer [simsgd|momsgd|adam|adadelta|adagrad]', default='momsgd')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ebin', type=int, default=1, help='1 if binary embeddings else 0')
    parser.add_argument('--train', help="<train-file>")
    parser.add_argument('--dev', help="<development-file>")
    parser.add_argument('--embedding', help="<word2vec-embedding>")
    group.add_argument('--save-model', dest='save_model')
    group.add_argument('--load-model', dest='load_model')
    parser.add_argument('-d', '--daemonize', dest='isDaemon', help='Daemonize me?', action='store_true', default = False)
    parser.add_argument('-p', '--port', type=int, dest='daemonPort', help='Specify a port number')
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dev:
        with io.open(args.dev, encoding='utf-8') as fp:
            inputGenDev = fp.readlines()

    meta = Meta()
    if not args.load_model:
        with io.open(args.train, encoding='utf-8') as fp:
            inputGenTrain = fp.readlines()
        try:
            wvm = Word2Vec.load_word2vec_format(args.embedding, binary=True)#args.ebin)
        except:
            wvm = Word2Vec.load(args.embedding)#args.ebin)
        meta.w_dim = wvm.syn0.shape[1]
        meta.lstm_word_dim = meta.w_dim
        meta.c_dim = meta.w_dim / 2
        meta.lstm_char_dim = meta.w_dim / 2
        meta.n_words = wvm.syn0.shape[0]+meta.add_words

        tdlabels = set()
        train_sents = []
        meta.cc = Counter()
        tdlabels.update(['Hypernym', 'Unrelated'])
        meta.cc.update(['<*>', '_UNK_'])

        read(inputGenTrain)
    
        meta.c2i = dict(zip(meta.cc.keys(), range(len(meta.cc))))
        meta.n_chars = len(meta.c2i)
        meta.n_out = len(meta.tdmaps)
        meta.rmaps = {v:k for k,v in meta.tdmaps.items()}

        meta.w2i = {}
        for w in wvm.vocab:
            meta.w2i[w] = wvm.vocab[w].index

    if args.save_model:
        pickle.dump(meta, open('%s.meta' %args.save_model, 'wb'))
    if args.load_model:
        ontoparser = SubsumptionLearning(model=args.load_model)
    else:
        ontoparser = SubsumptionLearning(meta=meta)
        trainers = {
            'momsgd'  : dy.MomentumSGDTrainer(ontoparser.model, edecay=0.25),
            'adam'    : dy.AdamTrainer(ontoparser.model, edecay=0.25),
            'simsgd'  : dy.SimpleSGDTrainer(ontoparser.model, edecay=0.25),
            'adagrad' : dy.AdagradTrainer(ontoparser.model, edecay=0.25),
            'adadelta' : dy.AdadeltaTrainer(ontoparser.model, edecay=0.25)
            }
        trainer = trainers[args.trainer]
        nntraining(train_sents)

    if args.dev:
        accuracy = Test(inputGenDev)
        sys.stdout.write("Accuracy: {}%\n".format(accuracy))

    if args.isDaemon and args.daemonPort:
        sys.stderr.write('Leastening at port %d\n' %args.daemonPort)
        host = "0.0.0.0" #Listen on all interfaces
        port = args.daemonPort #Port number

        tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        tcpsock.bind((host,port))

        while True:
            tcpsock.listen(4)
            #print "nListening for incoming connections..."
            (clientsock, (ip, port)) = tcpsock.accept()

            run_client(ip, port, clientsock)
