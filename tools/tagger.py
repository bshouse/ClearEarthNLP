from __future__ import unicode_literals

import io
import re
import sys
import math
import string
import random
import pickle
from argparse import ArgumentParser
from collections import Counter, defaultdict

import dynet as dy
import numpy as np
from gensim.models.word2vec import Word2Vec

class Meta:
    def __init__(self):
        self.c_dim = 32  # character-rnn input dimension
        self.add_words = 1  # additional lookup for missing/special words
        self.n_hidden = 64  # pos-mlp hidden layer dimension
        self.lstm_char_dim = 32  # char-LSTM output dimension
        self.lstm_word_dim = 64  # LSTM (word-char concatenated input) output dimension


class Tagger():
    def __init__(self, model=None, meta=None):
        self.model = dy.Model()
        if model:
            self.meta = pickle.load(open('%s.meta' %model, 'rb'))
        else:
            self.meta = meta
        self.WORDS_LOOKUP = self.model.add_lookup_parameters((self.meta.n_words, self.meta.w_dim))
        if not model:
            for word, V in wvm.vocab.iteritems():
                self.WORDS_LOOKUP.init_row(V.index+self.meta.add_words, wvm.syn0[V.index])

        self.CHARS_LOOKUP = self.model.add_lookup_parameters((self.meta.n_chars, self.meta.c_dim))

        # MLP on top of biLSTM outputs 100 -> 32 -> ntags
        self.W1 = self.model.add_parameters((self.meta.n_hidden, self.meta.lstm_word_dim*2))
        self.W2 = self.model.add_parameters((self.meta.n_tags, self.meta.n_hidden))
        self.B1 = self.model.add_parameters(self.meta.n_hidden)
        self.B2 = self.model.add_parameters(self.meta.n_tags)

        # word-level LSTMs
        self.fwdRNN = dy.LSTMBuilder(1, self.meta.w_dim+self.meta.lstm_char_dim*2, self.meta.lstm_word_dim, self.model) 
        self.bwdRNN = dy.LSTMBuilder(1, self.meta.w_dim+self.meta.lstm_char_dim*2, self.meta.lstm_word_dim, self.model)
        self.fwdRNN2 = dy.LSTMBuilder(1, self.meta.lstm_word_dim*2, self.meta.lstm_word_dim, self.model) 
        self.bwdRNN2 = dy.LSTMBuilder(1, self.meta.lstm_word_dim*2, self.meta.lstm_word_dim, self.model)

        # char-level LSTMs
        self.cfwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.lstm_char_dim, self.model)
        self.cbwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.lstm_char_dim, self.model)
        if model:
            self.model.populate('%s.dy' %model)

    def word_rep(self, word):
        if not self.eval and random.random() < 0.25:
            return self.WORDS_LOOKUP[0]
        idx = self.meta.w2i.get(word, self.meta.w2i.get(word.lower(), 0))
        return self.WORDS_LOOKUP[idx]
    
    def char_rep(self, w, f, b):
        bos, eos, unk = self.meta.c2i["bos"], self.meta.c2i["eos"], self.meta.c2i["unk"]
        char_ids = [bos] + [self.meta.c2i[c] if self.meta.cc[c]>5 else unk for c in w] + [eos]
        char_embs = [self.CHARS_LOOKUP[cid] for cid in char_ids]
        fw_exps = f.transduce(char_embs)
        bw_exps = b.transduce(reversed(char_embs))
        return dy.concatenate([ fw_exps[-1], bw_exps[-1] ])

    def enable_dropout(self):
        self.fwdRNN.set_dropout(0.3)
        self.bwdRNN.set_dropout(0.3)
        self.fwdRNN2.set_dropout(0.3)
        self.bwdRNN2.set_dropout(0.3)
        self.cfwdRNN.set_dropout(0.3)
        self.cbwdRNN.set_dropout(0.3)
        self.w1 = dy.dropout(self.w1, 0.3)
        self.b1 = dy.dropout(self.b1, 0.3)

    def disable_dropout(self):
        self.fwdRNN.disable_dropout()
        self.bwdRNN.disable_dropout()
        self.fwdRNN2.disable_dropout()
        self.bwdRNN2.disable_dropout()
        self.cfwdRNN.disable_dropout()
        self.cbwdRNN.disable_dropout()

    def build_tagging_graph(self, words):
        dy.renew_cg()
        # parameters -> expressions
        self.w1 = dy.parameter(self.W1)
        self.b1 = dy.parameter(self.B1)
        self.w2 = dy.parameter(self.W2)
        self.b2 = dy.parameter(self.B2)

        # apply dropout
        if self.eval:
            self.disable_dropout()
        else:
            self.enable_dropout()

        # initialize the RNNs
        f_init = self.fwdRNN.initial_state()
        b_init = self.bwdRNN.initial_state()
        f2_init = self.fwdRNN2.initial_state()
        b2_init = self.bwdRNN2.initial_state()
    
        self.cf_init = self.cfwdRNN.initial_state()
        self.cb_init = self.cbwdRNN.initial_state()
    
        # get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
        wembs = [self.word_rep(w) for w in words]
        cembs = [self.char_rep(w, self.cf_init, self.cb_init) for w in words]
        xembs = [dy.concatenate([w, c]) for w,c in zip(wembs, cembs)]
    
        # feed word vectors into biLSTM
        fw_exps = f_init.transduce(xembs)
        bw_exps = b_init.transduce(reversed(xembs))
    
        # biLSTM states
        bi_exps = [dy.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps))]

        # feed word vectors into biLSTM
        fw_exps = f2_init.transduce(bi_exps)
        bw_exps = b2_init.transduce(reversed(bi_exps))
    
        # biLSTM states
        bi_exps = [dy.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps))]
    
        # feed each biLSTM state to an MLP
        exps = []
        for xi in bi_exps:
            xh = self.w1 * xi
            xo = self.w2 * (dy.tanh(xh) + self.b1) + self.b2
            exps.append(xo)
    
        return exps
    
    def sent_loss(self, words, tags):
        self.eval = False
        vecs = self.build_tagging_graph(words)
        errs = []
        for v,t in zip(vecs,tags):
            tid = self.meta.t2i[t]
            err = dy.pickneglogsoftmax(v, tid)
            errs.append(err)
        return dy.esum(errs)
    
    def tag_sent(self, words):
        self.eval = True
        vecs = self.build_tagging_graph(words)
        vecs = [dy.softmax(v) for v in vecs]
        probs = [v.npvalue() for v in vecs]
        tags = []
        for prb in probs:
            tag = np.argmax(prb)
            tags.append(self.meta.i2t[tag])
        return zip(words, tags)

def read(fname):
    data = []
    sent = []
    pid = 3 if args.pos else 2
    fp = io.open(fname, encoding='utf-8')
    for i,line in enumerate(fp):
        line = line.split()
        if not line:
            data.append(sent)
            sent = []
        else:
            w,p = line[1], line[pid]
            sent.append((w,p))
    if sent: data.append(sent)
    return data

def eval(dev, ofile=None):
    good_sent = bad_sent = good = bad = 0.0
    gall, pall = [], []
    #ofp = open(ofile, 'w')
    for sent in dev:
        words, golds = zip(*sent)
        tags = [t for w,t in tagger.tag_sent(words)]
        #ofp.write('\n'.join(tags)+'\n\n')
        #pall.extend(tags)
        if list(tags) == list(golds): good_sent += 1
        else: bad_sent += 1
        for go,gu in zip(golds,tags):
            if go == gu: good += 1
            else: bad += 1
    #print(cr(gall, pall, digits=4))
    print(good/(good+bad), good_sent/(good_sent+bad_sent))
    return good/(good+bad)

def train_tagger(train):
    pr_acc = 0.0
    num_tagged, cum_loss = 0, 0
    for ITER in xrange(args.iter):
        save = False
        random.shuffle(train)
        for i,s in enumerate(train, 1):
            if i > 0 and i % 500 == 0:   # print status
                trainer.status()
                print(cum_loss / num_tagged)
                cum_loss, num_tagged = 0, 0
            words, golds = zip(*s)
            loss_exp =  tagger.sent_loss(words, golds)
            cum_loss += loss_exp.scalar_value()
            num_tagged += len(golds)
            loss_exp.backward()
            trainer.update()
        print("epoch %r finished" % ITER)
        new_acc = eval(dev)
        if new_acc > pr_acc:
            pr_acc = new_acc
            save = True
        if save:
            print('Save Point:: %d' %ITER)
            if args.save_model:
                tagger.model.save('%s.dy' %args.save_model)
            save = False
        sys.stdout.flush()

def get_cc(data):
    tags, chars = set(), set()
    meta.cc = Counter()
    for sent in data:
        for w,p in sent:
            tags.add(p)
            chars.update(w)
            meta.cc.update(w)
    chars.update(['unk', 'bos', 'eos'])
    meta.n_chars = len(chars)
    meta.c2i = dict(zip(chars, range(meta.n_chars)))
    meta.n_tags = len(tags)
    meta.i2t = dict(enumerate(tags))
    meta.t2i = {t:i for i,t in meta.i2t.items()}

if __name__ == '__main__':
    parser = ArgumentParser(description="Syntactic Tagger")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('--dynet-mem')
    parser.add_argument('--dynet-gpu')
    parser.add_argument('--dynet-autobatch')
    parser.add_argument('--dynet-seed', dest='seed', type=int)
    parser.add_argument('--lang')
    parser.add_argument('--train')
    parser.add_argument('--dev')
    parser.add_argument('--embd')
    parser.add_argument('--trainer')
    parser.add_argument('--pos', type=int)
    parser.add_argument('--iter', type=int, default=500)
    parser.add_argument('--evec', type=int)
    group.add_argument('--save-model', dest='save_model')
    group.add_argument('--load-model', dest='load_model')
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    meta = Meta()
    if args.dev:
        dev = read(args.dev)
    if not args.load_model: 
        train = read(args.train)
        wvm = Word2Vec.load_word2vec_format(args.embd, binary=args.evec)
        meta.w_dim = wvm.syn0.shape[1]
        meta.n_words = wvm.syn0.shape[0]+meta.add_words
        
        get_cc(train)
        meta.w2i = {}
        for w in wvm.vocab:
            meta.w2i[w] = wvm.vocab[w].index + meta.add_words
    
    if args.save_model:
        pickle.dump(meta, open('%s.meta' %args.save_model, 'wb'))
    if args.load_model:
        tagger = Tagger(model=args.load_model)
        eval(dev) 
    else:
        tagger = Tagger(meta=meta)
        trainer = dy.MomentumSGDTrainer(tagger.model)
        train_tagger(train)
