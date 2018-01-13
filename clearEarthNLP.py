#!/usr/bin/python3

import os
import sys

import codecs
import shutil
#import posix1e
import tempfile
import tkinter as tk
from tkinter import *
from glob import glob
from functools import partial
from tkinter import ttk, font, filedialog
from tkinter.messagebox import showinfo, askokcancel, askyesno

from irtokz import RomanTokenizer

from utils import plotTree
from widgets.toolsWidget import ToolsWidget

from tools import parser
from tools.tagger import *
from tools.parser import *
from utils.keyPhraseExtraction import *
from tools.subsumptionExtractor import *


def printHelp():
   print ("No help yet!")

def runApplication():
    selectedTask = system_outputs.get()
    lbox.task = selectedTask
    lboxContent = lbox.get(0, END) #NOTE list of all items in listbox
    #NOTE load the file and select a task
    if (not selectedTask.strip()) or (not lboxContent):return
    if selectedTask == "parsing":
        if lbox.nlpprocesses['parsing']:return
        parsermodel = Parser(model='models/parser/clearnlp-parser')
        for sid, sentence in enumerate(lboxContent):
            if not sentence.strip():continue
            graph, ppos, Roott = parser.Test(parsermodel, sentence.strip().split())
            for node in xrange(len(graph)):
                graph[node] = graph[node]._replace(tag=ppos[node],
                                                    parent=graph[node].pparent,
                                                    drel=graph[node].pdrel.strip('%')
                                                       )
            nodes = [Roott]+graph+[Roott]
            dp_graph = plotTree.adjacencyMatrixplot(nodes)
            graph = plotTree.BFSPlot(nodes, dp_graph, 0)
            img_file = tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False)
            graph.write_png(img_file.name)
            lbox.nlpprocesses['parsing'][sid] = [img_file.name, nodes[1:-1]]
        del parsermodel
        lbox.nlpprocesses['stash'] = True
    elif selectedTask == "tagging":
        if lbox.nlpprocesses['tagging']:return
        if lbox.nlpprocesses['parsing']:
            for sent_id in lbox.nlpprocesses['parsing']:
                tags = [(node.form, node.tag) for node in lbox.nlpprocesses['parsing'][sent_id][1]]
                lbox.nlpprocesses['tagging'][sent_id] = tags
        else:
            tagger = Tagger(model='models/tagger/clearnlp-tagger')
            for sid, sentence in enumerate(lboxContent):
                if not sentence.strip():continue
                tags =  tagger.tag_sent(sentence.split())
                lbox.nlpprocesses['tagging'][sid] = list(tags)
            del tagger
        lbox.nlpprocesses['stash'] = True
    elif selectedTask == "nentity":
        if lbox.nlpprocesses['nentity']:return
        tagger = Tagger(model='models/ner/clearnlp-ner')
        for sid, sentence in enumerate(lboxContent):
            if not sentence.strip():continue
            tags =  tagger.tag_sent(sentence.split())
            lbox.nlpprocesses['nentity'][sid] = list(tags)
        del tagger
        lbox.nlpprocesses['stash'] = True
    elif selectedTask == "ontorels":
        if lbox.nlpprocesses['ontorels']:return
        ontoextractor = SubsumptionLearning(model='models/onto/clearnlp-onto')
        pairs = list(generatePairs(lboxContent))
        subsumptionRelations = list()
        for oid, (firstword, secondword) in enumerate(pairs,1):
            ooutput = ontoextractor.predict_hyp(firstword, secondword)
            if ooutput:
                reltype, confidence, distance = ontoextractor.predict_hyp(firstword, secondword)
                if (reltype == "Hypernym") and (distance >= 0.4):
                    subsumptionRelations.append([firstword, secondword, distance, confidence, 'positive'])
        del ontoextractor
        lbox.nlpprocesses['ontorels'] = subsumptionRelations
        lbox.nlpprocesses['stash'] = True
            
def contentReader(ifile):
    statusmsg.set('')
    #annotatemsg.set('')
    system_outputs.set('')
    lbox.nlpprocesses = {'tagging'  : {},
                         'nentity'  : {},
                         'parsing'  : {},
                         'ontorels' : {}}

    lbox.delete(0, END)
    for idx, line in enumerate(ifile):
        text = tok.tokenize(line)
        for tline in text.split("\n"):
            lbox.insert(tk.END, tline)

def contentWriter():
    lbox.nlpprocesses['stash'] = False
    base = lbox.file.split(".")[0]
    
    for nlproc, output in lbox.nlpprocesses.items():
        if (nlproc == 'stash') or (not output):continue
        if nlproc == "tagging":
            with open("%s.pos"%base, "w") as tfp:
                for sent_id, sentence in output.items():
                    for tid, tnode in enumerate(sentence, 1):
                        tfp.write("%s\t%s\t%s\n" % (tid, tnode[0], tnode[1]))
                    tfp.write("\n")
        elif nlproc == "parsing":
            with open("%s.parse"%base, "w") as pfp:
                for sent_id, sentence in output.items():
                    #NOTE sentence[0] = parse tree image, sentence[1] = conll namedtuple
                    for pnode in sentence[1]:
                        pN = pnode
                        pfp.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t_\t_\n" % \
                                    (str(pN.id), pN.form, pN.lemma, pN.ctag, pN.tag, pN.features, str(pN.parent), pN.drel))
                    pfp.write("\n")
        elif nlproc == "nentity":
            with open("%s.ner"%base, "w") as nfp:
                for sent_id, sentence in output.items():
                    for nid, nnode in enumerate(sentence, 1):
                        nfp.write("%s\t%s\t%s\n" % (nid, nnode[0], nnode[1]))
                    nfp.write("\n")
        else:
            with open("%s.onto"%base, "w") as ofp:
                for oid, relation_info in enumerate(output,1):
                    first, second, confidence, distance, relation = relation_info
                    ofp.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (oid, first, second, relation, confidence, distance))

def openFile():
    if lbox.nlpprocesses.get('stash', False) is True:
        what = askPopup()
        if what:
            contentWriter()
        else:
            lbox.nlpprocesses['stash'] = False
    else:
        if sys.platform.startswith('linux'):
            inputFile = filedialog.askopenfilename(parent=root, title='Choose a file', filetypes=[("all files", ".*")])
        else:
            inputFile = filedialog.askopenfilename(parent=root, title='Choose a file')
        lbox.file = inputFile
        if not inputFile:return
        with codecs.open(inputFile, encoding='utf-8') as ifp:
            contentReader(ifp)

def askPopup():
    #what = askokcancel(message="Would you like to save the system outputs?", title="Save the outputs?")
    what = askyesno(message=message, title="Save the outputs?")
    return what

def Quit():
    if lbox.nlpprocesses.get('stash', False) is True:
        #what = askokcancel(message="Would you like to save the system outputs?", title="Save the outputs?")
        what = askyesno(message=message, title="Save the outputs?")
        if what:
            contentWriter()
        else:
            root.quit()
    else:
        root.quit()

if __name__ == "__main__":
    tok = RomanTokenizer(split_sen=True)
    root = Tk()
    style = ttk.Style()
    root.wm_title("Clear Earth NLP Toolkit")
    message = "Would you like to save the system outputs?"
    root.option_add("*Dialog.msg.wrapLength", font.Font().measure(message))

    if sys.platform.startswith("linux"):
        current_path = os.path.dirname(os.path.realpath(__file__))
        #root.wm_iconbitmap('@%s/favicon.xbm' % current_path)
        #img = PhotoImage(file='%s/favicon.png' % current_path)
        img = PhotoImage(file='%s/misc/icon.png' % current_path)
        #root.tk.call('wm', 'iconphoto', root._w, img)
        root.iconphoto(True, img)
        #root.config(background="Gray")
    else: pass

    root.minsize(width=800, height=400)
    
    framefont = font.Font(size=11)
    itemfont = font.Font(size=11, weight=font.BOLD)
    root.option_add("*Font", framefont)
    
    # State variables
    statusmsg = StringVar()
    annotatemsg = StringVar()
    system_outputs = StringVar()
    
    # Create and grid the outer content frame
    frame = ttk.Frame(root, padding=(6, 6, 12, 6))
    frame.option_add("*Font", framefont)
    frame.grid(column=0, row=0, sticky=(N,W,E,S))
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0,weight=1)
    
    lbox = ToolsWidget(frame, height=10, width=40, font=("MS Serif", 12))
    vsb = ttk.Scrollbar(orient="vertical", command=lbox.yview)
    hsb = ttk.Scrollbar(orient="horizontal", command=lbox.xview)
    vsb.grid(column=2, row=0, rowspan=7, sticky=(N,S,E,W), in_=frame)
    hsb.grid(column=0, row=7, sticky=(N,S,E,W), in_=frame)
    lbox.config(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    lbox.nlpprocesses = dict()
    lbox.grid(column=0, row=0, rowspan=7,sticky=(N,S,E,W))

    label = ttk.Label(frame, text="NLP Tools:")
    
    pos = ttk.Radiobutton(frame, text='POS Tagging', variable=system_outputs, value='tagging')
    rev = ttk.Radiobutton(frame, text='NER Tagging', variable=system_outputs, value='nentity')
    rel = ttk.Radiobutton(frame, text='Dependency Parsing', variable=system_outputs, value='parsing')
    neg = ttk.Radiobutton(frame, text='Relation Extraction', variable=system_outputs, value='ontorels')
    run = ttk.Button(frame, text='Run', command=runApplication, default='active')
    
    menubar=Menu(root)
    
    menu=Menu(menubar,tearoff=0)#, postcommand=refresh)
    
    menu.add_separator()
    menu.add_command(label="Open", command=openFile)
    menu.add_command(label="Exit", command=Quit)
    menubar.add_cascade(label="File", menu=menu)
    
    helpmenu=Menu(menubar,tearoff=0)
    helpmenu.add_command(label="Help",command=printHelp)
    menubar.add_cascade(label="Help",menu=helpmenu)
    root.config(menu=menubar)
    
    # Grid all the widgets
    lbox.grid(column=0, row=0, rowspan=7, sticky=(N,S,E,W))
    label.grid(column=3, row=0, padx=10, pady=5)
    pos.grid(column=3, row=1, sticky=W, padx=20)
    rev.grid(column=3, row=2, sticky=W, padx=20)
    rel.grid(column=3, row=3, sticky=W, padx=20)
    neg.grid(column=3, row=4, sticky=W, padx=20)
    run.grid(column=3, row=5, sticky=(N,W), padx=20, pady=20)
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_rowconfigure(5, weight=1)
    
    lbox.selection_set(0)
    ttk.Sizegrip(root).grid(column=999, row=999, sticky=(S,E))
    root.protocol("WM_DELETE_WINDOW", Quit)
    root.mainloop()

    for filename in glob("/tmp/*.png"):
        os.remove(filename)
