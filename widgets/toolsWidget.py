#!/usr/bin/python3

import sys
import tkinter
from tkinter import *
from PIL import Image, ImageTk

from widgets.ontoWidget import OntoWidget


class ToolsWidget(tkinter.Listbox):

    def __init__(self, parent, *args, **kwargs):
        tkinter.Listbox.__init__(self, parent, *args, **kwargs)
        self.popup_menu = tkinter.Menu(self, tearoff=0)
       
        self.flags = {'tagging':False, 'ner':False, 'parsing':False, 'onto':False} 
        #NOTE event (left click, right click etc.) and function as arguments
        if sys.platform.startswith("linux"):
            self.bind("<Button-3>", self.popup)
        else:
            self.bind("<Button-2>", self.popup)
        self.bind_all("<FocusOut>", self.focusOut)

    def add_popup_menu(self, task):
        if self.flags[task]: return
        if task == 'tagging':
            self.popup_menu.add_command(label="POS Sequence",
                                            command=self.pos_tagger)
            self.flags[task] = True
        elif task == 'ner':
            
            self.popup_menu.add_command(label="NER Sequence",
                                        command=self.ner_tagger)
            self.flags[task] = True
        elif task == 'parsing':
            self.popup_menu.add_command(label="Parse Tree",
                                        command=self.dep_parser)
            self.flags[task] = True
        elif task == 'onto':
            self.popup_menu.add_command(label="Subsumption Pairs",
                                        command=self.onto_extractor)
            self.flags[task] = True
        else: return
            
    def popup(self, event):
        #NOTE if no text/tool is loaded, don't popup
        if not self.curselection():return
        if self.nlpprocesses.get("stash", False) is False:return
        if self.nlpprocesses['tagging']: self.add_popup_menu('tagging')
        if self.nlpprocesses['nentity']: self.add_popup_menu('ner')
        if self.nlpprocesses['parsing']: self.add_popup_menu('parsing')
        if self.nlpprocesses['ontorels']: self.add_popup_menu('onto')
        try:
            #self.popup_menu.tk_popup(event.x_root, event.y_root, 0)
            self.popup_menu.post(event.x_root, event.y_root)
        finally:
            self.popup_menu.grab_release()

    def focusOut(self, event):
        self.popup_menu.unpost()

    def pos_tagger(self):
        if not self.nlpprocesses['tagging']:return
        sent_id = self.curselection()
        if not sent_id: return
        else: sent_id = sent_id[0]
        word_tag_seq = self.nlpprocesses['tagging'][sent_id]
        if not word_tag_seq: return
        toplevel = Toplevel()
        toplevel.focus_set()
        toplevel.grab_set() 
        toplevel.title("Part of Speech Tag Sequence")
        words = [w[0] for w in word_tag_seq]
        tags = [w[1] for w in word_tag_seq]
        maxstrings = [word if len(word) >= len(tag) else tag for (word, tag) in word_tag_seq]
        tree = ttk.Treeview(toplevel, columns=maxstrings, selectmode='browse')

        for maxstring in maxstrings:
            tree.heading(maxstring)
            tree.column(maxstring, width=font.Font().measure(maxstring))

        #tree.column("#0", width=0)
        tree['show'] = 'headings'
        #vsb = ttk.Scrollbar(orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(toplevel, orient="horizontal", command=tree.xview)
        #vsb.grid(column=len(word_tag_seq), row=0, rowspan=3, sticky=(N,S,E,W), in_=toplevel)
        #hsb.grid(column=0, row=3, sticky=(N,S,E,W))
        hsb.pack(side=BOTTOM, fill='x')
        tree.config(xscrollcommand=hsb.set)
        #tree.config(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.insert('', 'end',values=words)
        tree.insert('', 'end',values=tags)
        tree.pack(fill=BOTH, expand=True)
    
    def ner_tagger(self):
        if not self.nlpprocesses['nentity']:return
        sent_id = self.curselection()
        if not sent_id: return
        else: sent_id = sent_id[0]
        word_tag_seq = self.nlpprocesses['nentity'][sent_id]
        if not word_tag_seq: return
        toplevel = Toplevel()
        toplevel.focus_set()
        toplevel.grab_set() 
        toplevel.title("Named-entity Tag Sequence")
        words = [w[0] for w in word_tag_seq]
        tags = [w[1] for w in word_tag_seq]
        maxstrings = [word if len(word) >= len(tag) else tag for (word, tag) in word_tag_seq]
        tree = ttk.Treeview(toplevel, columns=maxstrings, selectmode='browse')

        for maxstring in maxstrings:
            tree.heading(maxstring)
            tree.column(maxstring, width=font.Font().measure(maxstring))

        #tree.column("#0", width=0)
        tree['show'] = 'headings'
        #vsb = ttk.Scrollbar(orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(toplevel, orient="horizontal", command=tree.xview)
        #vsb.grid(column=len(word_tag_seq), row=0, rowspan=3, sticky=(N,S,E,W), in_=toplevel)
        #hsb.grid(column=0, row=3, sticky=(N,S,E,W))
        hsb.pack(side=BOTTOM, fill='x')
        tree.config(xscrollcommand=hsb.set)
        #tree.config(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.insert('', 'end',values=words)
        tree.insert('', 'end',values=tags)
        tree.pack(fill=BOTH, expand=True)
    
    def dep_parser(self):
        if not self.nlpprocesses["parsing"]:return
        sent_id = self.curselection()
        if not sent_id: return
        else: sent_id = sent_id[0]
        dtree = self.nlpprocesses['parsing'][sent_id]
        if not dtree: return
        root = Toplevel()
        root.focus_set()
        root.grab_set() 
        root.title("Dependency Tree")
        treeLoader = Image.open(dtree[0])
        parseTree = ImageTk.PhotoImage(treeLoader.convert("RGB"))
        canvas = tkinter.Canvas(root, borderwidth=0, 
                                      background="#ffffff", 
                                      highlightthickness=0, 
                                      width=treeLoader.size[0], 
                                      height=treeLoader.size[1]
                                )
        root.config(bg="white")
        vsb = tkinter.Scrollbar(root, orient="vertical", command=canvas.yview)
        hsb = tkinter.Scrollbar(root, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        hsb.pack(side=BOTTOM, fill="x")
        vsb.pack(side="right", fill="y")
        canvas.create_image(0, 0, anchor=tkinter.NW, image=parseTree)
        canvas.image = parseTree #NOTE important to show image
        canvas.pack()

        def scrollForAll(canvas):
            '''Reset the scroll region to encompass the inner frame'''
            canvas.configure(scrollregion=canvas.bbox("all"))
        root.bind("<Configure>", lambda event, canvas=canvas: scrollForAll(canvas))

    def onto_extractor(self):
        if not self.nlpprocesses['ontorels']:return
        selection_id = self.curselection()
        if not selection_id: return
        ontoFrame = OntoWidget(self.nlpprocesses)
        ontoFrame.contentReader()
