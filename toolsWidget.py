#!/usr/bin/python3

import tkinter
from tkinter import *
from PIL import Image, ImageTk

from ontoWidget import OntoWidget


class ToolsWidget(tkinter.Listbox):

	def __init__(self, parent, *args, **kwargs):
		tkinter.Listbox.__init__(self, parent, *args, **kwargs)
		self.popup_menu = tkinter.Menu(self, tearoff=0)
		self.popup_menu.add_command(label="POS Sequence",
		                            command=self.pos_tagger)
		self.popup_menu.add_command(label="NER Sequence",
		                            command=self.ner_tagger)
		self.popup_menu.add_command(label="Parse Tree",
		                            command=self.dep_parser)
		self.popup_menu.add_command(label="Subsumption Pairs",
		                            command=self.onto_extractor)
		
		#NOTE event (left click, right click etc.) and function as arguments
		self.bind("<Button-3>", self.popup) 
		self.bind_all("<FocusOut>", self.focusOut)

	def popup(self, event):
		#NOTE if no text/tool is loaded, don't popup
		if not self.curselection():return
		if self.nlpprocesses.get("stash", False) is False:return
		try:
			#self.popup_menu.tk_popup(event.x_root, event.y_root, 0)
			self.popup_menu.post(event.x_root, event.y_root)
		finally:
			self.popup_menu.grab_release()

	def focusOut(self, event):
		self.popup_menu.unpost()

    #def delete_selected(self):
    #    for i in self.curselection()[::-1]:
    #        self.delete(i)

	#def select_all(self):
	#	self.selection_set(0, 'end')

	def pos_taggerSimple(self):
		if not self.nlpprocesses['tagging']:return
		sent_id = self.curselection()[0]
		toplevel = Toplevel()
		tagseq = Label(toplevel, text=self.nlpprocesses['tagging'][sent_id])
		tagseq.pack(expand=1)

	def pos_tagger(self):
		if not self.nlpprocesses['tagging']:return
		sent_id = self.curselection()
		if not sent_id: return
		else: sent_id = sent_id[0]
		toplevel = Toplevel()
		toplevel.focus_set()
		toplevel.grab_set() 
		toplevel.title("Part of Speech Tag Sequence")
		word_tag_seq = self.nlpprocesses['tagging'][sent_id]
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
		#if self.task != "parsing":return
		self.selection_set(0, 'end')

	def dep_parser(self):
		if not self.nlpprocesses["parsing"]:return
		sent_id = self.curselection()
		if not sent_id: return
		else: sent_id = sent_id[0]
		root = Toplevel()
		root.focus_set()
		root.grab_set() 
		root.title("Dependency Tree")
		treeLoader = Image.open(self.nlpprocesses['parsing'][sent_id][0])
		parseTree = ImageTk.PhotoImage(treeLoader)
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
		#if not self.nlpprocesses['ontorels']:return
		selection_id = self.curselection()
		#if not selection_id: return
		#content = ["tiger\tcat\t0.8\t0.8\tpositive", "car\tvehicle\t0.8\t0.8\tnegative", "tiger\tplant\t0.8\t0.8\tpositive"]
		content = ["frazil\tice type\t0.5\t0.8\tpositive", 
					"frazil\tgrease ice\t0.7\t0.8\tpositive", 
					"frazil ice\ttiny ice platelets\t0.55\t0.7\tpositive",
					"nilas\tice\t0.9\t0.6\tpositive"
					]
		ontoFrame = OntoWidget()
		ontoFrame.contentReader(content)
