#!/usr/bin/python3

import os
import sys

import shutil
#import posix1e
import tkinter as tk
from tkinter import *
from tkinter import ttk, font
from functools import partial
from tkinter.messagebox import showinfo, askokcancel


class OntoWidget(object):
	def __init__(self, content=dict()):
		self.content = content
		self.root = Toplevel()
		self.root.focus_set() 
		self.root.grab_set() 
		self.style = ttk.Style()
		self.root.wm_title("Subsumption Relations")
		self.root.minsize(width=800, height=400)
    	
		self.framefont = font.Font(size=11)
		self.itemfont = font.Font(size=11, weight=font.BOLD)
		self.root.option_add("*Font", self.framefont)
    	
		# State variables
		self.statusmsg = StringVar()
		self.annotatemsg = StringVar()
		self.system_outputs = StringVar()
    	
		# Create and grid the outer content frame
		self.frame = ttk.Frame(self.root, padding=(6, 6, 12, 6))
		self.frame.option_add("*Font", self.framefont)
		self.frame.grid(column=0, row=0, sticky=(N,W,E,S))
		self.root.grid_columnconfigure(0, weight=1)
		self.root.grid_rowconfigure(0, weight=1)
    	
		self.tree = ttk.Treeview(self.frame, columns=['Hyponym', 'Hypernym'], show="headings", selectmode="extended")
		self.vsb = ttk.Scrollbar(self.frame, orient="vertical", command=self.tree.yview)
		self.hsb = ttk.Scrollbar(self.frame, orient="horizontal", command=self.tree.xview)
		#vsb.grid(column=2, row=0, rowspan=6, sticky=(N,S,E,W), in_=frame)
		#hsb.grid(column=0, row=6, sticky=(N,S,E,W), in_=frame)
		self.vsb.grid(column=2, row=0, rowspan=6, sticky=(N,S,E,W), in_=self.frame)
		self.hsb.grid(column=0, row=6, sticky=(N,S,E,W), in_=self.frame)
		#self.hsb.pack(side=BOTTOM, fill='x')
		#self.vsb.pack(side=BOTTOM, fill='y')
		self.tree.config(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)
		self.tree.grid(column=0, row=0, rowspan=6, sticky=(N,S,E,W))
		#self.tree.pack()
    	
		for col in ['Hyponym', 'Hypernym']:
			self.tree.heading(col, text=col)
			self.tree.column(col, width=font.Font().measure(col))
    	
		self.style.configure("Treeview.Heading", background="snow4", foreground='black', font=("Times", 12), weight=font.BOLD)
		#font.nametofont('TkHeadingFont').configure(size=12, weight=font.BOLD)
    	
		self.label = ttk.Label(self.frame, text="Modify System output:")
    	
		self.pos = ttk.Radiobutton(self.frame, text='Hyponym-hypernym', variable=self.system_outputs, value='positive')
		self.rev = ttk.Radiobutton(self.frame, text='Hypernym-hyponym', variable=self.system_outputs, value='reverse')
		self.rel = ttk.Radiobutton(self.frame, text='Related Concepts', variable=self.system_outputs, value='related')
		self.neg = ttk.Radiobutton(self.frame, text='None of the Above', variable=self.system_outputs, value='negative')
		self.modify = ttk.Button(self.frame, text='Modify', command=self.modifyAnnotation, default='active')
		self.move = ttk.Button(self.frame, text='Next', command=self.nextSelection, default='active')
		self.annotate = ttk.Label(self.frame, textvariable=self.annotatemsg)
		self.status = ttk.Label(self.frame, textvariable=self.statusmsg, anchor=W, 
    	            foreground='DarkOrchid4', font=("Roman", 11, "bold"), justify="center")
    	
		# Grid all the widgets
		self.label.grid(column=3, row=0, padx=10, pady=5)
		self.pos.grid(column=3, row=1, sticky=W, padx=20)
		self.rev.grid(column=3, row=2, sticky=W, padx=20)
		self.rel.grid(column=3, row=3, sticky=W, padx=20)
		self.neg.grid(column=3, row=4, sticky=W, padx=20)
		self.modify.grid(column=3, row=5, sticky=(N,W), padx=20, pady=20)
		self.move.grid(column=3, row=5, sticky=(N,E), padx=120, pady=20)
		#annotate.grid(column=1, row=6, columnspan=2, sticky=(N,W))
		self.status.grid(column=0, row=7, columnspan=2, pady=5, padx=(50,50))
		self.frame.grid_columnconfigure(0, weight=1)
		self.frame.grid_rowconfigure(5, weight=1)
    	
		# Set event bindings for when the selection in the listbox changes,
		self.tree.bind('<<TreeviewSelect>>', self.showInfo)
    	
		#showInfo()
		ttk.Sizegrip(self.root).grid(column=999, row=999, sticky=(S,E))
		#self.root.protocol("WM_DELETE_WINDOW", self.Quit)

	def contentReader(self, ontorelations):
	    self.statusmsg.set('')
	    #annotatemsg.set('')
	    self.system_outputs.set('')
	    self.content = dict()
	
	    self.status.config(relief="raised")
	    self.tree.delete(*self.tree.get_children())
	    for idx, item in enumerate(ontorelations):
	        phrase1,phrase2,similarity,confidence,relation = item.strip().split("\t")
	        phrase_pair = "\t".join([phrase1, phrase2])
	        self.content[phrase_pair] = [idx, similarity, confidence, relation]
	        self.tree.insert('', 'end', iid='%s'%idx, values=(phrase1, phrase2), tags='%s'%(idx))
	        if len(relation.split("#")) > 1:
	            if relation.split("#")[-1] == "modified":
	                self.tree.tag_configure('%s'%(idx), foreground="red", font=self.itemfont)
	            else:
	                self.tree.tag_configure('%s'%(idx), foreground="blue", font=self.itemfont)
	        else:
	            self.tree.tag_configure('%s'%(idx), foreground="black", font=self.itemfont)
	
	    if self.tree.get_children():
	        self.tree.selection_set('"0"')
	        #tree.focus_set()
	        firstItem = "\t".join(self.tree.item('0')['values'])
	        self.system_outputs.set(self.content[firstItem][-1].split("#")[0])
	        self.tree.focus('0')
	    else:
	        self.system_outputs.set('')
	
	#def saveAnnotations(self):
	#    if self.content.get("MoDiFiEd", False) == False: return
	#
	#    currentFileUser = os.path.join(SINKDIR, "%s.tsv.%s" % (currentFile, USER))
	#    self.content.pop("MoDiFiEd")
	#    with open(currentFileUser, "w") as ofp:
	#        for key, value in sorted(self.content.items(), key=lambda x: x[1][0]):
	#            key = "%s\t%s\t%s\t%s" % (key,value[1], value[2], value[3])
	#            ofp.write("%s\n"%key)
	#    self.content['MoDiFiEd'] = False
	            
	def showInfo(self, *args):
	    idx = self.tree.focus()
	    if idx != '':
	        instance = "\t".join(self.tree.item(idx)['values'])
	        currentItem = self.content[instance]
	        self.statusmsg.set("Similarity for '%s' is: '%s' and system confidence is: '%s'" % 
	                                (instance.upper(), currentItem[1], currentItem[2]))
	        self.system_outputs.set(currentItem[-1].split("#")[0])
	
	def modifyAnnotation(self):
	    selection_indices = self.tree.focus()
	
	    # make sure at least one item is selected
	    if selection_indices != '':
	        last_selection = selection_indices
	   
	        #instance = lbox.get(last_selection).strip()
	        instance = "\t".join(self.tree.item(last_selection)['values'])
	        index = self.content[instance][0]
	        previousOutput = self.content[instance][-1]
	        systemOutput = self.system_outputs.get()
	
	        #self.content[phrase_pair][-1] = "%s#modified" % (systemOutput)
	        lenPrev = len(previousOutput.split("#"))
	        previousOutput = previousOutput.split("#")[0]
	        if systemOutput == previousOutput:
	            if lenPrev == 1: self.content[instance][-1] = "%s#unchanged" % (systemOutput)
	            #annotatemsg.set("System label unchanged for '%s'!" % (instance))
	            self.tree.tag_configure('%s'%(index), foreground="blue")
	            #lbox.itemconfig(index, background="green")
	        else:
	            #annotatemsg.set("System label modified for '%s'!" % (instance))
	            self.content[instance][-1] = "%s#modified" % (systemOutput)
	            self.tree.tag_configure('%s'%(index), foreground="red")
	        self.content['MoDiFiEd'] = True
	
	def nextSelection(self):
	    current_selection = self.tree.focus()
	
	    # default next selection is the beginning
	    next_selection = 0
	
	    # make sure at least one item is selected
	    if current_selection != '':
	        last_selection = int(current_selection)
	
	        # Make sure we're not at the last item
	        if last_selection < len(self.tree.get_children()) - 1:
	            next_selection = last_selection + 1
	            #annotatemsg.set('')
	    
	    if not self.tree.get_children(): return
	    self.tree.selection_set('"%s"' % (next_selection))
	    self.tree.focus("%s" % (next_selection))
	    instance = "\t".join(self.tree.item('%s'%next_selection)['values'])
	    nextItem = self.content[instance]
	    self.statusmsg.set("Similarity for '%s' is: '%s' and system confidence is: '%s'" % 
	                            (instance.upper(), nextItem[1], nextItem[2]))
	    self.system_outputs.set(nextItem[-1].split("#")[0])
	
	def askPopup(self):
	    what = askokcancel(message="You have not saved the changes to the current file. Opening a new file will \
	                                cancel those changes.", title="Save the changes?")
	    return what
	
	def Quit(self):
	    if self.content.get("MoDiFiEd", False):
	        what = askokcancel(message="You have not saved the changes to the current file. Closing the window will \
	                                    cancel those changes.", title="Save the changes?")
	        if what:
	            pass
	        else:
	            self.root.quit()
	    else:
	        self.root.quit()

if __name__ == "__main__":
	s=OntoWidget()
	s.root.mainloop()
