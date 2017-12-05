#!/usr/local/python3/bin/python3


"""Crappy code! Need to write it well and organize properly.
"""

import os
import sys

import shutil
#import posix1e
import tkinter as tk
from tkinter import *
from tkinter import ttk, font
from functools import partial
from tkinter.messagebox import showinfo, askokcancel

def printHelp():
   print ("No help yet!")

def refresh():
    menu.delete(0, END)
    updateFileMenu(TASKDIR, SINKDIR, menu)
    menu.add_separator()
    menu.add_command(label="Save", command=saveAnnotations)
    menu.add_command(label="Exit", command=Quit)

def updateFileMenu(raw, sink, menu):
    sink = [(sink, sfile) for sfile in os.listdir(sink)]
    taken = set([tfile.split(".")[0] for _, tfile in sink])
    raw = [(raw, rfile) for rfile in os.listdir(raw) if rfile.split(".")[0] not in taken]
    
    for rsource, rfile in raw:
        rfilepath = os.path.join(rsource, rfile)
        menu.add_command(label=rfile, command=partial(fileReader, rfilepath))
    
    for ssource, sfile in sink:
        username = sfile.split(".")[-1]
        if username == USER:
            sfilepath = os.path.join(ssource, sfile)
            menu.add_command(label=sfile, command=partial(fileReader, sfilepath))

def fileReader(ifile):
    global currentFile, currentContent
    if currentContent.get('MoDiFiEd', False):
        what = askPopup()
        if what == True:
            return
        else:
            pass
    else:
        pass

    statusmsg.set('')
    #annotatemsg.set('')
    system_outputs.set('')
    currentContent = dict()
    basename = os.path.basename(ifile)
    currentFile = basename.split(".")[0]
    if basename.endswith(".tsv"):
        ofile = os.path.join(SINKDIR, "%s.%s" % (basename, USER))
        shutil.copy(ifile, ofile)
        #ofp = open(ofile, 'w')
        #ofp.close()

    status.config(relief="raised")
    with open(ifile) as ifp:
        #lbox.delete(0, END)
        tree.delete(*tree.get_children())
        for idx, item in enumerate(ifp):
            phrase1,phrase2,similarity,confidence,relation = item.strip().split("\t")
            phrase_pair = "\t".join([phrase1, phrase2])
            currentContent[phrase_pair] = [idx, similarity, confidence, relation]
            #lbox.insert(tk.END, phrase_pair.strip())
            #tree.insert('', 'end', iid='%s'%idx, values=(phrase1, phrase2), tags='%s%s'%(currentFile,idx))
            tree.insert('', 'end', iid='%s'%idx, values=(phrase1, phrase2), tags='%s'%(idx))
            if len(relation.split("#")) > 1:
                if relation.split("#")[-1] == "modified":
                    #lbox.itemconfig(idx, background="red")
                    #tree.tag_configure('%s%s'%(currentFile,idx), background="red")
                    tree.tag_configure('%s'%(idx), foreground="red", font=itemfont)
                else:
                    #lbox.itemconfig(idx, background="green")
                    #tree.tag_configure('%s%s'%(currentFile,idx), background="green")
                    tree.tag_configure('%s'%(idx), foreground="blue", font=itemfont)
            else:
                tree.tag_configure('%s'%(idx), foreground="black", font=itemfont)
                #tree.tag_configure('%s%s'%(currentFile,idx), background="green")
            #if basename.endswith(".txt"):ofp.write(item)

    if tree.get_children():
        tree.selection_set('"0"')
        #tree.focus_set()
        firstItem = "\t".join(tree.item('0')['values'])
        system_outputs.set(currentContent[firstItem][-1].split("#")[0])
        tree.focus('0')
    else:
        system_outputs.set('')
    #if basename.endswith(".txt"): ofp.close()

def saveAnnotations():
    global currentContent
    if currentContent.get("MoDiFiEd", False) == False: return

    currentFileUser = os.path.join(SINKDIR, "%s.tsv.%s" % (currentFile, USER))
    currentContent.pop("MoDiFiEd")
    with open(currentFileUser, "w") as ofp:
        for key, value in sorted(currentContent.items(), key=lambda x: x[1][0]):
            key = "%s\t%s\t%s\t%s" % (key,value[1], value[2], value[3])
            ofp.write("%s\n"%key)
    currentContent['MoDiFiEd'] = False
            
def showInfo(*args):
    #idxs = lbox.curselection()
    idx = tree.focus()
    if idx != '':
        #currentItem = currentContent[lbox.get(idx)]
        instance = "\t".join(tree.item(idx)['values'])
        currentItem = currentContent[instance]
        statusmsg.set("Similarity for '%s' is: '%s' and system confidence is: '%s'" % 
                                (instance.upper(), currentItem[1], currentItem[2]))
        system_outputs.set(currentItem[-1].split("#")[0])

def modifyAnnotation():
    global currentContent
    selection_indices = tree.focus()

    # make sure at least one item is selected
    if selection_indices != '':
        last_selection = selection_indices
   
        #instance = lbox.get(last_selection).strip()
        instance = "\t".join(tree.item(last_selection)['values'])
        index = currentContent[instance][0]
        previousOutput = currentContent[instance][-1]
        systemOutput = system_outputs.get()

        #currentContent[phrase_pair][-1] = "%s#modified" % (systemOutput)
        lenPrev = len(previousOutput.split("#"))
        previousOutput = previousOutput.split("#")[0]
        if systemOutput == previousOutput:
            if lenPrev == 1: currentContent[instance][-1] = "%s#unchanged" % (systemOutput)
            #annotatemsg.set("System label unchanged for '%s'!" % (instance))
            #tree.tag_configure('%s%s'%(currentFile,index), background="green")
            tree.tag_configure('%s'%(index), foreground="blue")
            #lbox.itemconfig(index, background="green")
        else:
            #annotatemsg.set("System label modified for '%s'!" % (instance))
            currentContent[instance][-1] = "%s#modified" % (systemOutput)
            #lbox.itemconfig(index, background="red")
            #tree.tag_configure('%s%s'%(currentFile,index), background="red")
            tree.tag_configure('%s'%(index), foreground="red")
        currentContent['MoDiFiEd'] = True

def nextSelection():
    global currentContent
    current_selection = tree.focus()

    # default next selection is the beginning
    next_selection = 0

    # make sure at least one item is selected
    if current_selection != '':
        last_selection = int(current_selection)

        # clear current selections
        #lbox.selection_clear(selection_indices)
    
        # Make sure we're not at the last item
        if last_selection < len(tree.get_children()) - 1:
            next_selection = last_selection + 1
            #annotatemsg.set('')
    
    if not tree.get_children(): return
    tree.selection_set('"%s"' % (next_selection))
    tree.focus("%s" % (next_selection))
    instance = "\t".join(tree.item('%s'%next_selection)['values'])
    #lbox.activate(next_selection)
    #lbox.selection_set(next_selection)
    nextItem = currentContent[instance]
    statusmsg.set("Similarity for '%s' is: '%s' and system confidence is: '%s'" % 
                            (instance.upper(), nextItem[1], nextItem[2]))
    system_outputs.set(nextItem[-1].split("#")[0])

def askPopup():
    #showinfo("Window", "Hello World!")
    what = askokcancel(message="You have not saved the changes to the current file. Opening a new file will \
                                cancel those changes.", title="Save the changes?")
    return what

def Quit():
    if currentContent.get("MoDiFiEd", False):
        what = askokcancel(message="You have not saved the changes to the current file. Closing the window will \
                                    cancel those changes.", title="Save the changes?")
        if what:
            pass
        else:
            root.quit()
    else:
        root.quit()

if __name__ == "__main__":
    try:
        assert len(sys.argv) == 3
        TASKDIR = sys.argv[1]
        SINKDIR = sys.argv[2]
    except AssertionError:
        sys.stdout.write("python3 bootstrapOntologies.py <annotation directory> <annotated directory>\n")
        sys.exit()
    
    if not os.path.exists(SINKDIR):
        os.makedirs(SINKDIR)
   
    with open("/home/riyaz/Dropbox/ClearEarth/gui/users.txt") as ufp:
        users = set([u.strip().split("\t")[-1] for u in ufp])
   
    USER = os.environ.get("USER")
    if USER not in users:
        sys.stderr.write("%s is not a listed user. Drop a mail to Riyaz asking for permission." % (USER))
        sys.exit()

    currentFile = str()
    currentContent = dict()
    
    outputs = {
                'positive':'Hyponym-hypernym', 
                'reverse':'Hypernym-hyponym', 
                'related':'Related Concepts', 
                'negative': 'None of the Above'
              }
    
    root = Tk()
    style = ttk.Style()
    root.wm_title("Clearonto Bootstrap")
    root.wm_iconbitmap('@/home/riyaz/Dropbox/ClearEarth/gui/favicon.xbm')
    #root.config(background="Gray")
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
    
    tree = ttk.Treeview(frame, columns=['Hyponym', 'Hypernym'], show="headings", selectmode="extended")
    vsb = ttk.Scrollbar(orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(orient="horizontal", command=tree.xview)
    vsb.grid(column=2, row=0, rowspan=6, sticky=(N,S,E,W), in_=frame)
    hsb.grid(column=0, row=6, sticky=(N,S,E,W), in_=frame)
    tree.config(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    tree.grid(column=0, row=0, rowspan=6,sticky=(N,S,E,W))
    
    for col in ['Hyponym', 'Hypernym']:
        tree.heading(col, text=col)
        tree.column(col,
            width=font.Font().measure(col))
    
    style.configure("Treeview.Heading", background="snow4", foreground='black', font=("Times", 12), weight=font.BOLD)
    #font.nametofont('TkHeadingFont').configure(size=12, weight=font.BOLD)
    
    #lbox = Listbox(frame, height=10, width=40, font='Courier')
    label = ttk.Label(frame, text="Modify System output:")
    
    pos = ttk.Radiobutton(frame, text=outputs['positive'], variable=system_outputs, value='positive')
    rev = ttk.Radiobutton(frame, text=outputs['reverse'], variable=system_outputs, value='reverse')
    rel = ttk.Radiobutton(frame, text=outputs['related'], variable=system_outputs, value='related')
    neg = ttk.Radiobutton(frame, text=outputs['negative'], variable=system_outputs, value='negative')
    modify = ttk.Button(frame, text='Modify', command=modifyAnnotation, default='active')
    move = ttk.Button(frame, text='Next', command=nextSelection, default='active')
    annotate = ttk.Label(frame, textvariable=annotatemsg)
    status = ttk.Label(frame, textvariable=statusmsg, anchor=W, 
                foreground='DarkOrchid4', font=("Roman", 11, "bold"), justify="center")
    
    menubar=Menu(root)
    
    menu=Menu(menubar,tearoff=0, postcommand=refresh)
    
    updateFileMenu(TASKDIR, SINKDIR, menu)
    
    menu.add_separator()
    menu.add_command(label="Save", command=saveAnnotations)
    menu.add_command(label="Exit", command=Quit)
    menubar.add_cascade(label="File", menu=menu)
    
    helpmenu=Menu(menubar,tearoff=0)
    helpmenu.add_command(label="Help",command=printHelp)
    menubar.add_cascade(label="Help",menu=helpmenu)
    root.config(menu=menubar)
    
    # Grid all the widgets
    #lbox.grid(column=0, row=0, rowspan=6, sticky=(N,S,E,W))
    label.grid(column=3, row=0, padx=10, pady=5)
    pos.grid(column=3, row=1, sticky=W, padx=20)
    rev.grid(column=3, row=2, sticky=W, padx=20)
    rel.grid(column=3, row=3, sticky=W, padx=20)
    neg.grid(column=3, row=4, sticky=W, padx=20)
    modify.grid(column=3, row=5, sticky=(N,W), padx=20, pady=20)
    move.grid(column=3, row=5, sticky=(N,E), padx=120, pady=20)
    #annotate.grid(column=1, row=6, columnspan=2, sticky=(N,W))
    status.grid(column=0, row=7, columnspan=2, pady=5, padx=(50,50))
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_rowconfigure(5, weight=1)
    
    # Set event bindings for when the selection in the listbox changes,
    #tree.bind('<<ListboxSelect>>', showConfidence)
    tree.bind('<<TreeviewSelect>>', showInfo)
    
    #lbox.selection_set(0)
    showInfo()
    ttk.Sizegrip(root).grid(column=999, row=999, sticky=(S,E))
    root.protocol("WM_DELETE_WINDOW", Quit)
    root.mainloop()
