#!/usr/bin/python

import sys
import codecs
import commands

from keyPhraseExtraction import extractKeyphrases


def generatePairs(phrases):
    #for kp_i in phrases:
    while phrases:
        kp_i = phrases.pop(0)
        for kp_j in phrases:
            if kp_i == kp_j: continue
            if kp_i in kp_j: continue
            input_pair = "%s\t%s\n" % (kp_i, kp_j)
            yield input_pair

    text = inputItem.file.read().decode("utf-8", "ignore")
    glossary = [ep for ep in extractKeyphrases(text) if len(ep) > 2]

    pairs = generatePairs(glossary)
    outputTerms = set(ontoExtract(pairs))
    outputTermsSorted = sorted(outputTerms, key=lambda o: (o[3]*.4)+(o[4]*.6), reverse=True)
    
newOntoterms = []
settled = set()
for id1, out1 in enumerate(outputTermsSorted):
    count = 1
    if out1[0] in settled:continue
    settled.add(out1[0])
    newOntoterms.append(out1)
    for id2, out2 in enumerate(outputTermsSorted):
        if id1 == id2: continue
        if count <= 4:
            if out1[0] == out2[0]:
                count += 1
                newOntoterms.append(out2)
        else: break

#<input type="button" name="Yes" value="Yes">
#<input type="button" name="No" value="No"></td></p>
kcount = 0
#for output_pair in sorted(outputTerms, key=lambda o: o[4], reverse=True):
for output_pair in newOntoterms:
    subtype, supertype, relation, probability, distance = output_pair
    if subtype in supertype.split()[:-1]:continue
    if supertype in subtype.split()[:-1]:continue
    if subtype.split()[-1] == supertype: relation = "Hypernym"
    #if (relation == "Hypernym") #and (probability >= 0.6) and distance > 0.6:
