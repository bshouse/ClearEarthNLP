# ClearEarthNLP

ClearEarthNLP is an easy to use toolkit for text processing in Earth Science domains such as Cryosphere, Ecology and Earthquake. It contains implementations for part-of-speech tagging, named entity recognition, dependency parsing and subsumption learning. All the models are trained using deep neural networks and give state-of-the-art performance.

# Table of Contents

  * [ClearEarthNLP](#clearearthnlp)
  * [Table of Contents](#table-of-contents)
  * [Installation](#installation)
    * [Linux](#linux)
    * [Mac](#mac)
  * [Run](#run)
  * [NLP Terminology](#nlp-terminology)
    * [POS](#pos)
    * [NER](#ner)
    * [Dependency Parsing](#dependency-parsing)

# Installation

## Linux

Install Python 3, pip, and tkinter:

    sudo apt-get install python3 python3-pip python3-tk

Install Python packages:

    pip3 install git+git://github.com/irshadbhat/indic-tokenizer.git nltk gensim==0.13.1 graphviz networkx pydot3 pillow dynet==2.0

Download NLTK corpora:

    python3 -m nltk.downloader stopwords wordnet
    
Download and unzip ClearEarthNLP.

## Mac

Install [Python 3](https://www.python.org/downloads/mac-osx/).

Install pip:

    curl https://bootstrap.pypa.io/get-pip.py | python3
   
Install Python packages:

    pip3 install gensim git+git://github.com/irshadbhat/indic-tokenizer.git nltk graphviz pydot pillow dynet
    
Download NLTK corpora:

    python3 -m nltk.downloader stopwords wordnet

Download and unzip ClearEarthNLP.

# Run

From inside the ClearEarthNLP directory, run:

    ./clearEarthNLP.sh

The model files will automatically download and unzip into the "models/" directory the first time ClearEarthNLP is run. Expect this to take several minutes.

Under "File" in the menu, you can select the text file to load.

Select the NLP Tool you wish to run on the file and click "Run". This will load the model, which will likely take several minutes. The tool will not indicate progress, other than to be non-responsive until it's done. After it is loaded, right-click on a sentence and select the relevant menu option to view the results.

# NLP Terminology

## POS

ClearEarthNLP uses the Penn Treebank part of speech tags.

| Tag  | Description | 
| ------------- | ------------- |
| CC | Coordinating conjunction |
| CD | Cardinal number |
| DT | Determiner |
| EX | Existential there |
| FW | Foreign word |
| IN | Preposition or subordinating conjunction |
| JJ | Adjective |
| JJR | Adjective, comparative |
| JJS | Adjective, superlative |
| LS | List item marker |
| MD | Modal |
| NN | Noun, singular or mass |
| NNS | Noun, plural |
| NNP | Proper noun, singular |
| NNPS | Proper noun, plural |
| PDT | Predeterminer |
| POS | Possessive ending |
| PRP | Personal pronoun |
| PRP$ | Possessive pronoun |
| RB | Adverb |
| RBR | Adverb, comparative |
| RBS | Adverb, superlative |
| RP | Particle |
| SYM | Symbol |
| TO | to |
| UH | Interjection |
| VB | Verb, base form |
| VBD | Verb, past tense |
| VBG | Verb, gerund or present participle |
| VBN | Verb, past participle |
| VBP | Verb, non-3rd person singular present |
| VBZ | Verb, 3rd person singular present |
| WDT | Wh-determiner |
| WP | Wh-pronoun |
| WP$ | Possessive wh-pronoun |
| WRB | Wh-adverb |

## NER

ClearEarthNLP uses standard "BIO" NER tagging. Each named entity (such as "Biotic_Entity") is appended with "B" (**B**eginning of named entity) or "I" (**I**nside the entity). Tokens that are not an entity are **O**utside.

## Dependency Parsing

Dependency parse edges are labeled using [Stanford Dependencies tags](https://nlp.stanford.edu/software/stanford-dependencies.shtml#English).

| Label | Description |
| ------------- | ------------- |
| root | root |
| dep | dependent |
| aux | auxiliary |
| auxpass | passive auxiliary |
| cop | copula |
| arg | argument |
| agent | agent |
| comp | complement |
| acomp | adjectival complement |
| ccomp | clausal complement with internal subject |
| xcomp | clausal complement with external subject |
| obj | object |
| dobj | direct object |
| iobj | indirect object |
| pobj | object of preposition |
| subj | subject |
| nsubj | nominal subject |
| nsubjpass | passive nominal subject |
| csubj | clausal subject |
| csubjpass | passive clausal subject |
| cc | coordination |
| conj | conjunct |
| expl | expletive (expletive “there”) |
| mod | modifier |
| amod | adjectival modifier |
| appos | appositional modifier |
| advcl | adverbial clause modifier |
| det | determiner |
| predet | predeterminer |
| preconj | preconjunct |
| vmod | reduced, non|finite verbal modifier |
| mwe | multi|word expression modifier |
| mark | marker (word introducing an advcl or ccomp |
| advmod | adverbial modifier |
| neg | negation modifier |
| rcmod | relative clause modifier |
| quantmod | quantifier modifier |
| nn | noun compound modifier |
| npadvmod | noun phrase adverbial modifier |
| tmod | temporal modifier |
| num | numeric modifier |
| number | element of compound number |
| prep | prepositional modifier |
| poss | possession modifier |
| possessive | possessive modifier (’s) |
| prt | phrasal verb particle |
| parataxis | parataxis |
| goeswith | goes with |
| punct | punctuation |
| ref | referent |
| sdep | semantic dependent |
| xsubj | controlling subject |
