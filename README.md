ClearEarthNLP is an easy to use toolkit for text processing in Earth Science domains such as Cryosphere, Ecology and Earthquake. It contains implementations for part-of-speech tagging, named entity recognition, dependency parsing and subsumption learning. All the models are trained using deep neural networks and give state-of-the-art performance.


# Installation

## Linux
The following is a list of all the commands needed to perform a manual install:
```
# Installing Python3 ClearNLP:

# get ClearNLP from GitHub
https://github.com/ClearEarthProject/ClearNLP.git

# install gensim
pip3 install gensim==0.13.1

# install tokenizer
pip3 install git+git://github.com/irshadbhat/indic-tokenizer.git

# install nltk
pip3 install -U nltk

# install graphviz
pip3 install graphviz

# install networkx
pip3 install networkx

# install pydot
pip3 install pydot3

# install PIL
pip3 install pillow

# install tkinter
apt-get install python3-tk

# install dynet
pip3 install dynet==2.0
```

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

    ./clearnlp.sh

The model files will automatically download and unzip into the "models/" directory the first time ClearEarthNLP is run. Expect this to take several minutes.

Under "File" in the menu, you can select the text file to load.

Select the NLP Tool you wish to run on the file and click "Run". This will load the model, which may take several minutes. After it is loaded, right-click on a sentence and select the relevant menu option to view the results.

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

Dependency parse edges are labeled using [Universal Dependencies tags](http://universaldependencies.org/en/dep/).
