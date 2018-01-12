ClearNLP is an easy to use toolkit for text processing in Earth Science domains such as Cryosphere, Ecology and Earthquake. It contains implementations for part-of-speech tagging, named entity recognition, dependency parsing and subsumption learning. All the models are trained using deep neural networks and give state-of-the-art performance.


__Manual Installation__

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


__Run__
```
bash clearnlp.sh
```

