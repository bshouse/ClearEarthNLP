#!/bin/bash

for model in tagger parser ner onto
do
    if [[ ! -f "models/$model.zip" ]] && [[ ! -d "models/$model" ]];
    then
        wget -O "models/$model.zip" "http://verbs.colorado.edu/~ribh9977/models/$model.zip"
        unzip "models/$model.zip" -d models
    fi
done

python3 clearnlp.py
