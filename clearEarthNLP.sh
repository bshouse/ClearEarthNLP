#!/bin/bash

platform=`uname`

if [[ ! -d models ]];
then
    mkdir models
fi

for model in tagger parser ner onto
do
    if [[ ! -d "models/$model" ]] && [[ ! -f "models/$model.zip" ]];
    then
        if [[ $platform == "Linux" ]];
        then
            wget -O "models/$model.zip" "http://verbs.colorado.edu/~ribh9977/models/$model.zip"
        else
            curl -o "models/$model.zip" "http://verbs.colorado.edu/~ribh9977/models/$model.zip"
        fi
        unzip "models/$model.zip" -d models
    fi
done

python3 clearEarthNLP.py
