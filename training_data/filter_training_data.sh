#!/bin/bash

mkdir matching_train_images
mkdir matching_train_labels
touch train.nyu
for f in ./train_images/*; do
    if [ -f "./train_labels/$(basename $f)" ]
    then 
        echo "match found: $f"
        cp "$f" ./matching_train_images
        cp "./train_labels/$(basename $f)" ./matching_train_labels
        echo "matching_train_images/$(basename $f) matching_train_labels/$(basename $f)" >> train.nyu
    fi
done

