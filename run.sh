#!/bin/sh -e

python train_MLM_with_validation.py -e 10 -g cam -n all -m noun_verb  -r 1 -f training_data/0.6_True_verb_noun_concat.tsv -pre ./model/cam/noun_verb_all_False_8epochs_1e-05