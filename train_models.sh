#!/usr/bin/env bash
set -e

for mode in up_low most_low;
do
    for vocab in 25 50 100;
    do
        for nl in {3,4,5};
        do
        if [ $nl -eq 5 -a $vocab -eq 100 ]; then
            BS=128
        else
            BS=192
        fi
        dir="work/${mode}${vocab}k"
        python fastai_scripts/pretrain_lm.py --dir-path "${dir}" --cuda-id 0 --cl 12 --bs "${BS}" --lr 0.01 --pretrain-id "nl-${nl}-small-minilr" --sentence-piece-model sp.model --nl "${nl}"
        done
    done
done
