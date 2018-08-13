# Submission to PolEval - task3 language model

Our solution is an extension of the work done by FastAI team to train language models for English.
We extended it with google sentence piece to tokenize polish words. 


## Installation
The source code needs cleaning up to minimise the amount of work needed to run it.

But for now here are rough manual steps:

- Install fastai from [our fork](https://github.com/n-waves/fastai/releases/tag/poleval2018) (python PATH) 
- Install sentencepiece from [source code](https://github.com/google/sentencepiece/commit/510ba80638268104811f89f6a8f702c4d6047a5f) (PATH and python PATH)

## Training
You should have the following structure:
```
.
├── data
│   └── task3 # here goes unzipped files
│       ├── test
│       └── train
├── fastai_scripts -> github.com/fastai/fastai/courses/dl2/imdb_scripts/
├── task3
└── work  # this will be created by scripts
    ├── up_low
    │   ├── models
    │   └── tmp
    └── up_low50k
        ├── models
        └── tmp 
```

Then to train a model run
```
cd task3/
./prepare-data3-up_low.sh
./prepare-data3-up_low50k.sh
cd ..

# to initially train the model
python fastai_scripts/pretrain_lm.py --dir-path ./work/up_low --cuda-id 0 --cl 10 --bs 192 --lr 0.01\
   --pretrain-id "v100k" --sentence-piece-model sp-100k.model --nl 4

# to check the perplexity
python fastai_scripts/infer.py --dir-path ./work/up_low --cuda-id 0 --bs 22 --pretrain-id "v100k"\
   --sentence-piece-model sp-100k.model --test_set tmp/val_ids.npy --correct_for_up=False --nl 4
   
# to fine tune with smaller dropout
python ./fastai_scripts/finetune_lm.py --dir-path work/up_low \
    --pretrain-path work/up_low --cuda-id 0 --cl 6 --pretrain-id "v100k" --lm-id "v100k_finetune"\
    --bs 192 --lr 0.001 --use_discriminative False --dropmult 0.5 --sentence-piece-model sp-100k.model --sampled True --nl 4 

# to check the perplexity on the valuation data set
python fastai_scripts/infer.py --dir-path ./work/up_low --cuda-id 0 --bs 22 --pretrain-id "v100k_finetune_lm"\
   --sentence-piece-model sp-100k.model --test_set tmp/val_ids.npy --correct_for_up=False --nl 4

```

To check the perplexity on test set run: 
```
# it expects your model in `work/up_low/models` and sentence piece model in `work/up_low/tmp` 
python fastai_scripts/infer.py --dir-path ./work/up_low --cuda-id 0 --bs 22 --pretrain-id "v100k_finetune_lm"\
   --sentence-piece-model sp-100k.model --test_set tmp/test_ids.npy --correct_for_up=False --nl 4      
```