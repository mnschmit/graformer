# Graformer - Graph Transformer for Graph-to-Text Generation
This repository contains the code for the TextGraphs-15 paper "Modeling Graph Structure via Relative Position for Text Generation from Knowledge Graphs".

If you find it useful, please consider citing
```
@inproceedings{schmitt-etal-2021-modeling,
    title = "Modeling Graph Structure via Relative Position for Text Generation from Knowledge Graphs",
    author = {Schmitt, Martin  and
      Ribeiro, Leonardo F. R.  and
      Dufter, Philipp  and
      Gurevych, Iryna  and
      Sch{\"u}tze, Hinrich},
    booktitle = "Proceedings of the Fifteenth Workshop on Graph-Based Methods for Natural Language Processing (TextGraphs-15)",
    month = jun,
    year = "2021",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/11.textgraphs-1.2",
    pages = "10--21",
}
```

# Data
## AGENDA
1. Download the unprocessed json files for the AGENDA dataset from [here](https://github.com/rikdz/GraphWriter). You will need to unpack the archive into the `data/agenda` directory.
2. Apply the patch we provide for training instances with special characters by running the following command inside the `data/agenda` directory: `patch unprocessed.train.json agenda.patch`

## WebNLG
1. Download the XML files used in the original challenge in 2017 from [here](https://gitlab.com/shimorina/webnlg-dataset/-/tree/master/webnlg_challenge_2017). Put the three folders `train`, `dev`, and `test` into the `data/webnlg` directory.
2. Run the following commands to convert the XML files to JSON.
```
python3 src/preprocessing/format_webnlg_xml.py data/webnlg/train/*triples/*.xml data/webnlg/train.json
python3 src/preprocessing/format_webnlg_xml.py data/webnlg/dev/*triples/*.xml data/webnlg/dev.json
python3 src/preprocessing/format_webnlg_xml.py data/webnlg/test/testdata_with_lex.xml data/webnlg/test.json
```

## Creating reference files

### WebNLG
```
python3 src/preprocessing/create_ref_webnlg.py data/webnlg/dev.json data/webnlg/val-ref.txt
python3 src/preprocessing/create_ref_webnlg.py data/webnlg/test.json data/webnlg/test-ref.txt
```
### AGENDA
```
python3 src/preprocessing/create_ref_agenda.py data/agenda/unprocessed.val.json data/agenda/val-ref.txt
python3 src/preprocessing/create_ref_agenda.py data/agenda/unprocessed.test.json data/agenda/test-ref.txt
```


# Tokenization

## Learn a BPE model

1. Create corpus for vocabulary training
```
python3 -m src.preprocessing.create_raw_corpus_webnlg data/webnlg/train.json data/webnlg/preprocessed_spm_corpus.txt
```
2. Train the model
```
python3 src/preprocessing/train_spm_webnlg.py data/webnlg/preprocessed_spm_corpus.txt data/webnlg/bpe 2100
```

For AGENDA, remove the `_webnlg` part from the scripts' names and adjust the vocabulary size from 2100 to 24100.

## Use the BPE model

Adapt one of the following commands to preprocess the appropriate data split, e.g., WebNLG dev:
```
python3 -m src.preprocessing.prepare_webnlg data/webnlg/dev.json data/webnlg/val-graphs.json data/webnlg/val-texts.json data/webnlg/bpe.model
```

Another example, AGENDA train:
```
python3 -m src.preprocessing.prepare_agenda data/agenda/unprocessed.train.json data/agenda/train-graphs.json data/agenda/train-texts.json data/agenda/bpe.model
```

# Training

For training a model, adapt one of the following commands:

```
python3 -m src.train.lightning_tokennodes data/agenda/ --gpus 0 --no-copy --beam-size 1 --batch-size 8 --attention-dropout 0.1 --coverage-weight 0.0 --dim-feedforward 2000 --hidden-dim 400 --dropout 0.1 --gpos-regularizer 0.0 --input-dropout 0.1 --l2-regularizer 0.0003 --label-smoothing 0.3 --max-graph-range 6 --max-text-range 50 --num-decoder-layers 5 --num-encoder-layers 4 --num-heads 8 --same-text-range 10 --word-dropout 0.0 --experiment-name agenda_seed1 --optimizer-name adafactor --num-epochs 40 --accum-count 2 --seed 1 --prenorm --grad-clip 1.0
```

```
python3 -m src.train.lightning_tokennodes data/webnlg/ --gpus 0 --webNLG --no-copy --beam-size 1 --batch-size 4 --attention-dropout 0.3 --coverage-weight 0.0 --dim-feedforward 512 --hidden-dim 256 --dropout 0.1 --gpos-regularizer 0.0 --input-dropout 0.0 --l2-regularizer 0.003 --label-smoothing 0.25 --max-graph-range 4 --max-text-range 25 --num-decoder-layers 3 --num-encoder-layers 3 --num-heads 8 --same-text-range 10 --word-dropout 0.0 --experiment-name webnlg_seed1 --optimizer-name adafactor --num-epochs 200 --accum-count 3 --seed 1 --prenorm --grad-clip 1.0
```

# Evaluation

```
python3 -m src.train.predict --gpus 0 checkpoints/agenda_seed1/version_0/epoch=37-step=91997.ckpt data/agenda/test-graphs.json data/agenda/test-texts.json data/agenda/test-ref.txt --batch-size 64 --beam-size 2 --length-penalty 5.0
```

```
python3 -m src.train.predict --gpus 0 checkpoints/webnlg_seed1/epoch=82.ckpt data/webnlg/test-graphs.json data/webnlg/test-text.json data/webnlg/test-ref.txt --batch-size 32 --beam-size 2 --length-penalty 5.0
```
