# MediaEval Memorability Challenge

## mediaeval_compute_scores.py
Used to compute memorability scores from deep captions using *pretrained* models on MediaEval 2020 dataset (`me20_svr_w2v_st_model.pickle` and `me20_svr_w2v_lt_model.pickle`).
The input CSV file (deep_caption_path) should contain the field "caption"
The output file (save_path) adds two columns to the input CSV: "results_st" and "results_lt" for short and long term predictions, respectively.

```
usage: mediaeval_compute_scores.py [-h] [-d DEEP_CAPTION_PATH]
                                   [-wv WORD_EMBEDDINGS_PATH] [-s SAVE_PATH]

Computing text scores for MediaEval 2020

optional arguments:
  -h, --help            show this help message and exit
  -d DEEP_CAPTION_PATH, --deep_caption_path DEEP_CAPTION_PATH
                        Path to the file containing deep captions
  -wv WORD_EMBEDDINGS_PATH, --word_embeddings_path WORD_EMBEDDINGS_PATH
                        Path to word embeddings (e.g. GloVe)
  -s SAVE_PATH, --save_path SAVE_PATH
                        Path to save the Short and Long predictions
```
