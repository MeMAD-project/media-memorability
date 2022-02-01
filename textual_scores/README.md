# MediaEval Memorability Challenge
## mediaeval_memorability_2021.py 
The script for the experiments we run for the MeMAD Memorability Challenge 2020 to evaluate the different models for the text modality.
Usage is similar to the 2020 edition (below). 
The final text embeddings used for the final submission, and for the different datasets are saved under [`embeddings_output_2021`](./embeddings_output_2021)

## mediaeval_memorability_2020.py
The script for the experiments we run for the MeMAD Memorability Challenge 2020 to evaluate the different models for the text modality.
To actually compute memorability scores, check `mediaeval_compute_scores.py`.

```
usage: textual_scores/mediaeval_memorability_2020.py [-h] [-d VIDEO_DESCRIPTIONS_PATH]
                                      [-c DEEP_CAPTION_PATH]
                                      [-s VIDEO_SCORES_PATH]
                                      [-t TEST_SET_PATH] [-r RESULTS_PATH]
                                      [-wv WORD_EMBEDDINGS_PATH]
                                      [--save_model]

Computing text scores for MediaEval 2020

optional arguments:
  -h, --help            show this help message and exit
  -d VIDEO_DESCRIPTIONS_PATH, --video_descriptions_path VIDEO_DESCRIPTIONS_PATH
                        Path to the CSV file containing video IDs and
                        corresponding description(s)
  -c DEEP_CAPTION_PATH, --deep_caption_path DEEP_CAPTION_PATH
                        Path to the file containing deep captions (in the same
                        order as the training data)
  -s VIDEO_SCORES_PATH, --video_scores_path VIDEO_SCORES_PATH
                        Path to the CSV file containing ground-truth scores.
  -t TEST_SET_PATH, --test_set_path TEST_SET_PATH
                        Path to the CSV file containing video descriptions of
                        the testset.
  -r RESULTS_PATH, --results_path RESULTS_PATH
                        Path to where to save the results for short and long
                        term predictions.
  -wv WORD_EMBEDDINGS_PATH, --word_embeddings_path WORD_EMBEDDINGS_PATH
                        Path to word embeddings (e.g. GloVe)
```


## mediaeval_memorability_2019.py

The script for the experiments we run for the MeMAD Memorability Challenge 2019 to evaluate the different models for the text modality.

```
usage: textual_scores/mediaeval_memorability_2019.py [-h] [-d VIDEO_DESCRIPTIONS_PATH]
                                      [-wv WORD_EMBEDDINGS_PATH]
                                      [--save_model]

Computing text scores for MediaEval 2019

optional arguments:
  -h, --help            show this help message and exit
  -d VIDEO_DESCRIPTIONS_PATH, --video_descriptions_path VIDEO_DESCRIPTIONS_PATH
                        Path to the CSV file containing video IDs and
                        corresponding description(s)
  -wv WORD_EMBEDDINGS_PATH, --word_embeddings_path WORD_EMBEDDINGS_PATH
                        Path to word embeddings (e.g. GloVe)

```



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
