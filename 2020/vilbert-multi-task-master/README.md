

# 12-in-1: Multi-Task Vision and Language Representation Learning

Please cite the following if you use this code. Code and pre-trained models for [12-in-1: Multi-Task Vision and Language Representation Learning](http://openaccess.thecvf.com/content_CVPR_2020/html/Lu_12-in-1_Multi-Task_Vision_and_Language_Representation_Learning_CVPR_2020_paper.html):

```
@InProceedings{Lu_2020_CVPR,
author = {Lu, Jiasen and Goswami, Vedanuj and Rohrbach, Marcus and Parikh, Devi and Lee, Stefan},
title = {12-in-1: Multi-Task Vision and Language Representation Learning},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

and [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265):

```
@inproceedings{lu2019vilbert,
  title={Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks},
  author={Lu, Jiasen and Batra, Dhruv and Parikh, Devi and Lee, Stefan},
  booktitle={Advances in Neural Information Processing Systems},
  pages={13--23},
  year={2019}
}
```

## Repository Setup

1. Create a fresh conda environment, and install all dependencies.

```text
conda create -n vilbert-mt python=3.6
conda activate vilbert-mt
git clone --recursive https://github.com/facebookresearch/vilbert-multi-task.git
cd vilbert-multi-task
pip install -r requirements.txt
```

2. Install pytorch
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch #Install cudatoolkit that fits the computer version , same as nvcc --version
```

3. Install apex, follows https://github.com/NVIDIA/apex

4. Install this codebase as a package in this environment.
```text
python setup.py develop
```
5. Install gitmodules with 
```text
 git submodule init
 git submodule update
cd vilbert-multi-task/tools/refer
python setup.py install
make
#Then replace refer.py byt https://gist.github.com/vedanuj/9d3497d107cfca0b6f3dfdc28d5cb226 to update from Python2 version to Python3
```


## Data Setup

Check `README.md` under `data` for more details.  

 
## Vilbert features extraction

In this part, the fine-tuned (VQA or NLVR2) model weights are being frozen. 

### Prepare Captions
Preparing captions consists of loading video IDs and captions from the `.txt` or `.csv` file, , tokenize, tensorize and save the cache file. An example of using this script
```
python script/ME/captions_preparation.py --captions_path /MediaEval/alto_titles_danny.csv --gt_path /MediaEval/dev-set/ground-truth/ground-truth_dev-set.csv --split trainval --dc
```

### Extract Frames from Video

Use the `frames` parameter for the number of frames to be extracted (default is 1 i.e., the middle frame of the video). The extracted frames are saved as `<output_folder>/<video-id>_<frame_count>.jpg` where `<frame_count>` in `[0..<frames>-1]` (and `<output_folder>/<video-id>.jpg` when extracting only one frame). Otherwise pass a list of frames you want to be extracted with `frame_list` parameter, which is a link to a csv file with columns=video_name, frame. Keep this structure since it is used by the `script/ME/average_features.py` or `script/extract_features.py` scripts.
Make sure to have writing permission for the `output_folder`. Otherwise, here is an example to use
```
python script/ME2020/extract_frames.py --output_folder <output_folder> --video_dir <video_dir> --frames <frames>
```

### Extract Features for Multiple Frames
Use `script/extract_features.py` and add `samples` parameter for the number of frames to use.
```
python script/extract_features.py --model_file data/detectron_model.pth --config_file data/detectron_config.yaml --image_dir datasets/ME/images/train --output_folder datasets/ME/features_100/ME_trainval_resnext152_faster_rcnn_genome.lmdb/ --samples 5
```

### Average Visual Feature Vectors
If using multiple extracted frames from each video, this script is used to average already extracted features. Features files should be named `<video-id>_<feature_count>.npy` where `<feature_count>` in `[0..<feature_number>]`.
```
python script/ME/average_features.py --features_dir <path_to_directory_with_features> --output_folder <path_to_output_averaged_features>
```
### Convert Visual Feature Vectors to lmdb
```
python script/convert_to_lmdb.py  ----features_dir <path_to_directory_with_features> --lmdb_file  <path_to_output_lmdb_file> 
```

### Get Vilbert representation

--tasks 20 allows you to extract vilbert features.
Save the visual and textual representations to `--rep_save_path` so they can be used later to train a regressor. 
Path to prepared captions and visual features must be specified in vilbert_tasks.yml (TASK20)

```
python script/ME/vilbert_representations.py --bert_model bert-base-uncased --from_pretrained save/VQA_bert_base_6layer_6conect-finetune_from_multi_task_model-task_1/pytorch_model_19.bin --config_file config/bert_base_6layer_6conect.json --tasks 20 --batch_size 128 --task_specific_tokens --rep_save_path datasets/ME/out_features/train_features.pkl
```


## License

vilbert-multi-task is licensed under MIT license available in [LICENSE](LICENSE) file.
