# MeMAD's participation to the MediaEval Media Memorability 2019 and 2020 challenges

 - [MediaEval Media Memorability 2019 Task](http://www.multimediaeval.org/mediaeval2019/memorability/) | [github](https://github.com/multimediaeval/2019-Predicting-Media-Memorability-Task)
 - [MediaEval Media Memorability 2020 Task](https://multimediaeval.github.io/editions/2020/tasks/memorability/) | [github](https://github.com/multimediaeval/2020-Predicting-Media-Memorability-Task)

Please cite the following if you use this code.
```
@inproceedings{reboud2019combining,
  title={Combining Textual and Visual Modeling for Predicting Media Memorability},
  author={Reboud, Alison and Harrando, Ismail and Laaksonen, Jorma and Francis, Danny and Troncy, Rapha{\"e}l and Mantec{\'o}n, H{\'e}ctor Laria},
  booktitle = {MediaEval 2019: Multimedia Benchmark Workshop},
  year={2019},
  address = {Sophia Antipolis, France}
}


```


```
@inproceedings{reboud2020predicting,
  title={Predicting Media Memorability with Audio, Video, and Text representation},
  author={Reboud, Alison and Harrando, Ismail and Laaksonen, Jorma and Troncy, Rapha{\"e}l and others},
  booktitle={MediaEval 2020: Multimedia Benchmark Workshop},
  year={2020}
}
```


Our approach for the 2020 edition is a weighted average method combining predictions made separately from visual, audio, textual and visiolinguisticrepresentations of videos.

![Model architecture](./images/2020_architecture.png)




```
python combine_scores_2020.py
```

## Usage

The approach consists in computing three different scores independtly and later averaging them. 


#### Computing the text scores


#### Computing the memorability visiolinguistic scores (2020 edition only)

The first step consists in extracting Vilbert features from the frozen task-agnostic Vilbert model, following the instructions in the  `README.md`under vilbert/vilbert-multi-task. 

The second step consists in obtaining  and computing the the memorability scores using 
``` python mediaeval2020_pred.py ```


#### Computing the visual and audio-visual scores






Then obtain the final score by running, combine_scores_2020.py, a code snippet for evluating all linear combinbations of values to combine different modalities.
