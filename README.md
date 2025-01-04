# RMem: Restricted Memory Banks Improve Video Object Segmentation (for DAVIS and LVOS evaluation)

Junbao Zhou, [Ziqi Pang](https://ziqipang.github.io/), [Yu-Xiong Wang](https://yxw.web.illinois.edu/)

University of Illinois Urbana-Champaign

## Introduction

This is the repository of [RMem](https://github.com/Restricted-Memory/RMem), for evaluating on DAVIS and LVOS dataset. [RMem](https://github.com/Restricted-Memory/RMem) is based on AOT and DeAOT, but the evaluation code on VOST and DAVIS/LVOS is a bit different. Therefore we split it into 2 repo.


## Data preparation

Download the DAVIS dataset from [davis2017](https://davischallenge.org/davis2017/code.html) , and organize the directory structure as follows:

```bash
.
├── aot-benchmark
│   ├── configs
│   ├── dataloaders
│   └── datasets
│       └── DAVIS-2017
│           ├── Annotations
│           │   └── 480p
│           ├── ImageSets
│           │   ├── 2016
│           │   │   ├── train.txt
│           │   │   └── val.txt
│           │   └── 2017
│           │       ├── train.txt
│           │       └── val.txt
│           └── JPEGImages
│               └── 480p
├── davis2017-evaluation
└── README.md
```

> hint: you can achieve it by soft link

## Checkpoint

| Method            | $\mathcal{J} \& \mathcal{F}$ | $\mathcal{J}$ | $\mathcal{F}$ |                                             |
| ----------------- | ---------------------------- | ------------- | ------------- | ------------------------------------------- |
| R50 DeAOTL        | 85.2                         | 82.3          | 88.1          | [download link][deaot-vanilla-ckpt-link]    |
| R50 DeAOTL + RMem | 85.2                         | 82.3          | 88.2          | [download link][deaot-rmem-davis-ckpt-link] |

Download the checkpoint and put them in `./aot-benchmark/pretrain_models/`

[deaot-rmem-davis-ckpt-link]: https://drive.google.com/file/d/1eVpkueDix6xOoq0C_V_Ei9BK7OXQ4uX5/view?usp=sharing

[deaot-vanilla-ckpt-link]: https://drive.google.com/file/d/1edAk8O2PWRS4jpD3m8K_H1kmj89VF-sd/view?usp=sharing

## Evaluation

Firstly prepare the pytorch environment. Please follow the instructions on [pytorch.org](https://pytorch.org/) and choose the pytorch version that is most compatible with your machine.

Then
```bash
conda install numpy matplotlib scipy scikit-learn tqdm pyyaml pandas
pip install opencv-python
```

Now you can replicate the result of our checkpoint.
```bash
cd ./aot-benchmark
./train_eval.sh
```
