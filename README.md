# 3DMolFormer
This is the code repository for our paper published in ICLR 2025: [3DMolFormer: A Dual-channel Framework for Structure-based Drug Discovery](https://arxiv.org/abs/2502.05107).

## Dependencies

```bash
pytorch
numpy
scipy
tqdm
tensorboard
multiprocessing
rdkit
openbabel
```

## Datasets

For pre-training, we use:

- Molecular pretraining and pocket pretraining datasets by Uni-Mol: https://github.com/deepmodeling/Uni-Mol/tree/main/unimol
- Pocket-ligand pair data by CrossDocked2020: https://github.com/gnina/models/tree/master/data/CrossDocked2020

For the docking task, we use:

- The PDBbind v2020 dataset: https://www.pdbbind-plus.org.cn/download

For the pocket-aware drug design task, we use 

- The training / test split by TargetDiff: https://github.com/guanjq/targetdiff

## Pre-training

```bash
python pretraining.py
```

## Supervised Fine-tuning for Docking

```bash
python docking_finetuning.py
```

## RL Fine-tuning for Molecule Generation

```bash
python generation_finetuning.py
```

## Citation

```bash
@inproceedings{3DMolFormer,
  title={3DMolFormer: A Dual-channel Framework for Structure-based Drug Discovery},
  author={Hu, Xiuyuan and Liu, Guoqing and Chen, Can and Zhao, Yang and Zhang, Hao and Liu, Xue},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

