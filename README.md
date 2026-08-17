# (KDD 2026) SIMS: Scale-Invariant Merit-Function-Based Scalarization for Multi-Task Learning

Official implementation of **SIMS**, a SAM 2 based framework for multi-task dense prediction.

## Abstract

Multi-task learning (MTL) requires navigating unavoidable trade-offs among competing objectives. This paradigm is frequently formulated as multi-objective optimization (MOO), where the scalarization is favored to reduce an MOO problem to a single objective. We empirically find that existing merit-function-based scalarization approaches are sensitive to the relative scales of different objectives in practical MTL, where task losses commonly differ by orders of magnitude. The optimization process often favors objectives with larger scales even though the underlying Pareto optimal solutions remains invariant to rescaling (i.e., multiplying an objective by a positive constant). To address this issue, we propose **S**cale-**I**nvariant **M**erit-function-based **S**calarization (**SIMS**) for MTL. Specifically, SIMS adopts a transformation-induced merit function to convert the MOO problem of MTL to a single objective that renders optimization invariant to the magnitudes of losses. Theoretically, we prove that the requirement for scale invariance uniquely determines this transformation to be logarithmic. We further show that this general transformation-induced merit function preserves weak Pareto optimality and admits a smooth surrogate with controllable approximation error. Extensive experiments on representative multi-task benchmarks demonstrate that SIMS consistently outperforms existing scalarization methods and achieves state-of-the-art performance.

## Pre-trained Model

This code uses the official **SAM 2.1 Hiera-Large** checkpoint:

- Official SAM 2 repository: https://github.com/facebookresearch/sam2
- Hugging Face model card: https://huggingface.co/facebook/sam2.1-hiera-large
- Direct checkpoint URL: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt



## Installation

The local `sam2` environment used for this release is Python 3.11.11 with PyTorch 2.5.1 and TorchVision 0.20.1.

```bash
conda create -n sims python=3.11 -y
conda activate sims
pip install -r requirements.txt
```




## How to Run

PASCAL-Context:

```bash
bash run_pascal.sh
```

Cityscapes:

```bash
bash run_city.sh
```

NYUv2:

```bash
bash run_nyu.sh
```

 Edit `CUDA_VISIBLE_DEVICES`, `nproc_per_node`, `sam_checkpoint`, and `data_root` in those scripts for your machine.

## Citation

If this repository is useful for your research, please cite our paper:

```bibtex
@inproceedings{chen2026sims,
  title={SIMS: Scale-Invariant Merit-Function-Based Scalarization for Multi-Task Learning},
  author={Chen, Zebin and Xing, Fei and Chen, Yang and Liu, Hua and Chow, Andy HF and Qian, Yuhua and Zhang, Yu},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={520--531},
  year={2026}
}
```

