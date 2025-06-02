<h1 align="center">
Autoregression-free video prediction using <br/> diffusion model for mitigating error propagation <br/> (ARFree)
</h1>
<h4 align="center">

Woonho Ko, Jin Bok Park, Il Yong Chun

<h4 align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2409.08026-b31b1b.svg)](https://arxiv.org/abs/2505.22111)

This is the official implementation of <span style="color:blue;"><strong>ARFree</strong></span>.


</h4>


<br/>

## Abstract

Existing long-term video prediction methods often rely on an autoregressive video prediction mechanism. However, this approach suffers from error propagation, particularly in distant future frames. To address this limitation, this paper proposes the first  <b>AutoRegression-Free <i>(ARFree)</i></b> video prediction framework using diffusion models. Different from an autoregressive video prediction mechanism, ARFree directly predicts any future frame tuples from the context frame tuple. The proposed ARFree consists of two key components: 1) a motion prediction module that predicts a future motion using motion feature extracted from the context frame tuple; 2) a training method that improves motion continuity and contextual consistency between adjacent future frame tuples. Our experiments with two benchmark datasets show that the proposed ARFree video prediction framework outperforms several state-of-the-art video prediction methods. Please check the paper here: [ARFree](https://arxiv.org/abs/2505.22111)

<br/>

## Setup

```bash
conda create --name ARFree python==3.8.0
conda activate ARFree
pip install -r requirements.txt
pip install natten==0.17.3+torch240cu118 -f https://shi-labs.com/natten/wheels/
```


## Training

```bash
sh scripts/train.sh
```

## Inference 

```bash
sh scripts/valid.sh
```



## Acknowledgments

This project is built on the following resources:

- [**HDiT**](https://github.com/crowsonkb/k-diffusion): Our code is built upon the foundational work provided by HDiT, which is a high-resolution diffusion model for image generation.


<br/>

