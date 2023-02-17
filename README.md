# Revisiting-Pre-training-in-Audio-Visual-Learning
Here is the official repo for "Revisiting Pre-training in Audio-Visual Learning", which brings some interesting findings of pre-training in audio-visual learning.

**Paper Title: "Revisiting Pre-training in Audio-Visual Learning"**

**Authors: Ruoxuan Feng, Wenke Xia and [Di Hu](https://dtaoo.github.io/index.html)**

**[[arXiv](https://arxiv.org/abs/2302.03533)]**

## Is the effectiveness of pre-trained models always held?

We focus on a pair of typical heterogeneous modalities, audio and visual modality, where the heterogeneous data format is considered to bring more chances in exploring the effectiveness of pre-trained models.  Concretely, we concentrate on two typical cases of audio-visual learning: **cross-modal initialization** and **multi-modal joint learning**.

We find that **the underutilization of not only model capacity, but also model knowledge limits the potential of the pre-trained model** in the multi-modal scenario, as shown in the following figure.

![](https://raw.githubusercontent.com/GeWu-Lab/Revisiting-Pre-training-in-Audio-Visual-Learning/main/demo/findings.png)

## Why are the pre-trained models underutilized?

### In cross-modal initialization 

We discover that the absolute value of some parameter pairs ($\lvert\gamma_k\rvert$, $\lvert\beta_k\rvert$) is **significantly smaller** than other parameters, after checking all the Batchnorm layers in VGGSound and ImageNet pre-trained ResNet-18 models. The blue bars in the following figure indicate that these abnormal Batchnorm parameters could be found in every Batchnorm layer inside VGGSound pre-trained model. 

In experiments we find that the channels with abnormal Batchnorm parameters are more likely to produce “**dead features** ” after ReLU. This phenomenon does not merely exist for particular samples or classes but for most samples, indicating the channels are hard to be activated.  We name these channels as “**dead channels**”. The abnormal $\gamma_k$ also slows down the back-propagation of gradients. Thus, the corresponding filters in Conv-BN-ReLU structure are **difficult to be updated**.

![](https://raw.githubusercontent.com/GeWu-Lab/Revisiting-Pre-training-in-Audio-Visual-Learning/main/demo/bn_problem.png)

### In multi-modal joint learning

Although introducing a stronger pre-trained encoder for one modality in the multi-modal model is very likely to improve the performance, we find that this could **damage the representation ability of the other**, as shown in the first figure. Both of the pre-trained encoders could have **not yet adapted to the target task and thoroughly exploited the knowledge** from the pre-trained model. 

Recent work preliminarily pointed out that high-quality predictions of one modality could reduce the gradient back-propagated to another encoder. When directly initializing encoder with pre-trained model, predictions with high quality could be produced on the samples of one modality at the beginning, which are considered to be easy-to-learn. This could **exacerbate the insufficient optimization** of the encoders.

## Our solution

### For cross-modal initialization

We propose **Adaptive Batchnorm Re-initialization (ABRi)** to minimize the negative impact of abnormal parameters while ensuring coordination. An additional initialized Batchnorm layer is adaptively combined with each original Batchnorm layer.

![](https://raw.githubusercontent.com/GeWu-Lab/Revisiting-Pre-training-in-Audio-Visual-Learning/main/demo/abri.png)

### For multi-modal joint learning

We propose a **Two-stage Fusion Tuning (FusT)** strategy:

1.  Fine-tune the uni-modal encoders on their own uni-modal dataset detached from the multi-modal dataset. (Stage 1)
2.  Fine-tune the complete multi-modal model while randomly masking some parts of the easy-to-learn samples. (Stage 2)

The difficulty of the samples reflected by the sample-wise mean confidence in Stage 1.

![](https://raw.githubusercontent.com/GeWu-Lab/Revisiting-Pre-training-in-Audio-Visual-Learning/main/demo/fust.png)

Our proposed methods are simple and intuitive attempts. We hope these could inspire future works.

## Citation

If you find this work useful, please consider citing it.

```
@article{feng2023revisiting,
  title={Revisiting Pre-training in Audio-Visual Learning},
  author={Feng, Ruoxuan and Xia, Wenke and Hu, Di},
  journal={arXiv preprint arXiv:2302.03533},
  year={2023}
}
```

