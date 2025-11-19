# The Transformer architecture

> **Disclaimer**
>
> This Transformer implementation misses a lot of architectural improvements that have been developed since the first paper release in 2017. It's main purpose is to show off the architecture design and training methods as a learning project.


## Introduction

The Transformer architecture was first introduced in the paper "Attention Is All You Need" by Google[@vaswani2023attentionneed] as an alternative to recurrent or convolution-based networks for sequence processing. Since then it has been a disruptive architecture achieving important results in Natural Language Processing and in many other field, such as Computer Vision, Audio processing, Generative AI and Multimodal learning.



![Inference schema](architecture_explain/img/inference_schema.png){ loading=lazy }
/// caption
Caption. [@build-llms-from-scratch-book]
///
