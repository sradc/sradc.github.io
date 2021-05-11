---
layout: post
title:  "The Fourier transform is a neural network - follow-up"
date:   2021-05-11 00:00:00 +0000
date_edited: 2021-05-11 00:00:00 +0000
categories:
comments: true
nolink: false
---

Two weeks ago, ["The Fourier transform is a neural network"](https://sidsite.com/posts/fourier-nets/) [1], reached the front page of Hacker News and gained traction on Twitter.

However, it did face some criticism:

> Is there anything deep here? There's a parametrized representation of a function and it's being interpreted as a neural network, why is this surprising?...
[source](https://news.ycombinator.com/item?id=26981283) (top comment)

> This is kind of silly. Neural networks are universal function approximators, meaning any function "is" (more accurately, "can be represented by") a neural network... [source](https://news.ycombinator.com/item?id=26981090)

> ...Where do we go next ? -- look addition is a neural network, look weighted average is a neural network, look linear regression, logistic regression, Poisson regression are neural networks... [source](https://news.ycombinator.com/item?id=26980700)

These are all good points, and thanks also to the many other people who commented.

The post ended with the question: 

> Do we ever benefit from explicitly putting Fourier layers into our models?

And two days ago, Lee-Thorp et al. released [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824) [2], which addresses this question. From the abstract:

> ...we find that replacing the self-attention sublayer in a Transformer  encoder  with  a  standard,  unparameterized Fourier Transform achieves 92% of the accuracy of BERT on the GLUE benchmark, but pre-trains and runs up to seven times faster on GPUs and twice as fast on TPUs. The resulting model, which we name FNet, scales very efficiently to long inputs, matching the accuracy of the most accurate “efficient” Transformers on the Long Range Arena benchmark, but training  and  running  faster  across  all  sequence lengths on GPUs and relatively shorter sequence lengths on TPUs. Finally, FNet has a light memory footprint and is particularly efficient at smaller model sizes: for a fixed speed and accuracy budget, small FNet models outperform Transformer counterparts.

Not only did they address the question that the post posed, but they also used the technique 
presented  (with code) 
in the post, i.e. creating a "Fourier matrix", to use as a Fourier layer:

> On  TPUs:  for  relatively  shorter  sequences (≤8192 tokens), it is faster to precompute the DFT matrix and then compute the Fourier Transform through matrix multiplications than using the FFT; for longer sequences, the FFT is faster.

Really interesting work; great to see!

---

[1] [The Fourier transform is a neural network](https://sidsite.com/posts/fourier-nets/)

[2] [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824)