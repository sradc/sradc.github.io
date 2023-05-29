---
layout: post
title:  "Notes on training BERT from scratch on an 8GB consumer GPU"
date:   2023-05-29 00:00:00 +0000
date_edited:
categories:
comments: true
---

<!-- TODO: link to code -->
<!-- TODO: compare with BERT training resources -->


<p align="center">
<figure>
    <img
        src="/assets/posts/bert-from-scratch/desktop.jpg" 
        alt="Photo of my desktop PC, with its Nvidia 3060 Ti 8GB GPU."
    />
    <figcaption>My desktop PC, with its Nvidia 3060 Ti 8GB GPU.</figcaption>
</figure>
</p>

During my free time, I trained a BERT model ([Devlin et al, 2019](https://arxiv.org/abs/1810.04805)) from scratch on my desktop PC. The model architecture, tokenizer, and training code all came from [Hugging Face](https://huggingface.co/) libraries, and my contribution was mainly setting up the code, setting up the [data](https://huggingface.co/datasets/sradc/chunked-shuffled-wikipedia20220301en-bookcorpusopen) (~20GB uncompressed text), and leaving my computer running.

Bearing in mind that I have thrown considerably less compute at the model than the original BERT paper, here are the [GLUE](https://gluebenchmark.com/) **dev-set** results:

| Model        | MNLI (m/mm) | SST-2 | STSB | RTE  | QNLI | QQP | MRPC | CoLA | Average |
|--|--|--|--|--|--|--|--|--|--|
| This model   | 79.3/80.1   | 89.1  | 61.9 | 55.9 | 86.3 | 86.4 | 74.8 | 41.0 | 72.7 |
| BERT-Base\*   | 83.2/83.4   | 91.9  | 86.7 | 59.2 | 90.6 | 87.7 | 89.3 | 56.5 | 80.9 |

\*BERT-Base refers to a fully trained BERT model, the results are taken from Cramming ([Geiping et al, 2022](https://arxiv.org/abs/2212.14034)).

While we can see that BERT-Base performed better at every task; the results for "this model" would have been very good (possibly SOTA for a few tasks) in early 2018. 

And 




This endeavor was inspired by Cramming ([Geiping et al, 2022](https://arxiv.org/abs/2212.14034)), which tries to find the best ways to train BERT-like models on modest compute resources, with only 24 hours.
 into how to train BERT-like models on modest compute resources, and with only 24 hours.

After pretraining for 100 hours (1 epoch), and fine-tuning for 12 hours,



No hyperparameter tuning was carried out.
No special techniques were used to improve the training.
Optimizer and learning rate schedule were guided by Cramming ([Geiping et al, 2022](https://arxiv.org/abs/2212.14034)), (but the model architecture changes and data ordering suggested by Cramming were not used).

I was able to monitor training remotely, using [Weights & Biases](https://wandb.ai/site).

### Plots from the 100 hours training run

<p align="center">
<figure>
    <img 
        src="/assets/posts/bert-from-scratch/loss.png" 
        alt="The pre-training loss."
    />
    <figcaption>The pre-training loss.</figcaption>
</figure>
</p>

<p align="center">
<figure>
    <img 
        src="/assets/posts/bert-from-scratch/learning_rate.png" 
        alt="The learning rate schedule, recommended by Cramming ([Geiping et al, 2022](https://arxiv.org/abs/2212.14034))."
    />
    <figcaption>The learning rate schedule, recommended by Cramming (Geiping et al, 2022).</figcaption>
</figure>
</p>

<p align="center">
<figure>
    <img 
        src="/assets/posts/bert-from-scratch/gpu_util.png" 
        alt="GPU utilization was around 98%."
    />
    <figcaption>GPU utilization was around 98%.</figcaption>
</figure>
</p>

<p align="center">
<figure>
    <img 
        src="/assets/posts/bert-from-scratch/gpu_memory.png" 
        alt="GPU memory usage was around 98%, this was achieved by adjusting the batch size."
    />
    <figcaption>GPU memory usage was around 98%, this was achieved by adjusting the batch size.</figcaption>
</figure>
</p>

<p align="center">
<figure>
    <img 
        src="/assets/posts/bert-from-scratch/gpu_temp.png" 
        alt="GPU temperature stayed between 76 - 80 degrees celsius, with a higher temperature on hotter days."
    />
    <figcaption>GPU temperature stayed between 76 - 80 degrees celsius, with a higher temperature on hotter days.</figcaption>
</figure>
</p>


### References:
- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805 [cs], May 2019. URL [http://arxiv.org/abs/1810.04805](http://arxiv.org/abs/1810.04805).
- Vaswani et al. (2017) Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. arXiv:1706.03762 [cs], December 2017. URL http://arxiv.org/abs/1706.03762. 
- Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. Improving language understanding with unsupervised learning. Technical report, OpenAI, https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
