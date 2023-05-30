---
layout: post
title:  "Notes on training BERT from scratch on an 8GB consumer GPU"
date:   2023-05-29 00:00:00 +0000
date_edited: 2023-05-29 00:00:00 +0000
categories:
comments: true
---

I trained a BERT model ([Devlin et al, 2019](https://arxiv.org/abs/1810.04805)) from scratch on my desktop PC (which has a Nvidia 3060 Ti 8GB GPU). The model architecture, tokenizer, and training code all came from [Hugging Face](https://huggingface.co/) libraries, and my contribution was mainly setting up the [code](https://github.com/sradc/pretraining-BERT/tree/main), setting up the [data](https://huggingface.co/datasets/sradc/chunked-shuffled-wikipedia20220301en-bookcorpusopen) (~20GB uncompressed text), and leaving my computer running. (And making sure it was working correctly, with good GPU utilization.)

People generally associate the training of large language models with GPU or TPU clusters, rather than desktop PCs, and the following plot illustrates the difference between the compute resources I used to train this model, and the resources used to train the original BERT-base model.

<p align="center">
    <img
        src="/assets/posts/bert-from-scratch/bert_vs_this_model.png" 
        alt="Plot comparing compute resources and model performance on GLUE-dev."
    />
</p>

Notably, although both BERT-base and this model were trained for the same amount of time, BERT-base saw ~30x more tokens of text, (BERT-base saw ~40 epochs of its training data, while this model saw just a single epoch of its training data).

The [GLUE](https://gluebenchmark.com/) **dev-set** score is shown in the plot above, to give an idea of how well the model performs at natural language tasks. 
Fine-tuning on GLUE took ~12 hours in total (on top of the 4 days / ~100 hours of pretraining). 
The following table shows the GLUE-dev results in more detail:

| Model        | MNLI (m/mm) | SST-2 | STSB | RTE  | QNLI | QQP | MRPC | CoLA | Average |
|--|--|--|--|--|--|--|--|--|--|
| This model   | 79.3/80.1   | 89.1  | 61.9 | 55.9 | 86.3 | 86.4 | 74.8 | 41.0 | 72.7 |
| BERT-Base\*   | 83.2/83.4   | 91.9  | 86.7 | 59.2 | 90.6 | 87.7 | 89.3 | 56.5 | 80.9 |

\*BERT-Base refers to a fully trained BERT model, the results are taken from Cramming ([Geiping et al, 2022](https://arxiv.org/abs/2212.14034)).

While we can see that BERT-Base performed better at every task; the results for "this model" would have been very good (possibly SOTA for a few tasks) in early 2018. 

No hyperparameter tuning was carried out.
No special techniques were used to improve the training.
Optimizer and learning rate schedule were guided by Cramming ([Geiping et al, 2022](https://arxiv.org/abs/2212.14034)),
but the model architecture changes and other suggestions in Cramming were not used.
I did a couple of smaller training runs first (~1-12 hours).

I was able to monitor training remotely, using [Weights & Biases](https://wandb.ai/site).

This endeavor was inspired by Cramming ([Geiping et al, 2022](https://arxiv.org/abs/2212.14034)),
a paper on how to train well-performing BERT models, on modest compute resources (in only 24 hours).

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

Code is available [here](https://github.com/sradc/pretraining-BERT/tree/main).

### References:
- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805 [cs], May 2019. URL [http://arxiv.org/abs/1810.04805](http://arxiv.org/abs/1810.04805).
- Vaswani et al. (2017) Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. arXiv:1706.03762 [cs], December 2017. URL http://arxiv.org/abs/1706.03762. 
- Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. Improving language understanding with unsupervised learning. Technical report, OpenAI, https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
