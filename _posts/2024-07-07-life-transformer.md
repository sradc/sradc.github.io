---
layout: post
title: "Training a Simple Transformer Neural Net on Conway's Game of Life"
date: 2024-07-07 00:00:00 +0000
date_edited: 2025-05-17 00:00:00 +0000
categories:
comments: true
thumbnail: /assets/posts/life-transformer/attention_matrix_training.gif
---

{%- include mathjax.html -%}

We find that a highly simplified transformer neural network
is able to compute [Conway's Game of Life](https://www.youtube.com/watch?v=R9Plq-D1gEk) perfectly, 
just from being trained on examples of the game.

The simple nature of this model allows us to look at its structure
and observe that it really is computing the Game of Life
— it is not a statistical model that predicts the most likely next state based 
on all the examples it has been trained on.

We observe that it learns to use its attention mechanism to compute `3x3` convolutions.

We refer to the model as SingleAttentionNet, 
because it consists of a single attention block, 
with single-head attention. 
The model represents a Life grid as a set of tokens,
with one token per grid cell.

The following figure shows a Life game, computed by a SingleAttentionNet model:

<p align="center">
<img 
    src="/assets/posts/life-transformer/life_grid_computed_by_transformer.gif"
    alt="Life game computed by a SingleAttentionNet model"
/>
</p>

The following figure shows examples of the SingleAttentionNet model's attention matrix, over the course of training:

<p align="center">
<img 
    src="/assets/posts/life-transformer/attention_matrix_training.gif"
    alt="Life game computed by a SingleAttentionNet model"
/>
</p>

This shows the model learning to compute a 3 by 3 average pool via its attention mechanism, 
(with the middle cell excluded from the average).

## Details

The full code is made available in the Jupyter notebook [here](https://github.com/sradc/training-a-simple-transformer-on-conways-game-of-life/blob/main/main.ipynb) on GitHub.

The problem is framed as:

```python
model(life_grid) = next_life_grid
```

Where gradient descent is used to minimize the loss:

```python
loss = cross_entropy(true_next_life_grid, predicted_next_life_grid)
```

Life grids are generated randomly, 
to provide a limitless source of training pairs,
`(life_grid, next_life_grid)`. Some examples:

<p align="center">
<img 
    src="/assets/posts/life-transformer/training_examples.png"
    alt="Life game computed by a SingleAttentionNet model"
/>
</p>

### Model diagram

The model in the diagram processes 2-by-2 Life grids, which means 4 tokens in total per grid. Blue text indicates parameters that are learned via gradient descent. The arrays are labelled with their shape, (with the batch dimension omitted).

<figure class="image">
<p align="center">
<img 
    src="/assets/posts/life-transformer/simple_transformer_detailed.drawio.png"
    alt="Detailed diagram of SimpleTransformer"
    width=500
/>
</p>
</figure>


### Training

On a GPU, training the model takes anywhere from a couple of minutes, 
to 10 minutes, or fails to converge, depending on the seed and other training hyperparameters.
The largest grid size we successfully trained was 16x16.

<p align="center">
<img 
    src="/assets/posts/life-transformer/training_progress.png"
    alt="Life game computed by a SingleAttentionNet model"
/>
</p>

### Notes

We tried replacing the attention layer of the model with a manually computed Neighbour Attention matrix,
and found the model learned its task far quicker, and generalised to arbitrary grid sizes.
We found that the same was true for replacing the layer with a 3-by-3 average pool.

We checked that the model worked by looking for `1024` batches with 100% accuracy, 
and then testing the model on 100 Life games for 100 steps each.

We found that training it on just the first Life step after randomly initialising a grid 
wasn't enough for it to pass the 100 Life games for 100 steps test,
and so randomly introduced pairs with an extra Life step taken.

### The rules of Life

Life takes place on a 2D grid with cells that are either dead or alive, (represented by 0 or 1). 
A cell has 8 neighbours, which are the cells immediately next to it on the grid.

To progress to the next Life step, the following rules are used:

- If a cell has 3 neighbours, it will be alive in the next step, regardless of it's current state, (alive or dead).
- If a cell is alive and has 2 neighbours, it will stay alive in the next step.
- Otherwise, a cell will be dead in the next step.

These rules are shown in the following plot.

<p align="center">
<img 
    src="/assets/posts/life-transformer/life_state_diagram.png"
    alt="Life game computed by a SingleAttentionNet model"
/>
</p>

## References:

- Springer et al - 2020 - It’s Hard For Neural Networks to Learn the Game of
Life - [https://arxiv.org/abs/2009.01398](https://arxiv.org/abs/2009.01398)

- McGuigan - 2021 - Its Easy for Neural Networks To Learn Game of Life - [https://www.kaggle.com/code/jamesmcguigan/its-easy-for-neural-networks-to-learn-game-of-life](https://www.kaggle.com/code/jamesmcguigan/its-easy-for-neural-networks-to-learn-game-of-life)

- Vaswani et al - 2017 - Attention Is All You Need - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) 

- Conway's Game of Life - [https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)


## Citation:

```
@misc{radcliffe_life_transformer_2024,
  title={Training a Simple Transformer Neural Net on Conway's Game of Life},
  url={https://sidsite.com/posts/life-transformer/},
  howpublished={Main page: \url{https://sidsite.com/posts/life-transformer/}, GitHub repository: \url{https://github.com/sradc/training-a-simple-transformer-on-conways-game-of-life}},
  author={Radclffe, Sidney},
  year={2024},
  month={July}
}
```
