---
layout: post
title: "Training a Simple Transformer Neural Net on Conway's Game of Life"
date: 2024-07-07 00:00:00 +0000
date_edited: 2024-07-07 00:00:00 +0000
categories:
comments: true
thumbnail: /assets/posts/life-transformer/attention_matrix_training.gif
---

{%- include mathjax.html -%}

We create a simplified transformer neural network,
and train it compute Conway's Game of Life
just by showing it examples of the game.

We call the model SingleAttentionNet,
because it uses just a single attention block, 
with single-head attention.

Before we get into the details, here's a Life game, 
computed by a SingleAttentionNet model.

<!-- TODO: revert image paths to /assets rather than ../assets -->
<p align="center">
<img 
    src="/assets/posts/life-transformer/life_grid_computed_by_transformer.gif"
    alt="Life game computed by a SingleAttentionNet model"
/>
</p>

And the following plot shows examples of the SingleAttentionNet model's attention matrix, over the course of training:


<p align="center">
<img 
    src="/assets/posts/life-transformer/attention_matrix_training.gif"
    alt="Life game computed by a SingleAttentionNet model"
/>
</p>

The pattern that emerges is the model learning to attend to just the 8 neighbours of each cell.
The attention of the model becomes nearly equivalent to a 3x3 average pool,
as is used in convolutional neural networks,
although unlike an average pool,
it excludes the middle cell from the average.
It is vastly more efficient to directly use an average pool, 
rather than an attention layer, 
but it's interesting to show that the attention layer can learn to approximate it.
(We found that average pooling does also work, even with the middle cell included.)

(For a recap of the rules of Life, check out the [appendix](#the-rules-of-life).)

## Problem formulation and training loop

We formulate the problem as:

```python
next_life_grid = model(life_grid)
```

Which means the model will take a `life_grid` as input, 
and the output will be the state of the grid in the next step,
`next_life_grid`.

If we usually run Life games using the function, `life_step`, e.g.

```python
for _ in range(num_steps):
    life_grid = life_step(life_grid)
```

We could replace that function with our model:

```python
for _ in range(num_steps):
    life_grid = model(life_grid)
```

In order to train our model, 
we show it many examples of 
`(life_grid, next_life_grid)` pairs. 
We can generate a practically limitless amount of these,
by randomly initialising grids and running the Game of Life on them.
The following plot shows some examples, 
where each row represents a pair.

<!-- TODO: white background -->
<p align="center">
<img 
    src="/assets/posts/life-transformer/training_examples.png"
    alt="Life game computed by a SingleAttentionNet model"
/>
</p>

And here is a simplified version of the training loop we will use:

```python
for life_grid, next_life_grid in life_data_generator():
    predicted_next_life_grid = model(life_grid)
    loss = loss_fn(predicted_next_life_grid, next_life_grid)
    run_gradient_descent_step(model, loss)
```

## The model

Our model uses embeddings to represent a Life grid as a set of tokens, with one token per grid cell.
These tokens then go through single-head attention, a hidden layer, 
and a classifier head, which classifies each token/grid cell as dead or alive in the next step.

This section presents code for the model,
then a diagram of the model, 
and finally a mode detailed description of the model.

### Model code

```python
class SingleAttentionNet(torch.nn.Module):

    def __init__(self, grid_dim: int, ndim: int):
        super().__init__()
        self.num_grid_cells = grid_dim * grid_dim
        self.sqrt_ndim = math.sqrt(ndim)
        self.W_state = weight_matrix(h=2, w=ndim, type="embedding")
        self.W_positional = weight_matrix(h=self.num_grid_cells, w=ndim, type="embedding")
        self.W_q = weight_matrix(h=ndim, w=ndim, type="weight")
        self.W_k = weight_matrix(h=ndim, w=ndim, type="weight")
        self.W_v = weight_matrix(h=ndim, w=ndim, type="weight")
        self.W_h = weight_matrix(h=ndim, w=ndim, type="weight")
        self.W_c = weight_matrix(h=ndim, w=1, type="weight")

    def forward(self, life_grids: torch.Tensor) -> dict[str, torch.Tensor]:
        # The input is a batch of grids,
        # life_grids.shape = [b, grid_dim, grid_dim],

        # Flatten the grids
        x = life_grids.reshape(-1, self.num_grid_cells)  # [b, num_grid_cells]

        # Use the embeddings to represent the grids as tokens
        x = self.W_state[x] + self.W_positional  # [b, num_grid_cells, ndim]

        # Single-head attention
        q = x @ self.W_q  # [b, num_grid_cells, ndim]
        k = x @ self.W_k  # [b, num_grid_cells, ndim]
        attention_matrix = torch.softmax(
            q @ k.transpose(-1, -2) / self.sqrt_ndim, dim=-1
        )  # [b, num_grid_cells, num_grid_cells]
        v = x @ self.W_v  # [b, num_grid_cells, ndim]
        x = x + attention_matrix @ v  # skip connection, [b, num_grid_cells, ndim]

        # # Hidden layer
        x = x + torch.nn.functional.silu(x @ self.W_h)  # [b, num_grid_cells, ndim]

        # # Classifier head
        x = x @ self.W_c  # [b, num_grid_cells, 1]

        return x, attention_matrix
```

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

### Detailed model description

#### Input tokens

The model represents each grid cell of the Game of Life as a token, (a vector of size `ndim`).
A given model instance will be trained on a fixed size Life grid.
The model will construct its *input tokens* by adding *positional embeddings* to *cell state embeddings*.
There will be one positional embedding for each grid cell.
There will be two cell state embeddings — one to represent that a cell is alive, and one to represent that a cell is dead.
The embeddings will be randomly initialised, and then learned through gradient descent.

#### Single-head attention

From the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762), single head attention will compute a weighted sum of a projection of the input tokens, for each token, as determined by a square *attention matrix* that is computed for each `before_grid`.

If we call the input tokens, $T$, then we can write single-head attention as, $x = softmax( \frac{ (T W_q) (T W_k )^T}{√d_k} ) (T W_v)$, where $W_q$, $W_k$ and $W_v$ are weight matrices that will be randomly initialised and then learned through gradient descent. The formula is more commonly written as $x = softmax( \frac{QK^T}{√d_k} )V$, where $Q = T W_q$, $K = T W_k$ and $V = T W_v$. We refer to $Q$, $K$ and $V$ as linear projections of the input tokens. We can break the formula up into an attention matrix, $A = softmax( \frac{QK^T}{√d_k})$, and the attention output, $x = AV$. The attention output will be a set of tokens, of the same shape as the input tokens, where each output token is a weighted sum of a linear projection of the input tokens, (i.e. a weighted sum of all the rows in $V$). The factor, $\frac{1}{√d_k}$, is a constant for a given model instance, (called `ndim` in the code below). The softmax is applied to each row of the attention matrix, making the values positive and each row sum to 1, it also tends to make the larger values in each row larger relative to the smaller values in the row.

#### Hidden layer

This is a single neural network layer, that operates on each token individually. It uses a weight matrix that will be randomly initialised and learned through gradient descent. The hidden layer and the single-head attention layer that precedes it can together be referred to as an "attention block"; where transformers typically have multiple attention blocks in series.

#### Classifier layer

This will take each of the tokens above, (recall there's one for each grid cell), and decide whether it should be dead or alive in the next Life step. It uses a weight matrix that will be randomly initialised and learned through gradient descent.

## Training

On a GPU, training the model takes anywhere from a couple of minutes, 
to 10 minutes, or a seemingly indefinite amount of time,
depending on the seed, and other training hyperparameters.

<p align="center">
<img 
    src="/assets/posts/life-transformer/training_progress.png"
    alt="Life game computed by a SingleAttentionNet model"
/>
</p>

## Appendix

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
Life - https://arxiv.org/pdf/2009.01398.pdf

- McGuigan - 2021 - Its Easy for Neural Networks To Learn Game of Life - https://www.kaggle.com/code/jamesmcguigan/its-easy-for-neural-networks-to-learn-game-of-life

- Vaswani et al - 2017 - Attention Is All You Need - https://arxiv.org/abs/1706.03762 

- Conway's Game of Life - https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
