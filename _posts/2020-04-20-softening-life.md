---
layout: post
title:  "Softening Life, to differentiate"
date:   2020-06-01 00:00:00 +0000
date_edited: null
categories:
comments: true
nolink: true
---
{%- include mathjax.html -%}

[TODO: GIFS / images / plots]

Finding the parent of a Life [[1]](#wiki) state is computationally expensive, and not necessarily possible [[2]](#eden).
However, switching to a differentiable approximation of Life
allows gradient descent to be used to find parent states.
The main downside is that the parent states found via this method are often approximate.


## Related Works

'Conway's Gradient of Life' [[3]](#hardmath123), presents the same idea as this post.

'Finding Mona Lisa in the Game of Life' [[4]](#kevingal), describes using a SAT solver to find parent states in Life, notably using it to find the parent state of an image of the Mona Lisa.

'Reversing the Game of Life for Fun and Profit' [[5]](#nbickford) presents various algorithms for finding a parent state.


## Hard Life

*The original Game of Life*

A finite Life board $b$ can be represented by a 2D array. An element of $b$ with value 0 represents a dead cell, 
and an element with value 1 represents a live cell.

The state of the board at time $t+1$ depends only on the state of the board at time $t$ (time in Life is discrete).

Going from $t$ to $t+1$:

- A live cell survives only if it has 2 or 3 neighbours. 
    
- A dead cell comes to life if it has 3 neighbours.

This can be written using the state transition function $f$.

$$ s_{t+1} = f(s_t, n_t) $$
<br/>

$$ s_{t+1} = s_t * (n_t == 2 | n_t == 3) + (1 - s_t) * (n_t == 3) $$
<br/>

Where $s_t$ is the state of the cell at time $t$, and $n_t$ is the number of neighbours the cell has at time $t$.

[plot the state transition function]

## Softening Life

- Allow elements of $b$ to take values in the range [0, 1]. E.g. 0.5 might represent a half alive cell.

- Replace the state transition function $f$ with a function that is differentiable.

Note that neighbour counts can be computed via convolution, and that this is differentiable.


The approximation presented here is equivalent to the Game of Life only if the inputs are 
binary, and the outputs are rounded, s.t. they become binary (for each time step).




## References

<a id="wiki" href="https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life">
[1] https://en.wikipedia.org/wiki/Conway's_Game_of_Life
</a>

<a id='eden' href='https://en.wikipedia.org/wiki/Garden_of_Eden_(cellular_automaton)'>
[2] https://en.wikipedia.org/wiki/Garden_of_Eden_(cellular_automaton)
</a>

<a id="hardmath123" href="http://hardmath123.github.io/conways-gradient.html ">
[3] http://hardmath123.github.io/conways-gradient.html
</a>

<a id="kevingal" href="https://kevingal.com/blog/mona-lisa-gol.html">
[4] https://kevingal.com/blog/mona-lisa-gol.html
</a>

<a id="nbickford" href="https://nbickford.wordpress.com/2012/04/15/reversing-the-game-of-life-for-fun-and-profit/">
[5] https://nbickford.wordpress.com/2012/04/15/reversing-the-game-of-life-for-fun-and-profit/
</a>
