---
layout: post
title:  "Implementing Naive Bayes in Python"
date:   2020-02-28 19:00:35 +0000
date_edited: 2020-03-02 00:00:00 +0000
categories: machine-learning python
comments: true
---
{%- include mathjax.html -%}


[Sidney Radcliffe](https://sidsite.com/) | [Github](https://github.com/sradc/MyNotebooks/blob/master/notebooks/Implementing%20Naive%20Bayes%20in%20Python.ipynb)


**Implementing a [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) machine learning classifier in Python.** Starting with a basic implementation, and then improving it. 
Libraries used: NumPy, Numba (and scikit-learn for comparison).

<table style="float:centre">

<tr>
    <td><a href="#first-implementation">First implementation</a></td>
    <td>A basic implementation of Naive Bayes.</td>
</tr>
<tr>
    <td><a href="#second-implementation">Second implementation</a></td>
    <td>The method is improved.</td>
</tr>
<tr>
    <td><a href="#background-to-first-implementation">Background to first implementation</a></td>
    <td>The theory behind the first implementation.</td>
</tr>
<tr>
    <td><a href="#background-to-second-implementation">Background to second implementation</a> </td>
    <td>The theory behind the second implementation.</td>
</tr>
<tr>
    <td><a href="#third-implementation">Third implementation</a></td>
    <td>A more practical implementation.</td>
</tr>
<tr>
    <td><a href="#references">References</a></td>
    <td></td>
</tr>
</table>


---

## First implementation

#### Generate toy data:

We'll assume our data is categorical and encoded as: `0, 1, ..., num_values_in_feature`.


```python
import numpy as np
np.random.seed(0)

def generate_data(n, m, rx, ry=3):
    X = np.random.randint(rx, size=(n, m))
    y =  np.random.randint(ry, size=(n))
    return X, y

N = 20
M = 4

X, y = generate_data(N, M, 3, 3)

print(X[:5, :])
print(y[:5])
```

    [[0 1 0 1]
     [1 2 0 2]
     [0 0 0 2]
     [1 2 2 0]
     [1 1 1 1]]
    [0 2 1 1 1]
    

### Train model

Using counts, we estimate the values of `probabilities[f][v][c]`, the probability of feature `f` having value `v`, given that the class is `c`.


```python
classes = np.unique(y)
class_priors = [sum(y==c)/N for c in classes]

def get_probs_of_values(column):
    values = np.unique(column)
    p = [0]*len(values)
    for v in values:
        p[v] = [0]*len(classes)
        for c in classes:
            p[v][c] = sum((column==v) & (y==c)) / sum(y==c)
    return p

probabilities = [0]*M
for f in range(M):
    column = X[:, f]
    probabilities[f] = get_probs_of_values(column)
```

### Make predictions
... using `probabilities[f][v][c]`.


```python
def get_class_probabilities(x):
    p_of_class = {}
    for c in classes:
        posterior = 1
        for f in range(M):
            v = x[f]
            posterior *= probabilities[f][v][c]
        prior = class_priors[c]
        p_of_class[c] = prior * posterior
    return p_of_class

def get_most_probable_class(x):
    class_probabilities = get_class_probabilities(x)
    return max(class_probabilities, key=class_probabilities.get)

def predict_classes(X):
    return [get_most_probable_class(row) for row in X]

predictions = predict_classes(X)
print("predictions =", predictions)
print(f"Accuracy on training data: {sum(y == predictions)/len(y)*100:.1f}%")
```

    predictions = [0, 0, 2, 1, 1, 0, 0, 1, 2, 0, 1, 2, 1, 1, 2, 1, 2, 0, 1, 2]
    Accuracy on training data: 70.0%
    

### Notes

#### Train model

- `classes`
    - the only classes the model is aware of are those in the training data, i.e. the unique values in `y`


- `class_priors`
    - estimate class prior probabilities from the data
    - could remove the 1/N term, since it doesn't affect results
    

- `column = X[:,f]` is column `f` of training data `X`


- `get_probs_of_values(column)`
    - Because of the Naive Bayes independence assumption, we can look at each feature in isolation
    - `column` is a column of the training data, corresponding to measurements of feature `f`
    - `values` is our estimate of all the possible values that can occur in feature `f`
    - For every value in `values`, we estimate the probability of that value occuring given class `c`, for all possible classes  
    - `p` contains our estimates
    - `p[v][c]` $\approx$ `p(v | c)`, the probability of class `c` given that we have value `v`
    - `p[v][c] = (number of times the value v has been of class c) / (number of times class c has occurred)`
    - The counts are calculated using elementwise boolean operations on numpy arrays
    
      
      
We end up with `probabilities[f][v][c]`, where an element is the probability of feature `f` having value `v`, given that the class is `c`.

- `probabilities[f][v][c]` $\approx$ `p(v in column f | c)`

- `probabilities[f][v][c] = (number of times the value v in column f has been of class c) / (number of times class c has occurred)`


#### Make predictions

- `get_class_probabilities(x)`
    - Use the values in `probabilities` to estimate the probability of each of the classes, given vector `x`.
    - `x` is a vector indexed by `f`. For example, it could be a row of `X`.
    - `for c in classes`
        - for each class, `c`, the inner loop calculates the probability of class `c` given `x`:
        - `p_of_class[c]` $\approx$ `p(c | x)`
        - `p_of_class[c] = p(c) * p(x[0] | c) * p(x[1] | c) * ... * p(x[M] | c)`
        - `p_of_class[c] = count(y==c)/N * probabilities[0][x[0]][c] * probabilities[1][x[1]][c] * ... * probabilities[M-1][x[M-1]][c]`
        

- `get_most_probable_class(x)`
    - Identify the class with the highest probability; this is our predicted class for `x`    


- `predict_classes(X)`
    - Allows us to make predictions for multiple vectors that have been stored in a matrix, of width `M`


**The section <a href="#background-to-first-implementation">"Background to first implementation"</a> explains the theory of the method in more detail.**

---

## Second implementation

*Like the first implementation but modified to use smoothing and log probabilities. More explanation in <a href="#background-to-second-implementation">"Background to second implementation"</a>.*


### Train model

Estimate values of `probabilities[f][v][c]`, now the **smoothed log probability** of feature `f` having value `v`, given that the class is `c`.


```python
alpha = 0.1
classes = np.unique(y)
class_priors = [np.log(sum(y==c)) for c in classes]

def get_probs_of_values(column):
    values = np.unique(column)
    p = [0]*len(values)
    for v in values:
        p[v] = [0]*len(classes)
        for c in classes:
            p[v][c] = (np.log(sum((column==v) & (y==c)) + alpha)
                       - np.log(sum(y==c) + alpha*len(values)))
    return p

probabilities = [0]*M
for f in range(M):
    column = X[:, f]
    probabilities[f] = get_probs_of_values(column)
```

### Make predictions
Now we add together log probabilities, instead of multiplying probabilities.


```python
def get_class_probabilities(x):
    p_of_class = {}
    for c in classes:
        posterior = 0
        for f in range(M):
            v = x[f]
            posterior += probabilities[f][v][c]
        prior = class_priors[c]
        p_of_class[c] = prior + posterior
    return p_of_class

def get_most_probable_class(x):
    class_probabilities = get_class_probabilities(x)
    return max(class_probabilities, key=class_probabilities.get)

def predict_classes(X):
    return [get_most_probable_class(row) for row in X]

predictions = predict_classes(X)
print("predictions =", predictions)
print(f"Accuracy on training data: {sum(y == predictions)/len(y)*100:.1f}%")
```

    predictions = [0, 0, 2, 1, 1, 0, 0, 1, 2, 0, 1, 2, 1, 1, 2, 1, 2, 0, 1, 2]
    Accuracy on training data: 70.0%
    

Some practical improvements are made this implementation in the <a href="#third-implementation">third implementation</a>.


**The changes to the first implementation are motivated in <a href="#background-to-second-implementation">"Background to second implementation"</a>.**

---

## Background to first implementation

### An expression to maximise

We'll represent instance $\mathbf{x}$ as a vector, where element $x_f$ contains a measurement of feature $f$:

$$ \mathbf{x} = [x_0, \dots, x_f, \dots, x_{M-1}] $$

Let $y$ be the class of instance $\mathbf{x}$. 

For an $\mathbf{x}$ with an unknown class $y$, we can classify $\mathbf{x}$ by predicting $y$ to be the class $c$ that has the highest probability, given the value of $\mathbf{x}$. I.e. we want to find the $c$ that maximises:

$$P(y = c \mid \mathbf{x} = [v_0, \dots, v_{M-1}])$$

Where $v_f$ is the value of element $x_f$.

More formally, we say that our predicted class $\hat{y}$ will equal the $c$ that maximises $P(y = c \mid \mathbf{x} = [v_0, \dots, v_{M-1}])$:

$$\hat{y} = \arg\max_c P(y = c \mid \mathbf{x} = [v_0, \dots, v_{M-1}]) $$

Bayes' theorem lets us write:

$$P(y = c \mid \mathbf{x} = [v_0, \dots, v_{M-1}]) = \frac{P(y = c) P(\mathbf{x} = [v_0, \dots, v_{M-1}] \mid y = c)}{P(\mathbf{x} = [v_0, \dots, v_{M-1}])}$$

Unfortunately this is difficult to calculate. However, Naive Bayes sidesteps the difficulty, by making the simplifying assumption that the value of a particular feature is independent of the value of any other feature, given the class variable:

$$P(x_f = v_f | y=c, x_0=v_0, \dots, x_{f-1}=v_{f-1}, x_{f+1}=v_{f+1}, \dots, x_{M-1}=v_{M-1}) = P(x_f = v_f | y = c)$$

This allows us to write:

$$P(\mathbf{x} = [v_0, \dots, v_{M-1}] \mid y = c) = \prod_{f=0}^{M-1} P(x_f = v_f \mid y = c) $$

And so the expression we base our classification on becomes:

$$P(y = c \mid \mathbf{x} = [v_0, \dots, v_{M-1}]) = \frac{P(y=c) \prod_{f=0}^{M-1} P(x_f = v_f \mid y = c)}{P(\mathbf{x} = [v_0, \dots, v_{M-1}])}$$

Since we are looking for the $c$ that maximises this expression, we can remove the denominator, which does not change with $c$.

$$P(y = c \mid \mathbf{x} = [v_0, \dots, v_{M-1}]) \propto P(y=c) \prod_{f=0}^{M-1} P(x_f = v_f \mid y = c)$$



**Therefore we can write our Naive Bayes classifier as:**

$$\hat{y} = \arg\max_c P(y=c) \prod_{f=0}^{M-1} P(x_f = v_f \mid y = c) $$



Note on terminology:

$P(y = c)$ is called the class prior probability. $P(\mathbf{x} = [v_0, \dots, v_{M-1}] \mid y = c)$ is called the posterior probability.


### Assumptions about our data

We will assume that all of our data is categorical, and encoded as integers, starting at 0 and incremented by 1. There are Naive Bayes implementations that deal with continuous data more directly, but for now let's say that any continuous data has been turned into categorical data, e.g. using bins.


### Estimating the possible classes

Class $c$ is an element of the set of possible classes $C$, i.e. $c \in C$. We need to go through every class in $C$, to find out which class has the largest probability. 

How do we estimate $C$? We say that the classes we see in our training data are the only possible classes that $c$ can be:

$$C \approx unique(\mathbf{y}) $$


### Estimating the probabilities

Given a matrix of training data, $X$ (size $N{\times}M$: $N$ instances, $M$ features), and vector of class labels, $\mathbf{y}$ (length $N$), we can estimate the required probabilities using frequency counts.

$$P(y = c) \approx \frac{count(\mathbf{y}==c)}{N} $$

$$P(x_f = v_f \mid y = c) \approx \frac{count(X[:,f] == v_f {\&} \mathbf{y} == c)}{count(\mathbf{y} == c)} $$

Where '$==$' and '${\&}$' denote elementwise boolean operations on vectors, and $X[:,f]$ is a vector created by taking column $f$ from $X$.

In words:

$P(y = c) \approx$ (the number of instances that are of class $c$) / (the total number of instances)

$P(x_f = v_f \mid y = c) \approx$ (the number of times value $v$ in $X[:,f]$ has been of class $c$) / (the number of instances that are of class $c$)


Alternatively, we could use values of $P(y = c)$ from a different information source than our training data. In which case, we would substitute in the values from the other information source.


### Putting it all together: 

We defined our Naive Bayes classifier as:

$$\hat{y} = \arg\max_c P(y=c) \prod_{f=0}^{M-1} P(x_f = v_f \mid y = c) $$

Putting in our posterior probability estimate from counts, we get:

$$\hat{y} = \arg\max_c \left[ P(y=c) \prod_{f=0}^{M-1} \frac{count(X[:,f] == v_f {\&} \mathbf{y} == c)}{count(\mathbf{y} == c)} \right] $$

Note that $P(y=c)$ is usually also estimated from the training data, but it is also possible to use another source of information (for example, saying that all classes are equally likely). If we estimate $P(y=c)$ from the data, we get:

$$\hat{y} = \arg\max_c \left[ \frac{count(\mathbf{y}==c)}{N} \prod_{f=0}^{M-1} \frac{count(X[:,f] == v_f {\&} \mathbf{y} == c)}{count(\mathbf{y} == c)} \right] $$

This form of the expression is used in the first implementation.

We can slightly simplify the expression. The $N$ doesn't change with $c$, so we can remove it:

$$\hat{y} = \arg\max_c \left[count(\mathbf{y}==c) \prod_{f=0}^{M-1} \frac{count(X[:,f] == v_f {\&} \mathbf{y} == c)}{count(\mathbf{y} == c)} \right] $$

We can also move the denominator out of the product:

$$\hat{y} = \arg\max_c \left[\frac{1}{count(\mathbf{y} == c)}^{M-2} \prod_{f=0}^{M-1} count(X[:,f] == v_f {\&} \mathbf{y} == c) \right] $$


**And that's the theory behind the <a href="#first-implementation">first implementation</a>**

---

## Background to second implementation

*Motivating the <a href="#second-implementation">second implementation</a>.*

There are two major improvements we can make to our Naive Bayes classifier:

1. Smoothing
2. Logs

### Smoothing

Glance up to how we estimate $P(x_f = v_f \mid y = c)$. Unfortunately it will be very common to have counts of zero in the numerator. If we get a zero, the probability for the class will go to zero:

$$\prod_{f=0}^{M-1} P(x_f = v_f \mid y = c) =  P(x_0 = v_0 \mid y = c) * \dots * 0 * \dots* P(x_{M-1} = v_{M-1} \mid y = c)$$

$$= 0$$

$$P(y=c) \prod_{f=0}^{M-1} P(x_f = v_f \mid y = c) =  0$$

$$P(y = c \mid \mathbf{x} = [v_0, \dots, v_{M-1}]) = 0$$

This is undesirable, since it makes the model inflexible - unable to cope with configurations of values and classes unseen in the training data, without zeroing out.

A fix for this is additive smoothing. We modify our estimate:

$$P(x_f = v_f \mid y = c) \approx \frac{count(X[:,f] == v_f {\&} \mathbf{y} == c) + \alpha}{count(\mathbf{y} == c) + \alpha n_f} $$

Where $n_f$ is an estimate of the possible values that feature $f$ can take:

$$n_f = unique(X[:, f])$$

$\alpha$ is a smoothing parameter, usually $0 \leq \alpha \leq 1$. We can search for a value of $\alpha$ that results in good model performance, e.g. using cross validation. 

We can motivate this smoothing by looking at the case where the counts are zero:

$$P(x_f = v_f \mid y = c) \approx \frac{\alpha}{\alpha n_f} = \frac{1}{n_f}$$

The counts are zero, therefore our training data doesn't contain any examples of this particular class-value combination. Without smoothing, our estimate of $P(x_f = v_f \mid y = c)$ would be 0. But now that we have smoothing, our estimate is that all possible values of $x_f$ are equally likely, given class $c$. This is a fair assumption to make if we don't have much data, and it stops the probability from going to 0.

When the counts are non-zero, $\alpha$ blends together the probability derived from the emperical data (i.e. the counts), and the probability based on assuming all values are equally likely.

We could also smooth our estimate of the class prior probability $p(y=c)$. However, because we estimated $C \approx unique(\mathbf{y}) $, we are guaranteed to not have values of 0 for any class prior. Therefore we will leave out the smoothing. 


### Logs

We have improved our implementation by including smoothing, but currently we are still multiplying together potentially very small values; which is not great with floating point numbers. 

Luckily, the class, $c$, that maximises $P(y = c \mid \mathbf{x} = [v_0, \dots, v_{M-1}])$ also maximises $\log{P(y = c \mid \mathbf{x} = [v_0, \dots, v_{M-1}])}$. So we can switch to using log probabilities, and get the same answer. In log space multiplication becomes addition, and the numbers become more reasonably sized.

$$\log{P(y = c \mid \mathbf{x} = [v_0, \dots, v_{M-1}])} \propto \log{P(y=c)} +  \sum_{f=0}^{M-1} \log{P(x_f = v_f \mid y = c)}$$

$$\hat{y} = \arg\max_c \log{P(y = c \mid \mathbf{x} = [v_0, \dots, v_{M-1}])}$$

$$\hat{y} = \arg\max_c \left[ \log{P(y=c)} +  \sum_{f=0}^{M-1} \log{P(x_f = v_f \mid y = c)} \right]$$

### Putting it all together: 

Combining our smoothed estimates with the log probabilities:

$$\hat{y} = \arg\max_c \left[\log{P(y=c)} +  \sum_{f=0}^{M-1} \log{\frac{count(X[:,f] == v_f {\&} \mathbf{y} == c) + \alpha}{count(\mathbf{y} == c) + \alpha n_f}} \right]$$

$$\hat{y} = \arg\max_c \left[ \log{P(y=c)} +  \sum_{f=0}^{M-1} \log{\left(count(X[:,f] == v_f {\&} \mathbf{y} == c) + \alpha\right)} - \sum_{f=0}^{M-1} \log{\left(count(\mathbf{y} == c) + \alpha n_f\right)} \right]$$

The $P(y=c)$ term can either be estimated from the training data, or values from a different information source can be used. If we estimate the values using our training data, we end up with:

$$\hat{y} = \arg\max_c \left[ \log{count(\mathbf{y}==c)} - \log{N} +  \sum_{f=0}^{M-1} \log{\left(count(X[:,f] == v_f {\&} \mathbf{y} == c) + \alpha\right)} - \sum_{f=0}^{M-1} \log{\left(count(\mathbf{y} == c) + \alpha n_f\right)} \right]$$

We can remove the $log{N}$ term because it does not change with $c$, and therefore does not affect our results.

$$\hat{y} = \arg\max_c \left[ \log{count(\mathbf{y}==c)} +  \sum_{f=0}^{M-1} \log{\left(count(X[:,f] == v_f {\&} \mathbf{y} == c) + \alpha\right)} - \sum_{f=0}^{M-1} \log{\left(count(\mathbf{y} == c) + \alpha n_f\right)} \right]$$

For a particular class, $c$, the first and third terms will be constant, given a set of training data, $X$, $\mathbf{y}$.

---

## Third implementation

This implementation computes the same thing as the <a href="#second-implementation">second implementation</a>, but:
- slower parts are sped up
    - using [Numba](http://numba.pydata.org/)
    - using NumPy features, such as [broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html), and ['advanced' indexing](https://numpy.org/devdocs/user/basics.indexing.html)
- a class is used, to create a [scikit-learn](https://scikit-learn.org) style API
- the denominator of the prior probability is computed seperately, since it doesn't depend on a given value



```python
from numba import jit

@jit
def get_counts(X, y, len_feature_values, n_classes):
    counts = []
    for f in range(X.shape[1]):
        col_counts = np.zeros(shape=(len_feature_values[f], n_classes))
        counts.append(_get_column_counts(X[:,f], y, col_counts))
    return counts

@jit
def _get_column_counts(column, y, col_counts):
    N = column.shape[0]
    for i in range(N):
        v = column[i]
        c = y[i]
        col_counts[v, c] += 1
    return col_counts

@jit
def p_of_values(X, probabilities):
    log_probs = np.zeros((X.shape[0], probabilities[0].shape[1]))
    for f in range(X.shape[1]):
        log_probs += probabilities[f][X[:, f], :]
    return log_probs
```


```python
class NaiveBayes:
    
    def __init__(self, alpha=1):
        self.alpha = alpha
    
    def fit(self, X, y):
        self.N, self.M = X.shape
        self.classes = np.unique(y)
        self.class_counts = np.bincount(y)
        self.feature_values = [np.unique(X[:, f]) for f in range(self.M)]
        self.len_feature_values = np.array([len(self.feature_values[f]) 
                                            for f in range(self.M)])
        self._compute_counts(X, y)
        self._compute_probabilities()
        self._compute_terms_1_and_3()
    
    def _compute_counts(self, X, y):
        self.counts = get_counts(
            X, y, self.len_feature_values, len(self.classes))
    
    def _compute_probabilities(self):
        self.probabilities = tuple(np.log(self.counts[f] + self.alpha) 
                                   for f in range(self.M))

    def _compute_terms_1_and_3(self):
        t1 = [np.log(class_count) for class_count in self.class_counts]
        t3 = [-sum(np.log(self.class_counts[c] + self.alpha*self.len_feature_values[f]) 
                        for f in range(self.M)) 
                           for c in self.classes]
        self.terms_1_and_3 = np.array(t1) + np.array(t3)    
    
    def predict(self, X):
        if self._any_unseen_values(X): 
            raise ValueError("X contains values not seen in the training set.")
        probs = p_of_values(X, self.probabilities)
        probs += self.terms_1_and_3
        return np.argmax(probs, axis=1)
    
    def _any_unseen_values(self, X):
        return (X >= self.len_feature_values.reshape([1, -1])).any()
    
```

### Comparison with scikit-learn's CategoricalNB

The main implementation difference is how the counts are computed: this implementation uses a simple loop that's sped up with Numba, whereas the scikit-learn implementation uses NumPy masks (which is slightly less efficient, but presumably more generally compatible than Numba). Also the scikit-learn implementation has more features (and is more thoroughly tested, and is maintained).

#### Results

This implementation gets the same results as the [scikit-learn implementation](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html):


```python
X, y = generate_data(n=100_000, m=40, rx=12, ry=5)
```


```python
from sklearn.naive_bayes import CategoricalNB

ALPHA = .3

clf_sklearn = CategoricalNB(alpha=ALPHA)
clf_sklearn.fit(X, y)

clf = NaiveBayes(alpha=ALPHA)
clf.fit(X, y)

predictions = clf.predict(X)
pred_sklearn = clf_sklearn.predict(X)

print(f"Accuracy on training data: {sum(y == predictions)/len(y)*100:.1f}%")
print(f"Sklearn accuracy on training data: {sum(y == pred_sklearn)/len(y)*100:.1f}%")
print(f"Predictions the same: {100*sum(predictions == pred_sklearn)/len(predictions):.2f}%" )
```

    Accuracy on training data: 23.5%
    Sklearn accuracy on training data: 23.5%
    Predictions the same: 100.00%
    

#### Running times


```python
%timeit clf.fit(X, y)
%timeit clf_sklearn.fit(X, y)
%timeit clf.predict(X)
%timeit clf_sklearn.predict(X)
```

    142 ms ± 1.24 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    255 ms ± 323 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
    136 ms ± 3.67 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    160 ms ± 2.78 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    

#### Unseen values

Both this implementation and scikit-learn's CategoricalNB give errors if trying to predict for an $X$ that contains previously unseen values. The implementation could be made to work with unseen values, here's one way to do so:


```python
@jit
def p_of_values_with_mask(X, probabilities, len_feature_values, log_alpha):
    log_probs = np.zeros((X.shape[0], probabilities[0].shape[1]))    
    for f in range(X.shape[1]):
        mask = X[:, f] < len_feature_values[f]
        index = X[:, f][mask]
        log_probs[mask,:] += probabilities[f][index, :]
        mask = np.logical_not(mask)       
        log_probs[mask,:] += log_alpha
    return log_probs
```

## References

- Data mining: concepts and techniques, Han, Jiawei and Pei, Jian and Kamber, Micheline, 3rd ed.
- [https://scikit-learn.org/stable/modules/naive_bayes.html](https://scikit-learn.org/stable/modules/naive_bayes.html)
    - [https://scikit-learn.org/stable/modules/naive_bayes.html#categorical-naive-bayes](https://scikit-learn.org/stable/modules/naive_bayes.html#categorical-naive-bayes)
    - [https://github.com/scikit-learn/scikit-learn/blob/b194674c4/sklearn/naive_bayes.py#L1012](https://github.com/scikit-learn/scikit-learn/blob/b194674c4/sklearn/naive_bayes.py#L1012)
- [https://en.wikipedia.org/wiki/Naive_Bayes_classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- [https://numpy.org/](https://numpy.org/)
- [http://numba.pydata.org/](http://numba.pydata.org/)
