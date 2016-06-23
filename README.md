# Learning Machine Learning

This repository will basically document my progress in learning some basics of ML. I'm planning on implementing some algorithms by hand to get a feel for them, as well as learning some tools which will obviously outperform any naive implementation I do myself. 

[Genetic Algorithm](./genetic-algorithm)
-------------------

This is an example of a genetic algorithm which performs a polynomial fit of specified degree. The script simulates data for testing, but would of course also work on data of unknown origin. It treats lists of polynomial coefficients as individuals in a population. The initial population is of randomly generated individuals, with "sensible" coefficients: the coefficient of *x*^*n* is not larger than (range of data)^1/*n*.

Each individual (or rather, the polynomial corresponding to each individual) is then evaluated for goodness of fit to the data. Based on their performance, individuals breed, with those who perform better having a higher chance of breeding. The population size remains constant over generations. A child of two parents has a 50% chance of inheriting each coefficient from either one of its parents, and there is also some chance of mutation. This cycle repeats with each generation.

The process can stop adaptively, once the top 10% of individuals in the population has remained static enough for several consecutive generations. Optionally, the progress of the population converging onto a (hopefully) optimal solution can be plotted live using `matplotlib`, and/or the process can automatically be made into a gif for later viewing using Image Magick. Some sample gifs are available in [./genetic-algorithm/figs](./genetic-algorithm/figs).

[(Multiple) Linear Regression](./linear-regression)
------------------------------

Here we have the well-known linear regression. This is the first in the series of algorithms covered by sentdex [here](https://pythonprogramming.net/machine-learning-tutorial-python-introduction/). The tutorials cover the `LinearRegression` classifier from scikit-learn, and talks in great detail about performing simple linear regression by hand. Using [these notes](http://dept.stat.lsa.umich.edu/~kshedden/Courses/Stat401/Notes/401-multreg.pdf) I extended this script to do multiple linear regression, and compared my own implementation with that of scikit-learn. 

The comparison script uses Quandl to get some of Google's historical stock information as a Pandas dataframe, and uses some of this information as features. We assume that the features of the current day are linearly related to the stock price in 30 days. After a bit of processing, the data is used to train the classifier and to produce the coefficients of the by-hand implementation. Each model is used to predict stock prices for the coming 30 days. The results of scikit-learn and our own implementation are extremely similar, which is to be expected since there isn't really anything probabilistic going on here, so the only differences are rounding errors.