# Learning Machine Learning

This repository will basically document my progress in learning some basics of ML. I'm planning on implementing some algorithms by hand to get a feel for them, as well as learning some tools which will obviously outperform any naive implementation I do myself. 

[Genetic Algorithm](./genetic-algorithm)
-----------------

This is an example of a genetic algorithm which performs a polynomial fit of specified degree. The script simulates data for testing, but would of course also work on data of unknown origin. It treats lists of polynomial coefficients as individuals in a population. The initial population is of randomly generated individuals, with "sensible" coefficients: the coefficient of *x*^*n* is not larger than (range of data)^1/*n*.

Each individual (or rather, the polynomial corresponding to each individual) is then evaluated for goodness of fit to the data. Based on their performance, individuals breed, with those who perform better having a higher chance of breeding. The population size remains constant over generations. A child of two parents has a 50% chance of inheriting each coefficient from either one of its parents, and there is also some chance of mutation. This cycle repeats with each generation.

The process can stop adaptively, once the top 10% of individuals in the population has remained static enough for several consecutive generations. Optionally, the progress of the population converging onto a (hopefully) optimal solution can be plotted live using `matplotlib`, and/or the process can automatically be made into a gif for later viewing using Image Magick. Some sample gifs are available in [./genetic-algorithm/figs](./genetic-algorithm/figs).
