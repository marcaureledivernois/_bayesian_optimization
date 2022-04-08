## Bayesian Optimization

Optimization of a function : find an maximum or minimum of the target function with a 
certain set of parameter combination. In theory, roots of f'(x) = 0 
will be the optimal values of parameter x which in turn gives 
the optimal value of the function f(x). It is simple as long as you know 
the full algebraic form of the function f(x). But, in many real life scenarios, 
things are not so simple. If it is a black box, then you will only know output & 
input values of any f(x), but the full form will be unknown to you. So, 
analytically you cannot find the derivatives. On top of that, the function may be very 
costly to invoke. 

# Application : Finding out the optimal hyperparameter combination

Consider a large & complex Neural Network that solves a classification problem. 
Hyperparameters may be the number of hidden units, number of hidden layers, 
etc in that network. Relation between these can be thought as a hypothetical 
function that takes the hyperparameters as input and returns classification 
accuracy as output. Off course, you don’t know the actual algebraic form of 
this function . One way of finding the optimal combination will be trying out 
various random combinations by training the network repeatedly. The root of 
the problem lies there. We cannot afford that repeated training as it is a 
very large & complex network and training takes a lot of time & resources. 
Bayesian optimization can help here.

# Idea

In Bayesian Optimization, an initial set of input/output combination is 
generally given as said above or may be generated from the function.
Neural Network is trained a number of times on different hyper-parameter 
combinations and the accuracies are captured & stored. 
This set can be used as initial data points.

In short, it is a constrained optimization which solves two problem as given below:
1. Finding out the optimal parameters that give optimal value of the black box 
function in a numerical way as analytically derivatives cannot be found.
2. Keeping the number of function calls in the overall process as minimum as possible 
as it is very costly. (Apart from initial few runs)

# How does it work?

Bayesian approach is based on statistical modelling of the "costly" 
function and intelligent exploration of the parameter space.

# Concepts

1. Surrogate Model

It is the statistical/probabilistic modelling of the “costly” function. 
It works as a proxy to the later. For experimenting with different parameters, 
this model is used to simulate function output instead of calling the actual costly function. 
A Gaussian Process Regression (it is a multivariate Gaussian Stochastic process) is used as 
“surrogate” in Bayesian Optimization.

2. Acquisition Function

It is a metric function which decides which parameter value that can return 
the optimal value from the function. There are many variations of it. 
We will work with the one called “Expected Improvement”.

3. Exploration vs Exploitation

It is typical strategy to compensate between local & global optimal values in the parameter space.
While doing the parameter space exploration, many such local optimal data points 
can be found where the function has high or low value. But, the process should 
not stop there as more optimal values may be there in some other area. It is known as 
“Exploration”. On the other hand, importance should also be given to the points those 
are returning optimal (high or low) values from the function consistently. It is “Exploitation”. 
So, both have some significance. It is a trivial decision, 
“when to explore for more optimal data points in different locations or when to exploit 
& go in the same direction”. This is the area where Bayesian Optimization beats 
traditional Random search or Grid search approach for parameter space as it takes a 
middle ground. It helps to achieve the target more quickly with a small number of actual function calls.



## Credits

* [Towards Data Science - Avishnek Nag](https://towardsdatascience.com/bayesian-optimization-a-step-by-step-approach-a1cb678dd2ec)