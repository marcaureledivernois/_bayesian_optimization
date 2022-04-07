Optimizing a function is super important in many of the real life analytics use cases. 
By optimization we mean, either find an maximum or minimum of the target function with a 
certain set of parameter combination. Finding out that min or max value as well 
as the parameters should be the objective. In this article, we will discuss about basics 
of optimizing an unknown costly function with Bayesian approach.

Roots of the equation will be the optimal values of parameter x which in turn gives 
the optimal value of the function f(x). It is simple as long as you know 
the full algebraic form of the function f(x). But, in many real life scenarios, 
things are not so simple. If it is a black box, then you will only know output & 
input values of any f(x), but the full form will be unknown to you. So, 
analytically you cannot find the derivatives. On top of that, the function may be very 
costly to invoke. Consider two use cases like below:
1. Finding out the optimal hyperparameter combination of a neural network
Consider a large & complex Neural Network that solves a classification problem. 
2. Hyperparameters may be the number of hidden units, number of hidden layers, 
3. etc in that network. Relation between these can be thought as a hypothetical 
4. function that takes the hyperparameters as input and returns classification 
5. accuracy as output. Off course, you don’t know the actual algebraic form of 
6. this function . One way of finding the optimal combination will be trying out 
7. various random combinations by training the network repeatedly. The root of 
8. the problem lies there. We cannot afford that repeated training as it is a 
9. very large & complex network and training takes a lot of time & resources. 
10. Bayesian optimization can help here.