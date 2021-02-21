If github in unable to render a Jupyter notebook, copy the link of the notebook and enter into the nbviewer:
https://nbviewer.jupyter.org/

# Linear Regression - An Extensive Adventure


There are two very different **algorithms or approaches** for implementing the Linear Regression model.

- The “closed-form” solution approach known as the Ordinary Least Squares (OLS) method.
- Iterative optimization approach known as the Gradient Descent (GD).
We will perform an extensive investigation of these two approaches using Scikit-Learn in a series of four notebooks. For this exploration we will use the Boston Housing dataset that has 506 samples and 13 features.


## Index for the Notebook Series on Scikit-Learn Solutions for Linear Regression

There are four notebooks on sklearn Linear Regression.

1. Linear Regression-1-OLS
        
        -- OLS method & Regularized OLS Method (Ridge Rergression)
        
2. Linear Regression-2-OLS Polynomial Regression-Frequentist Approach (MLE)
        
        -- Polynomial regression using the OLS method
        
3. Linear Regression-3-OLS Polynomial Regression-Bayesian Approach (MAP)

        -- Polynomial regression using the regularized OLS method
        
4. Linear Regression-4-Gradient Descent

        -- Iterative optimization approach (Gradient Descent & Stochastic Gradient Descent)
        

5. PLinear Regression-5-Polynomial SGD Regressor Model Selection

        -- Perform model selection when using Stochastic Gradient Descent (SGD) algorithm for a Polynomial Regression Model



## Complexity of the Solution Approaches


#### Closed-form Solution (OLS Method)
The OLS method computes the inverse of $X^TX$, which is a $(d+1) × (d+1)$ matrix (where d is the number of features). 
The computational complexity of inverting such a matrix is typically about $O(d^{2.4})$ to $O(d^3)$ (depending on the implementation). 

- Limitations: 

        -- The OLS method gets very slow when the number of features are large (e.g., 100,000).
    
        -- It requires the entire dataset to be stored in the memory.

- When Should We Use the OLS Method?

This method is linear with regards to the number of instances in the training set (it is $O(n)$).
So, it handles large training sets efficiently, provided they can fit in memory.
Generally, we should use the OLS method when the **dataset is not large (can be stored in memory) and  the number of features is not large**.


#### Gradient Descent Approach

We explore two variants of the GD approach: Batch GD and Stochastic GD (SGD). There is yet another variant known as mini-batch GD.

    Batch Gradient Descent: 
It uses the whole batch of training data at every step. As a result it is terribly slow on very large training sets. 

- When Should We Use the Batch GD?

Gradient Descent scales well with the number of features. Thus, we should use it when the **dataset is not too large and there are hundreds of thousands of features**.



    Stochastic Gradient Descent:
It picks a random instance in the training set at every step and computes the gradients based only on that single instance. It makes the algorithm much faster since it has very little data to manipulate at every iteration. 

- When Should We Use the SGD?

We should use the SGD when we need to train huge training sets. Also, SGD can be implemented as an out-of-core algorithm.


## High-Level Summary (A Pragmatic Guideline)

- Determine the level of complexity of pattern in the dataset.
    
      -- Generate the degree vs. RMSE curve (see notebook 2) and determine the optimal complexity (polynomial degree) of the model
    
- Implement the regularized polynomial regression (using the optimal degree).
    
      -- Use the regularized OLS method if dataset and feature set is not too large (notebook 3)
    
      -- Use the regularized GD approach otherwise (notebook 4)
