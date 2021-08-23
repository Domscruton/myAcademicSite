---
title: "GLMs: The Link Function and Derivation of Iteratively Reweighted Least Squares"
summary: ""
authors: []
tags:
- Optimization
- R
- GLMs
- Mathematical Statistics

categories: []
date: 2021-09-01T17:28:03+01:00
---

*Intuition behind the Link function, discussion of the various model
fitting techniques and their advantages & disadvantages, derivation of
IRLS using Newton-Raphson and Fisher Scoring. IRLS is then implemented
in R*

GLMs - A Natural Extension of the Linear Model
----------------------------------------------

The first model we naturally learn on any statistics-based course is
Simple Linear Regression (SLR). Despite placing strong (linear)
assumptions on the relationship between the response and covariates, as
well as the error distribution if we are interested in statistical
inference, the Linear model is a surprisingly useful tool for
representing many natural processes. However, when we wish to deal with
non-linear random variable generating processes, such as the probability
of occurrence of an event from a binary or multinomial distribution, or
for the modelling of counts within a given time period, we need to
generalize the Linear Model.

Generalized Linear Models enable us to explicitly model the mean of a
distribution from the exponential family, such that predictions lie in a
feasible range and the relationship between the mean and the variance is
appropriate to perform reliable inference. In this brief blog we discuss
the intuition behind the Generalized Linear Model and Link function and
prove the method of Iteratively ReWeighted Least Squares enables us to
fit a GLM, before implementing it in R.

Generalized Linear Models have 3 components:

**1) Random Component Error Structure**

*y*<sub>*i*</sub> ∼ *e**x**p**o**n**e**n**t**i**a**l* *f**a**m**i**l**y* *d**i**s**t**i**b**u**t**i**o**n*

We also typically assume each *y*<sub>*i*</sub> is independent and
identically distributed, although this assumption can be relaxed through
the use of Generalized Estimating Equations (GEE's).

**2) Systematic Component/ Linear Predictor**

$$\\eta\_i = \\beta\_0 + \\sum\_{i = 1}^{p}\\beta\_p x\_{i, p}$$

**3) Link Function**

𝔼\[*y*<sub>*i*</sub>|*x*<sub>1, *i*</sub>, ..., *x*<sub>*p*, *i*</sub>\]=*μ*<sub>*i*</sub>
*g*(*μ*<sub>*i*</sub>)=*η*<sub>*i*</sub>

It should be made clear that we are not modellig the response
*y*<sub>*i*</sub> explicitly, but rather modelling the mean of the
distibution, *μ*<sub>*i*</sub>. Predictions for each observation *i* are
given by *μ*<sub>*i*</sub>, with each *y*<sub>*i*</sub> assumed to be
centred around *μ*<sub>*i*</sub> in expectation but with an error term
that has a distibution specified by the member of the exponential family
used. Therefore, the link function does not transform the response
*y*<sub>*i*</sub> but instead transforms the mean *μ*<sub>*i*</sub>.
Note that the linear model is a specific type of GLM, where
*y*<sub>*i*</sub> ∼ *N**o**r**m**a**l* and
$g(\\mu\_i) = \\mu\_i = \\eta\_i = \\beta\_0 + \\sum\_{i = 1}^{p}\\beta\_p x\_{i, p}$.
For a Poisson GLM, each *y*<sub>*i*</sub> is a r.v. simulated from the
Poisson distribution with mean *μ*<sub>*i*</sub>, hence
*y*<sub>*i*</sub> has a Poisson error distribution, the difference
between *y*<sub>*i*</sub> and $\\hat{y\_i} = \\mu\_i$.

Link Functions
--------------

So Generalized Linear models are simply a natural extension of the
Linear Model. They differ through the explicit introduction of a link
function (the link function for the linear model is simply the identity,
*g*(*μ*)=*μ*) and through the specification of a mean-variance
relationship (the response belongs to a member of the exponential
family). Using a link function allows us to transform values of the
linear predictor to predictions of the mean, such that these predictions
are always contained within the range of possible values for the mean,
*μ*<sub>*i*</sub>.

When choosing a link function, there is no 'correct' choice, however
there are a few properties we require to be able to interpret and fit a
model:

1.  The link function transforms the linear predictor such that the
    prediction of *μ*<sub>*i*</sub> for each *y*<sub>*i*</sub> is within
    the range of possible values of *μ*.

2.  The link function must be *monotonic* and therefore have a *unique
    inverse*. That is, each value on the linear predictor must be mapped
    to a unique value of the mean and the link function must preserve
    the order/ranking of predictions.

3.  The link function must be *differentiable*, in order to estimate
    model coefficients.

For OLS, the linear predictor *X**β* can take on any value in the range
( − ∞, ∞). For a Poisson model, we require the rate parameter *μ*
(equivalent to the commonly used *λ*), to lie in the range (0, ∞), thus
we need a link function that transforms the linear predictor *η* to lie
in this range. The common choice of link function for a Poisson GLM is
the log-link (log(*μ*<sub>*i*</sub>)=*η*<sub>*i*</sub>). Exponentiating
the linear predictor results in *μ*<sub>*i*</sub> ∈ (0, ∞) as required
for count data modeled via a Poisson. The log-link also results in a
nice interpretation, since it exponentiates the linear predictor
resulting in a multiplication of exponentiated coefficients:
$log(\\mu\_i) = exp(\\b\_0 + \\beta\_1 x\_{1, i}) = \\exp(\\beta\_0) \\exp(\\beta\_1 x\_{1, i})$.
Note that we can't use the square root function as a link function since
it does not have a *unique inverse* (i.e. $\\sqrt(4) = \\pm 2$).

For the Binomial, we choose a link function that maps *p*<sub>*i*</sub>,
the probability of success to the interval \[0, 1\]. The link function
$g(\\mu\_i) = log(\\frac{p\_i}{n - p\_i}) = X \\beta\_i$ is one
candidate. This transforms the Linear predictor:
$\\frac{p\_i}{1 - p\_i} = \\exp(X \\beta\_i)$
$\\implies p\_i = (\\frac{e^{X \\beta\_i}}{1 + e ^ {X \\beta\_i}}) \\in \[0, 1\]$
to the required range.

Whilst we are able to choose any link function that satisfies these
properties, the usual choice is to select the *Canonical* link function,
which arises from writing the distribution in its exponential form.
These link functions have nice mathematical properties and simplify the
derivation of the Maximum Likelihood Estimators. We could also use an
Information Criteria such as AIC to choose the best-fitting link
function, although there is typically little deviation in performance,
so we typically choose the link function with the most intuitive
interpretation (which is often the canonical link function anyway).

Some common distributions and Canonical links are show below:

<table>
<colgroup>
<col width="10%" />
<col width="42%" />
<col width="46%" />
</colgroup>
<thead>
<tr class="header">
<th>Family</th>
<th>Canonical Link (<span class="math inline"><em>η</em> = <em>g</em>(<em>μ</em>)</span>)</th>
<th>Inverse Link (<span class="math inline"><em>μ</em> = <em>g</em><sup>−1</sup>(<em>η</em>)</span>)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Binomial</td>
<td>Logit: <span class="math inline">$\eta = log \left( \frac{\mu}{n - \mu} \right)$</span></td>
<td><span class="math inline">$\mu = \frac{n}{1 + e^{-n}}$</span></td>
</tr>
<tr class="even">
<td>Gaussian</td>
<td>Identity: <span class="math inline"><em>η</em> = <em>μ</em></span></td>
<td><span class="math inline"><em>μ</em> = <em>η</em></span></td>
</tr>
<tr class="odd">
<td>Poisson</td>
<td>Log: <span class="math inline"><em>η</em> = <em>l</em><em>o</em><em>g</em>(<em>μ</em>)</span></td>
<td><span class="math inline"><em>μ</em> = <em>e</em><sup><em>η</em></sup></span></td>
</tr>
<tr class="even">
<td>Gamma</td>
<td>Inverse: <span class="math inline">$\eta = \frac{1}{\mu}$</span></td>
<td><span class="math inline">$\mu = \frac{1}{\eta}$</span></td>
</tr>
</tbody>
</table>

The Exponential Family of Distributions
---------------------------------------

A random variable y has a distribution within the exponential family if
its probability density function is of the form:

$$f(y;\\theta, \\phi) = exp \\left( \\frac{y \\theta - b(\\theta)}{a(\\phi)} + c(y, \\phi) \\right)$$

where:

-   *θ* is the location parameter of the distribution

-   *a*(*ϕ*) is the scale/dispersion parameter

-   *c*(*y*, *ϕ*) is a normalization term

It can also be shown that 𝔼\[*y*\]=*b*′(*θ*) and
*V**a**r*\[*y*\]=*b*″(*θ*)*a*(*ϕ*).

Likelihood Analysis and Newton-Raphson Method
---------------------------------------------

The fitting of any form of Statistical Learning algorithm involves an
optimization problem. Optimization refers to the task of either
minimizing or maximizing some function *f*(**β**) by altering *β*. We
usually phrase most optimization problems in terms of minimizing
*f*(*β*). For the case of parametric learning models (we place
distributional assumptions on the target/response *y*<sub>*i*</sub>,
such that they are drawn independently from some probability
distribution *p*<sub>*m**o**d**e**l*</sub>(*y*, *θ*)), we we can also
use the process of Maximum Likelihood to find model coefficients. Under
this approach, we have the following optimization problem:

**β**<sup>\*</sup> = *a**r**g**m**a**x*(*l*(*y*<sub>*i*</sub>|**β**))

where *l*(*y*<sub>*i*</sub>) is the likelihood function. The likelihood
can be interpreted as the probability of seeing the data in our sample
given the parameters and we naturally wish to maximize this quantity to
obtain a good model fit.

<strong>Why Maximum Likelihood?</strong>

There are several reasons why the process of Maximum Likelihood is
preferential:

1.  Simple and intuitive method to find estimates for any *parametric*
    model.

2.  For large samples, MLE's have useful properties, assuming large n
    and i.i.d (independent and identically distributed) samples

    1.  $\\mathop{\\mathbb{E}}\[\\hat{\\theta}\_{MLE}\] = \\theta$
        (**Unbiased**)

    2.  $Var(\\hat{\\theta}\_{MLE}) = \\frac{1}{n I(\\theta)}$, where
        *I*(*θ*) is the Fisher Information in the sample. That is, we
        can calculate the variance of model coefficients and hence
        perform inference

    3.  The MLE also achieves the **Cramer Lower Bound**, that is it has
        the smallest variance of any estimator, thus is **Asymptotically
        Efficient**.

    However, there are several disadvantages to Maximum Likelihood,
    particularly if the purpose of modelling is for prediction. Under
    Maximum Likelihood, we fit exactly to the training dataset,
    resulting in overfitting and poor generalization to unseen data.

Consider the general form of the probability density function for a
member of the exponential family of distributions:

$$f(y;\\theta, \\phi) = exp \\left( \\frac{y \\theta - b(\\theta)}{a(\\phi)} + c(y, \\phi) \\right)$$

The likelihood is then (assuming independence of observations):

$$L(f(y\_i)) = \\prod\_{i = 1}^n exp \\left( \\frac{1}{a(\\phi)} (y\_i \\theta\_i - b(\\theta\_i) + c(y\_i, \\phi) \\right)$$

with log-likelihood (since the log of the product of exponentials is the
sum of the exponentiated terms):

$$log(L(f(y\_i))) = l(f(y\_i)) = \\sum\_{i = 1}^n \\frac{1}{a(\\phi)}(y\_i \\theta\_i - b(\\theta\_i)) + c(y\_i, \\phi)$$

Since the logarithmic function is monotonically increasing, order is
preserved hence finding the maximum of the log-likelihood yields the
same result as finding the maximum of the likelihood. We wish to
maximize this log-likelihood, hence we can differentiate, equate to zero
and solve for *β*<sub>*j*</sub> (We could also ensure the second
derivative evaluated at *β*<sub>*j*</sub> is negative, therefore we have
maximized (and not minimized) the log-likelihood). Via the Chain Rule we
have:

**Equation 1**

$$\\frac{\\partial l(f(y\_i))}{\\partial \\beta\_j} = \\sum\_{i = 1}^n \\frac{\\partial l(f(y\_i))}{\\partial \\theta\_i} \\frac{\\partial \\theta\_i}{\\partial \\mu\_i} \\frac{\\partial \\mu\_i}{\\partial \\eta\_i} \\frac{\\partial \\eta\_i}{\\partial \\beta\_j} = 0$$

This is also known as the Score function, since it tells us how
sensitive the model is to changes in *β* at a given value of *β*. Since
we differentiate with respect to each coefficent, *β*<sub>*j*</sub>
(*j* ∈ \[0, 1, ..., *K*\]), we have a system of (*K* + 1) equations to
solve.

<strong>Chain Rule</strong>

The Chain Rule is often used in Optimization problems. Here we can
utilize the chain rule by recognizing that, as seen in **3.1**, the
likelihood is a function of the location parameter of the distribution,
*θ*, which in turn is a function of the mean *μ*. Via the link function,
we have that *g*(*μ*<sub>*i*</sub>)=*η*<sub>*i*</sub> and via the linear
predictor,
$\\eta\_i = \\beta\_0 + \\sum\_{i = 1}^{p}\\beta\_p x\_{i, p}$.

Each of these individual partial derivatives can then be identified:

$$\\frac{\\partial l\_i}{\\partial \\theta\_i} = \\frac{y\_i - b'(\\theta\_i)}{a(\\phi)}$$

$$\\frac{\\partial \\theta\_i}{\\partial \\mu\_i} = \\frac{1}{V(\\mu\_i)}$$

since $\\frac{\\mu\_i}{\\theta\_i} = b''(\\theta\_i) = V(\\mu\_i)$ where
V is the variance function of the model as it dictates the mean-variance
relationship.

$$\\frac{\\partial \\mu\_i}{\\partial \\eta\_i} = \\frac{1}{g'(\\mu\_i)}$$

since
$g(\\mu\_i) = \\eta\_i \\implies \\frac{\\partial \\eta\_i}{\\partial \\mu\_i} = g'(\\mu\_i)$.
Finally:

$$\\frac{\\partial \\eta\_i}{\\partial \\beta\_j} = x\_j$$

Putting this all together yields the Maximum Likelihood Equations:

$$\\sum\_{i = 1}^n \\frac{(y\_i - \\mu\_i) x\_{i, j}}{a(\\phi) V(\\mu\_i) g'(\\mu\_i)} = 0$$

<strong>Least Square Vs Maximum Likelihood</strong> If we assume a
normal distribution, there are two methods to solve exactly for the
coefficients (we could also use Gradient Descent however this does not
typically yield an exact solution).

However, this does not have a closed form solution (what does it mean
for a linear system of equations to have a closed form solultion)
(except for the normal distribution). As a sanity check to confirm our
conclusions thus far, note that for the normal distribution we have the
Normal Equations which have the standard closed form solution (what is
it)?. The Maximum Likelihood equations are a set of equations for the
regression coefficients **β** = (*β*<sub>0</sub>, ..., *β*<sub>1</sub>)

Effectively, we need to find coefficients *β*<sub>*j*</sub> (which for
each observation *y*<sub>*i*</sub> affect our prediction of
*μ*<sub>*i*</sub>, *g*′(*μ*<sub>*i*</sub>) and *V*(*μ*<sub>*i*</sub>)
via our distributional assumptions for the relationship between the mean
and the variance), such that summing these terms over all observations
gives 0. To solve this, we use the Newton - Raphson Method

Iteratively Reweighted Least Squares (IRLS)
-------------------------------------------

Recall the Newton - Raphson method for a single dimension. We wish to
find the root of the function (in this case the value of *β* such that
the derivative of the log-likelihood is 0). In One-Dimension, to find
the root of function f we have:

$$x\_{t+1} = x\_t - \\frac{f(x\_t)}{f'(x\_t)}$$

The closer *f*(*x*<sub>*t*</sub>) is to 0, the closer we are to the
root, hence the step change between iterations will be smaller.

<center>
<img src="NewtonRaphson.png" alt="Newton Raphson Method in One Dimension", width = "40%">
</center>
Instead of the log-likelihood function *f*(*x*), we want to optimize its
derivative. Therefore we have the following Newton-Raphson equation in
One-Dimension:

$$x\_{t+1} = x\_t - \\frac{f'(x\_t)}{f''(x\_t)}$$

or for the vector of coefficient estimates at iteration t:

$$\\beta\_{t+1} = \\beta\_t - \\left( \\frac{\\partial l}{\\partial \\beta\_j}(\\beta\_t) \\right) \\left( \\frac{\\partial^2 l}{\\partial \\beta\_j \\partial \\beta\_k} \\right)^{-1}$$

The Newton-Raphson technique is derived by considering the *Taylor
Expansion* about the solution *β*<sup>\*</sup> (that sets
$\\frac{\\partial l}{\\partial \\beta}$ to zero).

$$0 = \\frac{\\partial l}{\\partial \\beta}(\\beta^\*) - (\\beta - \\beta^\*) \\frac{\\partial^2 l}{\\partial \\beta\_j \\beta\_k} + ...$$

If we ignore all derivative terms higher than 2<sup>*n**d*</sup> order,
we can derive and iterative solution. Under the Newton-Raphson approach,
the function being minimized is approximated locally by a quadratic
function, and this approximated function is minimized exactly. We then
have:

*β*<sub>*t* + 1</sub> = *β*<sub>*t*</sub> − **H**<sub>*t*</sub><sup>−1</sup>**U**<sub>*t*</sub>

where
**U**<sub>*t*</sub>*i**s**t**h**e**S**c**o**r**e**V**e**c**t**o**r*
evaluated at *β*<sub>*t*</sub>. **H**<sub>*t*</sub> denotes the (p + 1)
x (p + 1) Hessian matrix of second Derivatives

Given that
$\\frac{\\partial l}{\\beta\_j} = \\nabla\_{\\beta} l = \\frac{(y\_i - \\mu\_i)}{a(\\phi)} \\frac{x\_{i,j}}{V(\\mu\_i)}\\frac{1}{g'(\\mu\_i)}$,
the Hessian is then:

**Equation 4**

$$\\nabla^{2}\_{\\beta} = \\sum\_{i = 1}^n \\frac{x\_{i,j}}{a(\\phi)} \\left( (y\_i - \\mu\_i)' \\frac{1}{g'(\\mu\_i)} \\frac{1}{V(\\mu\_i)} + (y\_i - \\mu\_i) \\left( \\frac{1}{g'(\\mu\_i)} \\frac{1}{V(\\mu\_i)} \\right)' \\right)$$

by Product Rule. *y*<sub>*i*</sub> is a data point so does not depend on
*β*.

$$\\frac{\\partial \\mu\_i}{\\partial \\beta\_k} = \\frac{\\mu\_i}{\\eta\_i} \\frac{\\eta\_i}{\\beta\_k} = \\frac{1}{g'(\\mu\_i)} x\_k$$

hence the first term becomes:

$$\\frac{x\_{i,j}}{a(\\phi)} \\left( - \\frac{x\_{i, k}}{g'(\\mu\_i)} \\right) \\frac{1}{g'(\\mu\_i)} \\frac{1}{V(\\mu\_i)} = - \\frac{x\_{i, j} x\_{i, k}}{a(\\phi)(g'(\\mu\_i))^2} \\frac{1}{V(\\mu\_i)}$$

Of course, if we differentiate by the same *β*-coefficient, we have
*x*<sub>*j*</sub><sup>2</sup>, which are values on the diagonal of the
Hessian matrix, which recall is
$\\begin{bmatrix} \\left( \\frac{\\partial ^2 l}{\\partial \\beta\_j^2} \\right) & \\left( \\frac{\\partial ^2 l}{\\partial \\beta\_k \\partial \\beta\_j} \\right)\\\\ \\left( \\frac{\\partial ^2 l}{\\partial \\beta\_j \\partial \\beta\_k} \\right) & \\left( \\frac{\\partial ^2 l}{\\partial \\beta\_k^2} \\right)\\\\ \\end{bmatrix}$
in 2-Dimensions.

Now consider the 2<sup>*n**d*</sup> term in **equation 4**:

$$\\left( \\frac{1}{g'(\\mu\_i)} \\frac{1}{V(\\mu\_i)} \\right)'$$

If we used Newton-Raphson, we would need to calculate this derivative.
However, if we use **Fisher Scoring**, this term cancels out and we
don't need to calculate the derivative. Fisher Scoring is a form of
Newton's Method used in statistics to solve Maximum Likelihood equations
numerically. Instead of usig the inverse of the Hessian, we use the
inverse of the Fisher Information matrix:

<strong>Fisher Scoring</strong>
*β*<sub>*t* + 1</sub> = *β*<sub>*t*</sub> = *J*<sup>−1</sup>∇*l*
 where *J* = 𝔼\[−∇<sup>2</sup>*l*\], the expected value of the negative
Hessian.

Taking the negative expected value from **equation 4**, the first term
becomes

$$\\mathop{\\mathbb{E}} \\left( - - \\frac{x\_{i,j} x\_{i,k}}{a(\\phi) (g'(\\mu\_i))^2} \\frac{1}{V(\\mu\_i)} = \\frac{x\_{i,j}x\_{i,k}}{a(\\phi) (g'(\\mu\_i))^2} \\frac{1}{V(\\mu\_i)}$$

since none of the above values depende on *y* hence are all constant.
The 2<sup>*n**d*</sup> term in **equation 4** becomes:

$$\\mathop{\\mathbb{E}} \\left( - \\frac{x\_{i,j}}{a(\\phi)}(y\_i - \\mu\_i) \\left( \\frac{1}{g'(\\mu\_i) V(\\mu\_i)} \\right) ' \\right) =  - \\frac{x\_{i,j}}{a(\\phi)}(y\_i - \\mu\_i) \\left( \\frac{1}{g'(\\mu\_i) V(\\mu\_i)} \\right)' \\mathop{\\mathbb{E}}(y\_i - \\mu\_i)$$

but this expectation is equal to zero and the second term therefore
vanishes. We then have:

$$J = \\sum\_{i = 1}^n \\frac{x\_{i,j} x\_{i,k}}{a(\\phi) (g'(\\mu\_i))^2 \\frac{1}{V(\\mu\_i)}} = \\mathbf{X}^T \\mathbf{W} \\mathbf{X}$$

This can be rewritten as:

*J* = **X**<sup>*T*</sup>**W****X**

where

$$\\mathbf{W} = \\frac{1}{a(\\phi)} \\begin{bmatrix} \\frac{1}{V(\\mu\_1)} \\frac{1}{(g'(\\mu\_1))^2} &  & \\\\ & ... & \\\\ & & \\frac{1}{V(\\mu\_n)(g'(\\mu\_n))^2}\\\\ \\end{bmatrix}$$

From Fisher Scoring, we have:

**β**<sub>*t* + 1</sub> = **β**<sub>**t**</sub> = **J**<sup>−1</sup>∇<sub>*β*</sub>*l*(*β*<sub>*t*</sub>)

We can rewrite
$\\nabla\_{\\beta}l = \\sum\_{i = 1}^n \\frac{y\_i - \\mu\_i}{a(\\phi)} frac{1}{a(\\phi)} \\frac{x\_{i,j}}{V(\\mu\_i g'(\\mu\_i))}$
as **X**<sup>*T*</sup>**D****V**<sup>−1</sup>(*y* − *μ*) where:

$$\\mathbf{D} = \\begin{bmatrix} \\frac{1}{g'(\\mu\_1)} & & \\\\ & ... & \\\\ & & \\frac{1}{g'(\\mu\_n)} \\\\ \\end{bmatrix}$$

$$\\mathbf{V}^{-1} = \\frac{1}{a(\\phi)} \\begin{bmatrix} \\frac{1}{V(\\mu\_1)} & & \\\\ & ... & \\\\ & & \\frac{1}{V(\\mu\_n)} \\\\ \\end{bmatrix}$$

Then for the Fisher Equation we have:

**β**<sub>*t* + 1</sub> = **β**<sub>*t*</sub> + (**X**<sup>*T*</sup>**W****X**)<sup>−1</sup>**X**<sup>*T*</sup>**D****V**<sup>−1</sup>(**y** − **μ**)

We can write this more generally, by noting that **W** is the same as
**D****V**<sup>−1</sup>, except we have $\\frac{1}{(g'(\\mu\_i))^2}$.

⟹**D****V**<sup>−1</sup> = **W****M**

where

$$\\mathbf{M} = \\begin{bmatrix}  \\frac{1}{g'(\\mu\_1)} & & \\\\ & ... & \\\\ & & \\frac{1}{g'(\\mu\_1)} \\\\ \\end{bmatrix}$$

⟹*β*<sub>*t* + 1</sub> = *β*<sub>*t*</sub> + (*X*<sup>*T*</sup>*W**X*)<sup>−1</sup>*X*<sup>*T*</sup>*W**M*(*y* − *μ*)

We can already calculate each of these terms and thus generate an
iterative model-fitting algorithm. We can update the *β*'s until
convergence of the algorithm. We can then simplify the equation:

*β*<sub>*t* + 1</sub> = (*X*<sup>*T*</sup>*W**X*)<sup>−1</sup>(*X*<sup>*T*</sup>*W**X*)*b**e**t**a*<sub>*t*</sub> + (*X*<sup>*T*</sup>*W**X*)<sup>−1</sup>*X*<sup>*T*</sup>*W**M*(*y* − *μ*)=(*X*<sup>*T*</sup>*W**X*)<sup>−1</sup>*X*<sup>*T*</sup>*W*(*X**β*<sub>*t*</sub> + *M*(*y* − *μ*)) = (*X*<sup>*T*</sup>*W**X*)<sup>−1</sup>*X*<sup>*T*</sup>*W**Z*<sub>*t*</sub>

where *Z*<sub>*t*</sub> := *η*<sub>*t*</sub> + *M*(*y* − *μ*)

This is "iteratively Reweighted Least Squares"- at each iteration we are
solving a weighted least squares problem, is iterative because we update
W's and Z's at the same time. The amount the algorithm updates depends
on two things- *Z*<sub>*t*</sub> and *W*. For *Z*<sub>*t*</sub>, larger
deviation between *y* and *μ* results in larger steps in the iteration
procedure. Unless we have a saturated model, $y \\not \\mu$ (observed
values don't equal predicted values), there is a trade-off as we vary
*β*, resulting in different discrepencies between *y*<sub>*i*</sub> and
*μ*<sub>*i*</sub>.

The Hessian is a matrix of second derivatives of the log-likelihood
function, that is, derivatives of the derivative of the log-likelihood
function. The second derivative therefore measures curvature, that is
how much the first derivative changes as we vary the input. If the
likelihood has a second derivative of zero, then the likelihood (cost
function) is a flat line, so its value can be predicted using only the
gradient. If the gradient is 1, a step size of *ϵ* &gt; 0 then the
likelihood function will increase by the value of *ϵ*. If the
2<sup>*n**d*</sup> derivative is negative (the gradient of the
likelihood function is becoming more negative fo an increase *ϵ* at
*x*), the likelihood function curves downwards, so the likelihood will
decrease by more than *ϵ*. That is, the likelihood decreases faster than
the gradient predicts for small *ϵ*.

When our function has multiple input dimensions, there are many
2<sup>*n**d*</sup> derivatives (one for each feature and then for each
feature crossed with every other feature), and can be collected in a
matrix called the *Hessian*. If we look at the IRLS equation, we have 3
terms. Firstly, we have the original value of the function, the expected
improvement due to the slope of the function and a correction we must
apply to account for the curvature of the function. When this last term
is too small, the likelihood step can actually move downhill.

<strong>Derivation of Model Fitting Algorithms/Coefficients</strong>
This derivation of Iteratively Reweighted Least Squares for GLMs follows
a similar procedure to the derivation of any model fitting algorithm.
Firstly, we identify an objective function over which to optimize.
Typical Machine Learning problems involve minimizing some loss function,
which gives discrepencies between the actual and true values. We then
differentiate this function to find a minimum and use Newton - Raphson
... What other algorithms are there for fitting other models and how are
their model-fitting algorithms derived?

<strong>Why Bother with Parametric Assumptions</strong>

**Advantages** - Large amounts of data can be modelled as random
variables from the exponential family of distributions - If there are
relatively few observations, providing a structure for the model
generating process can improve predictive performance - Enables us to
carry out inference on model covariates - Simple and intuitive to
understand. In some industries (such as insurance), this has huge
benefits- it is transparent and can fit into a rate structure

**Disadvantages** - Validity of inference dependent on assumptions being
satisfied - Places a very rigid structure - Typically has worse
predictive performance than non-linear models, such as Boosted Trees and
Neural Networks.

IRLS Implementation
-------------------

In the following, we implement IRLS for Generalized Linear Models
(GLMs). We could also write in C++ for a more efficient implementation.
