# Machine Learning by Andrew NG 课程笔记

## Lecture 1 Introduction 

### What is machine learning

#### Machine Learning definition

1. Arthur Samuel(1959): Field of study that gives computers the ability to learn without being explicitly programmed.
2. Tom Mitchell(1998): A computer program is said to learn from experience E with respect to some task T and some performance on T, as measured by P, improves with experience E.

#### Machine learning algorithms

1. Supervised learning
2. Unsupervised learning
3. Others: Reinforcement learning, recommender systems

### Introduction Supervised Learning

1. Supervised Learning "right answers" given
2. Regression: Predict continuous valued output
3. Classification: Discrete valued output

### Introduction Unsupervised Learning

Unsupervised Learning **no** "right answers" given

![屏幕快照 2018-08-03 11.02.52](/Users/lixinge/Desktop/屏幕快照 2018-08-03 11.02.52.png)

## Lecture 2 Linear regression with one variable 

### Model representation

1. Supervised Learning 
   1. Given "right answers" for each example in the data
2. Regression Problem
   1. Predict real-valued output
3. Notation:
   1. m = Number of training examples
   2. x's = "input" variable/features
   3. y's = "output" variable / "target" variable
4. model
   1. Training Set -> Learning Algorithm -> hypothesis
   2. Hypothesis: $h_\theta(x) = \theta_0 + \theta_1x$
   3. h maps from x's to y's

### Cost function

1. Hypothesis: $h_\theta(x) = \theta_0 + \theta_1x$
2. Parameters:  $\theta_0, \theta_1$
3. how to choose parameters $\theta_i's$
   1. Idea: Choose $\theta_0, \theta_1$ so that $h_\theta(x)$ is close to y for our training examples(x, y)
4. Cost Function: $J(\theta_0, \theta_1) = \frac{1}{2m}\sum_{i=1}^m (h_\theta(x^i) - y^i)^2$ (squared  error function)
5. Goal: minimize $J(\theta_0, \theta_1)$
   1. fixed $\theta$ -> function of x
   2. compute J
   3. plot the function of $J(\theta)$

![](https://ws4.sinaimg.cn/large/0069RVTdly1ftv6zyby92j31640ns48v.jpg)

![](https://ws2.sinaimg.cn/large/0069RVTdly1ftv6zrl9lqj31640ls7dd.jpg)

![](https://ws2.sinaimg.cn/large/0069RVTdly1ftv70uejw0j31680lgtgw.jpg)

### Gradient Descent

* Have some function $J(\theta_0, \theta_1)$
* Want min $J(\theta_0, \theta_1)$
* Outline
  * Start with some $\theta_0, \theta_1$
  * Keep changing $\theta_0, \theta_1$ to reduce $J(\theta_0, \theta_1)$ until we hopefully end up at a minimum

![](https://ws1.sinaimg.cn/large/0069RVTdly1ftv73u4s44j31580kc7iw.jpg)

![](https://ws4.sinaimg.cn/large/0069RVTdly1ftv73ofzghj314w0k6aof.jpg)

#### Gradient descent algorithm

* repeat until convergence
  * $\theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j} J(\theta_0, \theta_1)$ for (j=0 and j=1)  for (j=0 and j=1) 
* Correct: Simultaneous update
  - $temp0 := \theta_0 - \alpha \frac{\partial}{\partial\theta_0} J(\theta_0, \theta_1)$ 
  - $temp1 := \theta_1 - \alpha \frac{\partial}{\partial\theta_1} J(\theta_0, \theta_1)$ 
  - $\theta_0:= temp0$ 
  - $\theta_1:= temp1$ 
* Incorrect
  * $temp0 := \theta_0 - \alpha \frac{\partial}{\partial\theta_0} J(\theta_0, \theta_1)$
  * $\theta_0:= temp0$
  * $temp1 := \theta_1 - \alpha \frac{\partial}{\partial\theta_1} J(\theta_0, \theta_1)$  
  * $\theta_1:= temp1$ 

![](https://ws2.sinaimg.cn/large/0069RVTdly1ftv79sjvhpj31680nqjw7.jpg)

* learning rate $\alpha$

  * If $\alpha$ is too small, gradient descent can be slow

  * If $\alpha$ is too large, gradient descent can overshoot the minimum. It may fail to converge, or even diverge

    ![](https://ws4.sinaimg.cn/large/0069RVTdly1ftv7c5trlfj30he0nemz4.jpg)

* Gradient descent can converge to a local minimum, even with the learning rate $\alpha$ fixed

  * As we approach a local minimum, gradient descent will automatically take smaller steps. So, no need to decrease learning rate over time.

### Gradient descent for linear regression

####Gradient descent algorithm

* repeat until convergence
  * $\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^m(h_\theta(x^i)-y^i)$
  * $\theta_1 := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^m(h_\theta(x^i)-y^i) x^i$
  * update $\theta_0$ and $\theta_1$ simultaneously

#### "Batch"  Gradient Descent

* "Batch": Each step of gradient descent uses all the training examples

## Lecture 3 Linear Algebra review

#### Matrices and vectors

* Dimensions of matrix: number of rows $\times$ number of columns
* $A_(ij)$ = 'i, j entry' in the $i^{th}$ row, $j^{th}$column

#### Additon and scalar multplication

* Matrix Addition
* Scalar Multiplication
* Combination of Operands

#### Matrix­‐vector multplication

![](https://ws1.sinaimg.cn/large/0069RVTdgy1ftv7qxgr7lj30vm102n48.jpg)

#### Matrix­-matrix multplication

![](https://ws3.sinaimg.cn/large/0069RVTdly1ftv7s23wi0j30vm108tgg.jpg)

#### Matrix multplication properties

* Not commutative: $A \times B \neq B \times A$
* Identity Matrix:
  * For any matrix A,
  * A*I = I\*A = A

#### Inverse and transpose

* Matrix inverse 
  * If A is an m x m matrix, and if it has an inverse,
  * $AA^{-1} = A^{-1}A = I$
  * Matrices that don’t have an inverse are “singular” or “degenerate”
* Matrix Transpose
  * Let A be an m x n matrix, and let $B = A^T$ 
  * Then B is an n x m matrix, and
  * $B_{ij} = A_{ji}$

## Lecture 4 Linear Regression with multiple variables

### Multiple features

* Notation
  * n = number of features
  * $x^{(i)}$ = input (features) of $i^{th}$ training example
  * $x_j^{(i)}$ = value of feature j in  $i^{th}$ training example
* Hypothesis
  * $h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + ...  + \theta_nx_n$
  * For convenience of notation, deﬁne $x_0 = 1$

![](https://ws3.sinaimg.cn/large/0069RVTdgy1ftv83uhycmj318q0pgag1.jpg)

### Gradient descent for multiple variables

* Hypothesis: $h_\theta(x) = \theta^Tx = \theta_0 + \theta_1x_1 + \theta_2x_2 + ...  + \theta_nx_n$

- parameters: $\theta_0, \theta_1, ..., \theta_n$
- Cost function:$J(\theta_0, \theta_1, ..., \theta_n) = \frac{1}{2m} \sum_{i=1}^m(h(x^{(i)}-y^{(i)}))^2$
- Gradient descent: 
  - $\theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j}J(\theta_0, \theta_1, ..., \theta_n)$  (simultaneously update for every j = 0, ..., n)
  - $\theta_j := \theta_j - \alpha \frac{1}{m}\sum_{i=1}^m(h(x^{(i)}-y^{(i)}))x_j^{(i)}$
  - $x_0^{i} = 1$

### Gradient descent in practice I: Feature Scaling

#### Feature Scaling

* Idea: Make sure features are on a similar scale

![](https://ws1.sinaimg.cn/large/0069RVTdgy1ftvcy8532zj318y0pkqao.jpg)

#### Mean normalization

* Replace $x_i$ with $x_i - \mu$ to make features have approximately zero mean (Do not apply to $x_0 =1$)
* Mean normalization: $\frac{x_i - \mu}{max - min}$

####Gradient descent in practice II: Learning rate

####Gradient descent

* $\theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j}J(\theta_0, \theta_1, ..., \theta_n)$ 
* "Debugging": How to make sure gradient descent is working correctly
* How to choose learning rate

####Making sure gradient descent is working correctly

![](https://ws3.sinaimg.cn/large/0069RVTdgy1ftvd810ihsj318y0pswkf.jpg)

* For sufficiently small $\alpha$, $J(\theta)$ should decrease on every iteration

* But if $\alpha$ is too small, gradient desent can be slow to converge

* If $\alpha$ is too large, may not decrease on every iteration; may not converge

* To choose $\alpha$, try

  ..., 0.001, ..., 0.01, ..., 0.1,  ..., 1, ....

### Features and polynomial regression

#### Polynomial regression

![](https://ws1.sinaimg.cn/large/0069RVTdly1ftvdamfrnlj318w0poae5.jpg)

### Normal equation

* Normal equation: Method to solve for $\theta$ analytically

#### Intuition

![](https://ws4.sinaimg.cn/large/0069RVTdly1ftvdc8owg6j318y0poq85.jpg)

![](https://ws3.sinaimg.cn/large/0069RVTdly1ftvdccdyt5j318s0pmdl7.jpg)

* Normal equation: $\theta = (X^{T}X)^{-1}X^Ty$
  * $ (X^{T}X)^{-1}$ is inverse of matrix $X^{T}X$
  * Octave/matlab: pinv(X'*X)*X'*y

#### Gradient Descent & Normal Equation

| Gradient Descent                | Normal Equation                                           |
| ------------------------------- | --------------------------------------------------------- |
| Need to choose $\alpha$         | No need to choose $\alpha$                                |
| Needs many iterations           | Don't need to iterate                                     |
| Works well even when n is large | Slow if n is very large: need to compute $ (X^{T}X)^{-1}$ |

#### Normal equation and non-invertiblility 

* Normal equation: $\theta = (X^{T}X)^{-1}X^Ty$
* What if $ (X^{T}X)^{-1}$ is non-invertible?
  * Redundant features(linearly dependent)
  * Too many features(m $\leq$ n)
    * Delete some features, or use regularization
* Octave/matlab: pinv(X'*X)*X'*y

##Lecture 5 Octave Tutorial



##Lecture 6 Logistic Regression

### Classification

* y ∈ {0, 1}
  * 0: "Negative class"
  * 1: "Positive class"

* Threshold classifier output $h_\theta(x)$ at 0.5:
  * $h_\theta(x) \geq 0.5$, predict "y=1"
  * $h_\theta(x) < 0.5$, predict "y=0"
* classification: y = 0 or y = 1
* $h_\theta(x)$ can be > 1 or < 0
* Logistic Regression: 0 $\leq h_\theta(x) \leq$ 1

### Hypothesis Representation

#### Logistic Regression Model

* Want 0 $\leq h_\theta(x) \leq$ 1
* Sigmoid function(Logistic function)
  *  $h_\theta(x) = \frac{1}{1 + e^{-\theta^Tx}}$

#### Interpretation of Hypothesis Output

* $h_\theta(x)$ = estimated probability that y = 1 on input x

### Decision boundary

#### Logistic regression

* $h_\theta(x) = g(\theta^Tx)$      $g(z) = \frac{1}{1 + e^{-z}}$
* Suppose predict "y=1" if  $h_\theta(x) \geq 0.5$
  * $\theta^Tx \geq 0$
* predict "y=0" if $h_\theta(x) < 0.5$
  * $\theta^Tx < 0$

#### Decision Boundary

![](https://ws2.sinaimg.cn/large/0069RVTdgy1ftwb7fdjbqj30w8114dps.jpg)

### Cost function

* How to choose parameters $\theta$ ?
* $Cost(h_\theta(x), y) = $
  * $-log(h_\theta(x))$  if y = 1
  * $-log(1 - h_\theta(x))$  if y = 0

![](https://ws3.sinaimg.cn/large/0069RVTdgy1ftwbdczfqzj30w810y7ba.jpg)

### Simplified cost function and gradient descent

#### Logistic regression cost function

* $J(\theta) = \frac{1}{m}\sum_{i=1}^mCost(h_\theta(x^{(i)}, y^{(i)}))$

* $Cost(h_\theta(x), y) = $

  - $-log(h_\theta(x))$  if y = 1
  - $-log(1 - h_\theta(x))$  if y = 0

* $J(\theta) = \frac{1}{m}\sum_{i=1}^mCost(h_\theta(x^{(i)}, y^{(i)})) $

  $= -\frac{1}{m}[\sum_{i=1}^my^{(i)}logh_\theta(x^{(i)}) + (1 - y^{(i)})log(1 - h_\theta(x^{(i)}))]$

* To fit parameters $\theta$

  * min $J(\theta)$

* To make a prediction given new x

  * Output $h_\theta(x) = \frac{1}{1 + e^{-\theta^Tx}}$

#### Gradient Descent

* $J(\theta) = -\frac{1}{m}[\sum_{i=1}^my^{(i)}logh_\theta(x^{(i)}) + (1 - y^{(i)})log(1 - h_\theta(x^{(i)}))]$
* Want min $J(\theta)$
  * Repeate $\theta_j := \theta_j - \alpha\frac{\partial}{\partial\theta_j}J(\theta)$ (simultaneously update all $\theta_j$)
  * $\theta_j := \theta_j - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$
* Algorithms looks identical to linear regression!

### Advanced optimization

#### Optimization algorithm

* Cost function $J(\theta)$. Want $min_\theta J(\theta)$

* Given $\theta$,  we have code that compute

  * $J(\theta)$
  * $\frac{\partial}{\partial\theta_j}J(\theta)$

* Gradient descent

  * Repeate $\theta_j := \theta_j - \alpha\frac{\partial}{\partial\theta_j}J(\theta)$ (simultaneously update all $\theta_j$)

* Optimization algorithms

  * Gradient descent
  * Conjugate gradient
  * BFGS
  * L-BFGS

* Advantages 

  * No need to manually pick learning rate
  * Often faster than gradient descent

* Disavantages

  * More complex

* Example

  ![](https://ws2.sinaimg.cn/large/0069RVTdgy1ftwbs42sfrj30x611utke.jpg)

### Multi-class classification One-vs-all

#### Multiclass classification

* Example
  * Email foldering: Work, Friends, Family
  * Medical diagrams: Not ill, Cold, Flu
  * Weather: Sunny, Cloudy, Rain

#### One-vs-all(One-vs-rest)

* $h_\theta^{(i)}(x) = P(y = i|x;\theta) (i=1,2,3,...,n)$
* Train a logistic regression classifier $h_\theta^{(i)}(x)$ for each class i to predict the probability that $y = i$
* On a new input x, to make a prediction, pick the class i that maximizes $max_{i}  h_\theta^{(i)}(x)$

## Lecture 7 Regularization

### The problem of overfitting

![](https://ws4.sinaimg.cn/large/0069RVTdgy1ftwc1suinzj312i0eagon.jpg)

![](https://ws4.sinaimg.cn/large/0069RVTdgy1ftwc25orjoj31200jmtfs.jpg)

* Overfitting: If we have too many features, the learned hypothesis may fit the training set very well ($J(\theta) \approx 0$), but fail to generate to new examples

#### Addressing overfitting

* Options
  * Reduce number of features
    * Manually select which features to keep
    * Model selection algorithm
  * Regularization
    * Keep all the features, but reduce magnitude/values of parameters $\theta_j$
    * Works well when we have a lot of features, each of which contributes a bit to predicting y

### Cost function

####Intuition

![](https://ws2.sinaimg.cn/large/0069RVTdgy1ftwc5ajfo9j31440my432.jpg)

#### Regularization

![](https://ws4.sinaimg.cn/large/0069RVTdgy1ftwc63b5jbj30wk11845m.jpg)

* In regularized linear regression, we choose $\theta$ to minimize

  $J(\theta) = \frac{1}{2m} [\sum_{i=1}^m(h(x^{(i)}-y^{(i)}))^2 + \lambda\sum_{j=1}^n\theta_j^2]$

* What if $\lambda$ is set to an extremely large value?

  * Algorithm works fine; setting $\lambda$ to be very large can't hurt it
  * Algorithms fails to eliminate overfitting
  * Algorithm results in underfitting(Fails to fit even training data well)
  * Gradient descent will fail to converge

### Regularized linear regression

####Regularized linear regression

$J(\theta) = \frac{1}{2m} [\sum_{i=1}^m(h(x^{(i)}-y^{(i)}))^2 + \lambda\sum_{j=1}^n\theta_j^2]$

####Gradient descent

* Repeate
  * $\theta_0 := \theta_0 - \alpha \frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}-y^{(i)}))x_0^{(i)}$
  * $\theta_j := \theta_j(1-\alpha\frac{\lambda}{m}) - \alpha \frac{1}{m}\sum_{i=1}^m(h(x^{(i)}-y^{(i)}))x_j^{(i)}$

#### Normal equation

![](https://ws3.sinaimg.cn/large/0069RVTdgy1ftwcnivt10j312a0k241v.jpg)

### Regularized logistic regression

####Regularized logistic regression

$J(\theta) = -\frac{1}{m}[\sum_{i=1}^my^{(i)}logh_\theta(x^{(i)}) + (1 - y^{(i)})log(1 - h_\theta(x^{(i)}))] + \frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2$

####Gradient descent

* Repeat
  * $\theta_0 := \theta_0 - \alpha\frac{\partial}{\partial\theta_j}\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)} - y^{(i)})x_0^{(i)}$
  * $\theta_j := \theta_j - \alpha[\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} +\frac{\lambda}{m}\theta_j]$, (j = 1, 2, 3, ..., n)

#### Advanced optimization

![](https://ws2.sinaimg.cn/large/0069RVTdgy1ftwcv1lnwkj31460n2k00.jpg)

## Lecture 8 Neural Networks: Representation

### Non-linear hypotheses

![](https://ws2.sinaimg.cn/large/0069RVTdgy1ftwcy7zmyrj31460mwwl5.jpg)

### Neurons and the brain

#### Neural Networks

* Origins: Algorithms that try to mimic the brain.
* Was very widely used in 80s and early 90s; popularity diminished in late 90s.
* Recent resurgence: State-of-the-art technique for many applications

![屏幕快照 2018-08-03 11.37.53](/Users/lixinge/Desktop/屏幕快照 2018-08-03 11.37.53.png)

### Model representation I

#### Neuron model: Logistic unit

![](https://ws1.sinaimg.cn/large/0069RVTdgy1ftwd10um6lj31420msgqg.jpg)

#### Neural Network

* $a_i^{(j)} = $ "activation" of unit i in layer j
* $\theta^{(j)} = $ matrix of weights controlling function mapping from layer j to layer j+1
  * $a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3)$
  * $a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3)$
  * $a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3)$
  * $h_{\Theta}(x) = a_1^{(3)} = g(\Theta_{10}^{(1)}a_0 + \Theta_{11}^{(1)}a_1 + \Theta_{12}^{(1)}a_2 + \Theta_{13}^{(1)}a_3)$
* If network has $s_j$ units in layer j, $s_{j+1}$ units in layer j+1, then $\Theta^{(j)}$ will be of dimension $s_{j+1} \times (s_j + 1)$

![](https://ws2.sinaimg.cn/large/0069RVTdgy1ftwd23s99tj30ww11owpk.jpg)

### Model representation II

#### Forward propagation: Vectorized implementation

![](https://ws4.sinaimg.cn/large/0069RVTdgy1ftwdi9q7hrj30zw0kw7cc.jpg)

#### Other network architectures

![](https://ws3.sinaimg.cn/large/0069RVTdgy1ftwdjv7i0jj30xm0gejv8.jpg)

### Examples and intuitions I

#### Non-linear classification example: XOR/XNOR

* y = $x_1$ XNOR $x_2$
  * $x_1 = 1, x_2 = 1$ -> y = 1
  * $x_1 = 0, x_2 = 1$ -> y = 0
  * $x_1 = 0, x_2 = 0$ -> y = 1
  * $x_1 = 1, x_2 = 0$ -> y = 0

![](https://ws2.sinaimg.cn/large/0069RVTdgy1ftwdptyrryj313g0mmn26.jpg)

#### Simple example: AND

$h_\Theta(x) = g(-30 + 20x_1 + 20x_2)$

![](https://ws1.sinaimg.cn/large/0069RVTdgy1ftwdrvx6sdj313a0mgq8n.jpg)

####Example: OR function

$h_\Theta(x) = g(-10 + 20x_1 + 20x_2)$

![](https://ws1.sinaimg.cn/large/0069RVTdgy1ftwducmu4rj313a0mgn0n.jpg)

### Examples and intuitions II

![](https://ws1.sinaimg.cn/large/0069RVTdgy1ftwdvfmdngj30xe1247e5.jpg)

### Multi-class classification

#### Multiple output units: One-vs-all

![](https://ws1.sinaimg.cn/large/0069RVTdgy1ftwdyc6x0qj30y412ydxk.jpg)

