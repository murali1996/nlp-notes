# Contents

- [Courses](#Courses)
- [Universal Approximators](#Universal-Approximators)
- [Initialization](#Initialization)
- [Normalizations](#Normalizations)
    - [Batch Normalization](#Batch-Normalization)
- [Regularizations & Overfitting](#Regularizations)
- [Losses](#Losses)
- [Optimization](#Optimization)
- [Training at Large](#Training-at-Large)
- [RNN-LSTM](#RNN-LSTM)
- [Perceptron](#Perceptron)

# Courses

1. [MILA | IFT 6085: Theoretical principles for deep learning](http://mitliagkas.github.io/ift6085-dl-theory-class-2020/)

# Universal Approximators

1. [CHAPTER 4A visual proof that neural nets can compute any function](http://neuralnetworksanddeeplearning.com/chap4.html)
1. [Is-a-single-layered-ReLu-network-still-a-universal-approximator](https://www.quora.com/Is-a-single-layered-ReLu-network-still-a-universal-approximator)

# Initialization

1. [He et al. 2015, Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
1. [Xavier Glorot & Bengio 2010, Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
1. [Daniel Godoy, Hyper-parameters in Action! Part II — Weight Initializers](https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404)

#### [Go Back To Top](#Contents)

# Normalizations

1. [MLExplained blog-post](https://mlexplained.com/2018/01/13/weight-normalization-and-layer-normalization-explained-normalization-in-deep-learning-part-2/)

#### Batch Normalization

1. [Ioffe et al. 2015, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

#### [Go Back To Top](#Contents)

# Regularizations

1. Data Augmentation, Early Stopping, Dropout, DropConnect, L1/L2/MaxNorm
1. [CS231n Notes](http://cs231n.github.io/neural-networks-2/)
1. [Gal et al. 2017, Concrete Dropout](https://arxiv.org/abs/1705.07832)

###

1. [Are Deep Neural Networks Dramatically Overfitted?](https://lilianweng.github.io/lil-log/2019/03/14/are-deep-neural-networks-dramatically-overfitted.html)

### Some notes

- Statisticians look at L2 and L1 regularization as Gaussian and Laplace prior distributions for the
  objective of maximizing MLE.
- Algebraically, it is the L2 and L1 norm terms in Linear Regression are obtained by constraining
  them to a certain value, say C, and applying Lagrangian for this constraint along with OLS (
  Ordinary Least Squares). Solving for Closed-form Solution then yields a lambda term in the
  numerator for L1 and in the denominator for L2, explaining the increase in lambda resulting in
  variable selection in case of LASSO and lowering the magnitude of params in case of RIDGE.
- Geometrically, LASSO is a diamond-shaped contour while RIDGE is a circle. It is more likely that
  the OLS contour hits the LASSO constrained region at any sharp point on an axis and thus resulting
  in some params going to zeros. Whereas in RIDGE, it hits at a tangent point no necessarily
  on-axis, thereby no variable selection.

#### [Go Back To Top](#Contents)

# Losses

1. [Raul Gomez's blog-post, Understanding Categorical Cross-Entropy Loss, Binary Cross-Entropy Loss, Softmax Loss, Logistic Loss, Focal Loss and all those confusing names](https://gombru.github.io/2018/05/23/cross_entropy_loss/)
1.
    - [Learning to Rank](https://en.wikipedia.org/wiki/Learning_to_rank)
    - [TF-Ranking: A Scalable TensorFlow Library for Learning-to-Rank](https://ai.googleblog.com/2018/12/tf-ranking-scalable-tensorflow-library.html)
    - [Raul Gomez's blog-post, Understanding Ranking Loss, Contrastive Loss, Margin Loss, Triplet Loss, Hinge Loss and all those confusing names](https://gombru.github.io/2019/04/03/ranking_loss/)

###### Triplet Loss & Negative Sampling

1. [Olivier Moindrot's blog-post, Triplet Loss and Online Triplet Mining in TensorFlow](https://omoindrot.github.io/triplet-loss)
1. <https://omoindrot.github.io/triplet-loss>
1. [Le et al. 2019, Improving the Robustness of Deep Neural Networks via Adversarial Training with Triplet Loss](https://arxiv.org/abs/1905.11713)
1. [Zhang et al. 2019, Learning Incremental Triplet Margin for Person Re-identification](https://arxiv.org/abs/1812.06576)
1. [Wu et al. 2017, Sampling Matters in Deep Embedding Learning](https://arxiv.org/abs/1706.07567)
1. [Kotnis et al. 2017, Analysis of the Impact of Negative Sampling on Link Prediction in Knowledge Graphs](https://arxiv.org/abs/1708.06816)

#### [Go Back To Top](#Contents)

# Optimization

1. Gradient Clipping to mitigate exploding gradient
1. Vanishing gradient
   problem; [see Deep Learning Book Ch 8.2.5](https://www.deeplearningbook.org/contents/optimization.html#pff)

###

1. [Ruder's blog-post, Optimization for Deep Learning Highlights in 2017](https://ruder.io/deep-learning-optimization-2017/)
1. [Ruder's blog-post, An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)
1. [Berkeley AI, How to Escape Saddle Points Efficiently](https://bair.berkeley.edu/blog/2017/08/31/saddle-efficiency/)

###### Learning Rate

1. [Smith et al. 2015, Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
1. [Smith et al. 2017, Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)
1. [Liu et al. 2019, On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265)

###### Curriculum Learning

1. [Bengio et al. 2009, Curriculum Learning](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf)

###### Batch Size

1. [Keskar et al. 2017, On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836)
1. [Smith et al. 2017, Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489)
1. [Devarakonda et al. 2017, AdaBatch: Adaptive Batch Sizes for Training Deep Neural Networks](https://arxiv.org/abs/1712.02029)

###### Gradient Noising

1. [neelakantan et al. 2019, Adding Gradient Noise Improves Learning for Very Deep Networks](https://arxiv.org/abs/1511.06807)

###### Gradient Checking

1. <http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/>
1. <https://www.coursera.org/lecture/machine-learning/gradient-checking-Y3s6r>

#### [Go Back To Top](#Contents)

# Training at Large

1. [Chen et al. 2015, Strategies for Training Large Vocabulary Neural Language Models](https://arxiv.org/abs/1512.04906)

# RNN-LSTM

1. [The unreasonable effectiveness of the forget gate, Westhuizen and Lasenby et al. 2018](https://arxiv.org/abs/1804.04849)
1. [An Empirical Exploration of Recurrent Network Architectures, Rafal et al. 2015](http://proceedings.mlr.press/v37/jozefowicz15.pdf)
1. <https://www.quora.com/Why-doesnt-the-use-of-a-forget-gate-in-LSTMs-cause-vanishing-dying-gradients>
1. [LSTM: A Search Space Odyssey, Greff et al. 2015](https://arxiv.org/abs/1503.04069)

# Perceptron

- In the case of linearly separable data, the end classifier [wi] is not the optimal classifier.
  There can be many classifiers that give the same zero error but the order in which the data was
  fed dictates the final classifier w’s. Thus INPUT-DATA ORDER MATTERS.
- Why |wx+b-y| had no solution as opposed to OLS? because it is not strictly convex; can expect
  multiple optima

# Random

1. <https://www.kaggle.com/chandraroy/plotting-with-pandas-matplotlib-and-seaborn>
1. <https://medium.com/activewizards-machine-learning-company/top-15-python-libraries-for-data-science-in-in-2017-ab61b4f9b4a7>




