## The Papers

**The generalization error of random features regression: Precise asymptotics and double descent curve** [@meiGeneralizationErrorRandom2019]

 This paper derives an analytic expression for the generalization error of a random features (RF) model with regularization. In particular, for data $x_1,\ldots,x_n \overset{iid}{\sim} \text{Unif}(S^{d-1}(\sqrt{d}))$, $y_i = f_d(x_i) + \epsilon_i$, they minimize a regularized loss functional over the function space:

$$
  \mathcal F_{RF} (\theta) = \left\{ f(x, a, \theta) \equiv \sum_i a_i \sigma(\langle x, \theta_i\rangle/\sqrt{d}) \ | \ a_i \in \mathbb R \quad \forall i\in [N]\right\}\,,
$$

where $\theta \in \mathbb R^{N\times d}$, $\theta_i$ is a row vector with $\theta_1,\ldots \theta_N \overset{iid}{\sim} \text{Unif}(S^{d-1}(\sqrt{d}))$, and $\sigma$ is an activation function meeting suitable regularity conditions.

Note that this setting is equivalent to a two-layer neural network in which the first layer is parameterized by $\theta$ and is fixed at its randomly initialized value, while the coefficients $a=(a_i)_{i\leq N}$ parameterize the second layer and are allowed to vary, as in the "conjugate kernel" setting [@danielyDeeperUnderstandingNeural2017].

They learn the coefficients $a$ by performing ridge regression (with MSE loss functional):

$$
\hat{a}(\lambda) = \underset{a}{\text{argmin}} \left\{ \frac{1}{n}\sum_{j=1}^n  \left(y_j - \sum_i a_i \sigma(\langle x_j, \theta_i\rangle/\sqrt{d})\right)^2 + \frac{N\lambda}{d}||a||_2^2 \right\}\,.
$$

The authors note the correspondence between this form and kernel ridge regression, where the corresponding kernel is given by
$$
K(x, x') = \sum_{i=1}^N \sigma(\langle x, \theta_i\rangle/\sqrt{d}) \sigma(\langle x', \theta_i\rangle/\sqrt{d})\,,
$$
with RKHS $\mathcal F_{RF} (\theta)$. Note that this kernel converges to a deterministic function as $N\rightarrow\infty$ and can be regarded as a finite-rank approximation to this limiting kernel as in the random Fourier features literature [@rahimiRandomFeaturesLargeScalea].

The authors also note that the ridge regularization path (as a function of $\lambda$) is "naturally connected" to the path of gradient flow on the MSE term, initialized at $a=0$. In particular, gradient descent converges to the ridgeless limit $lim_{\lambda \rightarrow 0} \hat a(\lambda)$ and $\lambda > 0$ corresponds to early stopping (question: does this apply to arbitrary loss functionals, or is this fact exclusive to MSE?).

The main results of this paper are given in the limit that $N,n, d \rightarrow \infty$ with $N/d\rightarrow\psi_1, \ n/d\rightarrow \psi_2$ for some $\psi_1, \psi_2 \in (0,\infty)$, i.e the "linear limit". The authors analytically found a "double descent" curve as a function of $\psi_1/\psi_2 = N/n$, with a decreasing test error (underparameterized), followed by a cusp at $N/n=1$, finally followed by an asymptotic decrease to a global minimum (overparameterized). In particular, the ridgeless limit has a diverging test error at the cusp which is smoothed out by regularization, qualitatively matching previous findings observations of this "jamming" transition.

\
\
**Double Trouble in Double Descent : Bias and Variance(s) in the Lazy Regime** [@dascoliDoubleTroubleDouble2020]:

This paper claims that it extends the results of [@meiGeneralizationErrorRandom2019] to the "lazy learning" regime of neural networks, but I am rather unconvinced. The lazy regime corresponds to the regime in which the following linear approximation holds throughout gradient descent training of a neural network $f_\theta$:
$$
  f_\theta(x) \approx f_{\theta_0}(x) + \nabla_\theta f_\theta(x)|_{\theta=\theta_0}\cdot (\theta - \theta_0)
$$

The authors argue that this corresponds to learning a linear model with feature vector $\nabla_\theta f_\theta(x)|_{\theta=\theta_0}$ and coefficients $(\theta - \theta_0)$, which suggests that the double descent/jamming phenomenology should hold in this setting as well.

They go on to analyze some interesting and important scaling asymptotics, e.g. analysis of bias and variance due to noise in the initialization, sampling, and measurement processes, as well as the effect of ensembling networks in the over- and under-parameterized regime. However, they do this analysis in the specific random features formulation of [@meiGeneralizationErrorRandom2019], i.e. features of the form $\sigma(\langle x, \theta_i\rangle/\sqrt{d})$ with $\theta_i \overset{iid}{\sim} \text{Unif}(S^{d-1}(\sqrt{d}))\,.$

They avoid the discussion of whether the lazy regime can be cast in this form at all. They do, however, briefly discuss a numerical experiment which shows a network in the lazy regime displaying a sharp looking peak in the test error for a certain value of $N/n$ which is smoothed out when the network is ensembled (agreeing with one of their predictions) and show some slightly different behavior for a network in the active regime.

I don't mean to minimize the importance of the contributions of this paper, but I do want to emphasize some of its gaps.


## Possible Directions

I still need to do some literature review to see if any of these directions have been addressed in any papers I have yet to read, but here are my thoughts at this current stage.

1. **Explicit extension of the main result of [@meiGeneralizationErrorRandom2019] to the lazy learning setting:**

    There may be a route through [@fanSpectraConjugateKernel2020], which derives a recursive expression for the Steiltjes transform--which at first glance is the primary tool used in the main result of [@meiGeneralizationErrorRandom2019]--of the NTK at initialization (the kernel corresponding to the feature vector in the lazy regime) for linear depth networks.

    I forsee some road-blocks here.
     - [@fanSpectraConjugateKernel2020] claims that "in the linear-width regime, the ... NTK [is] expected to evolve over training," suggesting that the lazy learning regime may be incompatible with the linear-width regime. On the other hand, we can try to force lazy learning by using the trick introduced in [@chizatLazyTrainingDifferentiable2020] by re-scaling the learning rate and outputs appropriately. We can alternatively force lazy learning by using the linearized dynamics in equation (7) of [@leeWideNeuralNetworks2019]. The graphs I showed last week were the result of doing exactly this with unregularized squared hinge loss. I saw no cusp when varying $N/n$, which is quite surprising to me after reading the two papers I discussed in the previous question. I am interested in following up on this.
     - [@fanSpectraConjugateKernel2020] claims that the eigenvectors of the feature matrix/the kernel matrix are important in addition to the eigenvalues, and I am not yet sure if [@fanSpectraConjugateKernel2020] can describe those as well. I will need to get way more comfortable with random matrix theory to make this work.

2. **Relate the generalization cusp found in [@meiGeneralizationErrorRandom2019] to a dynamical jamming transition.**

    We can consider doing gradient descent on the coefficients $a$ in the same random features (RF) setting as is discussed in [@meiGeneralizationErrorRandom2019]. If we look at the loss Hessian with respect to these parameters, do we see the hallmarks of jamming in the vicinity of the generalization cusp? Are the two (qualitatively very similar phenomena) in direct correspondence, or are they distinct?

    Potential difficulties:
    - We may need to derive analytic predictions for the generalization error for either the cross entropy or, more preferrably, the squared hinge loss in order to make the comparison to [@baity-jesiComparingDynamicsDeep2019] and [ @spiglerJammingTransitionOverparametrization2019] more direct. This may take the entire time I have to complete this project, as this seems to be a nontrivial issue. Hopefully someone else has already done this.



## Citations

Check out "Disentangling feature and lazy learning in deep neural networks: an empirical study" [@geigerDisentanglingFeatureLazy2020].
