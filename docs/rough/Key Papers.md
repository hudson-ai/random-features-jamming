Here is a list of papers put in an order that roughly aligns with my mental organization of these topics.


- **Comparing Dynamics: Deep Neural Networks versus Glassy Systems** [@baity-jesiComparingDynamicsDeep2019]

  This is the original paper cited in the project abstract and compares the dynamics of DNN training with those of glassy (in particular p-spin) systems. This paper broaches the idea of a "jamming transition" between the over- and under-parameterized regimes of DNNs, where the under-parameterized dynamics correspond to the "rough" landscape of glassy systems and is charactarized by exponential timescales to reach global optima. The over-parameterized dynamics correspond to a "smooth" regime in which it is easy to achieve global maxima.

- **A jamming transition from under- to over-parametrization affects loss landscape and generalization** [@spiglerJammingTransitionOverparametrization2019]


  This paper examines the relationship between the jamming transition and generalization in DNNs. My main take-away from this paper is that if you start from a very narrow network and begin to increase the width, the training loss decreases more or less monotonically. On the other hand, test loss decreases to a local minimum before increasing to a cusp at the jamming transition and then decaying further as you move deeper into the over-parameterized regime. This suggests that the phenomena of over-fitting is related to being in close proximity to this jamming transition. One more interesting note is that early stopping, a method commonly used by practitioners to avoid over-fitting, effectively removes the cusp and recovers a near-monotonic decrease of test loss, even through the jamming transition.

- **The jamming transition as a paradigm to understand the loss landscape of deep neural networks** [@geigerJammingTransitionParadigm2019]

  This paper discusses more in depth the analogies between the jamming transition in disordered solids and the transition between the under- and over-parameterized regimes in DNNs. The authors describe the structure of the Hessian of the potential energy for spherical  and elliptical particles at jamming. For spheres at jamming, the number of degrees of freedom and the number of active constraints (stemming from contacts) are equal (*isostatic*). For ellipses at jamming, on the other hand, the number of constraints is smaller than the number of degrees of freedom (*hypostatic*). These properties can be seen from a decomposition of the Hessian into a sum of two matrices, one positive semidefinite, and one that is negative definite for spheres and one that is not necessarily negative definite for ellipses.

  While the perceptron has been shown to be in the same universality class as spherical particles, this paper indicated that DNNs jam in a manner more similar to ellipses. Loss landscapes differ for different activation functions, but we can see some phenomenological similarities between the empirical Hessian at jamming and what we should expect from elliptical particles.

  Additionally, the gap and overlap distributions (the fraction of data which are almost correctly or almost incorrectly classified) follow power-laws with a novel set of exponents near jamming. A lot of information not captured by the Hessian can be seen in these scaling properties, e.g. marginally stability of certain jammed states, leading to avalanches and bursts in training (*note: I don't really understand this fully and clearly need to read a bit more about universality classes and power-law scaling...*).

- **Scaling description of generalization with number of parameters in deep learning** [@geigerScalingDescriptionGeneralization2019]

  Here, the neural tangent kernel (NTK) comes to the stage as a lens through which to understand the generalization properties of DNNs at and past the jamming cusp. In short, all else held constant, test error increases linearly in the variance of the output of the learned function away from the training points.

  For large $N$, the NTK has initialization fluctuations $O(N^{-1/4})$ and training fluctuations $O(N^{-1/2})$, so initialization fluctuations dominate. Training dynamics of the DNN are linear in the NTK, leading the authors to hypothesize that fluctuations in the final learned function are due to initialization and are also of order $O(N^{-1/4})$. They present some experimental evidence for this claim.

  These scaling properties break down at the jamming transition, however, and the authors predict that the cusp seen in training error at this boundary is due to diverging fluctuations in the function outputs (if there is no regularization).

  Finally, the authors then suggest that the "most efficient" way to minimize test error is to ensemble average just after the jamming cusp but before $N$ is too big (*I don't understand why, and I need to dig into this more!*). This suggests that knowing the location of this cusp may be very useful in achieving good generalization.

- **Neural Tangent Kernel: Convergence and Generalization in Neural Networks** [@jacotNeuralTangentKernel2018]

  This paper introduced NTKs. Here is a brief review on the math:

  Let $\mathcal X = \{x_1,\ldots, x_N\}$ be our training data, $\mathcal Y = \{y_1,\ldots, y_N\}$ be our target outputs, and $f(\cdot,\theta_t)$ be our DNN/function with parameters at time $t$, $\theta_t = \{\theta_t^1,\ldots,\theta_t^P\}$. For simplicity, we will denote $f(\cdot, \theta_t) \equiv f_t(\cdot)$.

  Consider a loss function:
  $$
    \mathcal L (\mathcal X, \theta_t) = \sum_i l(f(x_i, \theta_t), y_i)\,.
  $$

  Then we can write standard gradient descent (here I will write the continuous analog rather than the discrete version for simplicity):

  $$
    \dot \theta_t = -\eta \nabla_{\theta_t}\mathcal L = -\eta\nabla_{\theta_t} f_t(\mathcal X)^T \nabla_{f_t(\mathcal X)}\mathcal L\,,
  $$
  where we simply used the chain rule. To clairify the notation:
  $$
    f_t(\mathcal X) = [f_t(x_1) \cdots f_t(x_N)]^T\,,
  $$
  and
  $$
    \nabla_{\theta_t} f_t(\mathcal X) = \begin{bmatrix}
      \frac{\partial}{\partial\theta_t^1} f_t(x_1) & \cdots & \frac{\partial}{\partial\theta_t^P} f_t(x_1) \\
      \vdots & \ddots & \vdots \\
      \frac{\partial}{\partial\theta_t^1} f_t(x_N) & \cdots & \frac{\partial}{\partial\theta_t^P} f_t(x_N) \\
    \end{bmatrix}\,.
  $$

  We can then write the evolution in function space:
  \[
    \begin{align}
      \dot f_t(\mathcal X) &= \nabla_{\theta_t} f_t(\mathcal X) \dot \theta_t = -\eta (\nabla_{\theta_t} f_t(\mathcal X)\nabla_{\theta_t} f_t(\mathcal X)^T)\nabla_{f_t(\mathcal X)}\mathcal L \\
      &=-\eta\Theta_t \nabla_{f_t(\mathcal X)}\mathcal L
    \end{align}
  \]
  where
  $$
    (\Theta_t)_{ij} = \sum_{l=1}^P \frac{\partial f_t(x_i)}{\partial\theta_t^l}  \frac{\partial f_t(x_j)}{\partial\theta_t^l} \,.
  $$

  The main insight of this paper is that $\Theta_t$ (hereby known as the neural tangent kernel) is random at initialization and varies during training, but in the infinite-width limit, it converges to an explicit limiting kernel, and it stays constant during training. This connects DNNs to the rich literature on kernel methods. In particular, the positive semidefinite-ness of the kernel has nice implications for the convergence of DNN training in the large width limit, and the authors proved positive *definiteness* for data supported on the sphere with non-polynomial nonlinearities. It can also be shown that training converges most quickly along the largest kernel principle components, giving some theoretical motivation for early stopping.

- **Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent** [@leeWideNeuralNetworks2019]

  This paper shows that NTK dynamics are *linear* in the weights. In particular:
  $$
    f_t(x) = f_0(x) + \nabla_{\theta}f_0(x)|_{\theta=\theta_0}(\theta_t - \theta_0)
  $$
  While this linearization only holds exactly in the infinite width limit, the authors found good experimental agreement for networks of "reasonable" size (robust across different architectures, optimization methods, and loss functions).

- **The asymptotic spectrum of the Hessian of DNN throughout training** [@jacotAsymptoticSpectrumHessian2020]

  I still need to read this, but here is the abstract:

  > In this article, we show that the NTK allows one to gain precise insight into the Hessian of the cost of DNNs. When the NTK is fixed during training, we obtain a full characterization of the asymptotics of the spectrum of the Hessian, at initialization and during training. In the so-called mean-field limit, where the NTK is not fixed during training, we describe the first two moments of the Hessian at initialization.

  It looks like we can characterize the Hessian from the NTK! In the lazy limit, this describes the Hessian throughout training, while it only describes the first two moments of the Hessian at initialization in the mean-field limit (is this the same as the "active limit"?). For jamming, we are interested in the Hessian of the optimized network (I think?), so this may only be useful in the lazy limit. I am curious about the "linear limit", i.e. when the number of parameters is linear in the number of data, and I don't yet know if this result generalizes to that case.

  After some reading, here are some insights. The loss Hessian can be decomposed into a sum of two matrices:
  $$
    H = I + S
  $$
  Where $S$ vanishes at critical points of the loss and $I$ can be expressed in terms of the NTK $\Theta$. Specifically (using the notation from the paper),
  $$
    \Theta = \mathcal D Y^{(L)}(\mathcal D Y^{(L)})^T \quad \text{and} \quad I = (\mathcal D Y^{(L)})^T \mathcal H C \mathcal D Y^{(L)}
  $$
  Where in our previously used notation,
  $$
  \mathcal D Y^{(L)} = \nabla_{\theta_t} f_t(\mathcal X)\,,
  $$
  and
  $$
    \mathcal H C = (\nabla_{f_t(\mathcal X)} \otimes \nabla_{f_t(\mathcal X)})\mathcal L\,.
  $$
  Notice that $\mathcal H C$ is typically a diagonal matrix, and for MSE loss, it is $\frac{1}{N} I$. Therefore in this case, the eigenvalues of the Hessian at a minimum are just the eigenvalues of the NTK scaled by $\frac{1}{N}$.

  Important question: if NTK is not constant (active regime), are we talking about the NTK at initialization or at its terminal value? I need to dig into this paper a bit deeper.

- **Spectra of the Conjugate Kernel and Neural Tangent Kernel for linear-width neural networks** [@fanSpectraConjugateKernel2020]

  This paper came out *just a few days ago* and is the first one that I have seen that derives properties of the NTK in the "linear limit," i.e. the number of parameters is linear in the number of data. I have a hunch that this is a really important limit because jamming transition happens when the parameters and the number of data are of the same order, but letting both of those numbers go to infinity in a coordinated way would make it possible to approach the jamming problem with ransom matrix theory! Here is the abstract (I haven't had the chance to read this one too closely yet):

  > We study the eigenvalue distributions of the Conjugate Kernel and Neural Tangent Kernel associated to multi-layer feedforward neural networks. In an asymptotic regime where network width is increasing linearly in sample size, under random initialization of the weights, and for input samples satisfying a notion of approximate pairwise orthogonality, we show that the eigenvalue distributions of the CK and NTK converge to deterministic limits. The limit for the CK is described by iterating the Marcenko-Pastur map across the hidden layers. The limit for the NTK is equivalent to that of a linear combination of the CK matrices across layers, and may be described by recursive fixed-point equations that extend this Marcenko-Pastur map. We demonstrate the agreement of these asymptotic predictions with the observed spectra for both synthetic and CIFAR-10 training data, and we perform a small simulation to investigate the evolutions of these spectra over training.

- **Nonlinear random matrix theory for deep learning** [@penningtonNonlinearRandomMatrix2019]

  This paper is cited by [@fanSpectraConjugateKernel2020] (and may be the first paper to derive spectral properties in the linear limit?). They compute the spectral density of $M=YY^T$ for $Y = f(WX)$ where f is a point-wise nonlinearity and $W_{P\times D}$, $X_{D\times N}$ are matrices with i.i.d. Gaussian entries (with appropriate scaling of variance) representing weights and data, respectively, where there are N data points in D dimensions being projected into P dimensional space. In particular, the limit is taken such that
  $$
    \phi = \frac{D}{N}, \quad \psi = \frac{D}{P}
  $$
  are finite constants as $D, N, P \rightarrow \infty$.

  This corresponds to the conjugate kernel (CK) of a single layer network with linear width (linear in the number of data) but also with linear data dimensionality, which feels less natural to me. I need to think about this a bit.

  The CK, is an alternative lens (compared to the NTK) to deep learning. The idea is that you initialize a random network and then only train a linear model that takes the last layer of network activations as inputs (i.e. you only train the last layer). This may be a far easier space to work in, but it also may not have a jamming transition like standard DNNs, and it is quite different from how people use DNNs in practice. I am not yet sure if this is useful.

- **Toward Deeper Understanding of Neural Networks: The Power of Initialization and a Dual View on Expressivity** [@danielyDeeperUnderstandingNeural2017]
  The source of the CK formulation? I need to understand the particular initialization regimes that give rise to CK, NTK, etc during gradient descent, and this paper *might* help.

- **The generalization error of random features regression: Precise asymptotics and double descent curve** [@meiGeneralizationErrorRandom2019]

  Incredible, incredible paper. Derives analytic results for the cusp in generalization error when $D,N,P \rightarrow \infty$ with finite ratios in the setting of random features ridge regression in where the features are activations of an untrained, random network (i.e. the CK setting).

- **Double Trouble in Double Descent : Bias and Variance(s) in the Lazy Regime** [@dascoliDoubleTroubleDouble2020]
  d’Ascoli, et. al. do something with the results if the Mei paper. Still need to read this.

## Citations
