## Project gist/proposal

**Background and the main idea**:

Following its hypothesized existence in [0], the jamming transition in neural networks between the over- and under-parameterized regimes was further investigated in [1] and [2]. I am wondering if you think it would be achievable to follow up on this work using the neural tangent kernel (NTK) formulation.

In the NTK point of view, as shown in [3], the evolution of the neural network in **function space** can be described by:

$$
\dot f(x) = -\eta \sum_iK_t(x, x_i)\frac{\partial\mathcal L(f(x_i), y_i)}{\partial f(x_i)}
$$

$$
f(x)_{t+1} = f(x)_{t} -\eta \sum_iK_t(x, x_i)\frac{\partial\mathcal L(f(x_i), y_i)}{\partial f(x_i)}
$$


where the summation is taken over the training data (or batches of the training data), $\mathcal L $ is the loss function, $\eta$ is the learning rate, and $K_t$ is the neural tangent kernel at time $t$ with

$$
K_t(x_i, x_j) = \nabla_\theta f(x_i, \theta)^T  \nabla_\theta f(x_j, \theta)_{\theta= \theta_t}
$$

where $\theta_t$ is the vector of network weights at time $t$. Something very interesting about the NTK is that in the large-width limit, where the number of parameters $P \rightarrow \infty$, the kernel function $K$ is deterministic and constant throughout training. Thus the interactions are a fixed function of random data, constituting quenched disorder.

[1] and [2] take the point of view that the degrees of freedom in a neural network correspond to the weights $\theta$. In the NTK approach, on the other hand, the degrees of freedom correspond to the *network outputs* evaluated on the training data $\mathcal X$, where the coordinates are correlated in a way that is specified by the kernel (which in turn is specified by a choice of network architecture). Of course if  $P << N = |\mathcal X|$, the Gramian matrix $K(\mathcal X,\mathcal X)$, is singular and has rank at most $P$. So in this regime, $P$ actually acts as the number of degrees of freedom as is more traditionally considered in the statistical learning literature.

**Question**:

What happens if we take the thermodynamic limit as $N \rightarrow \infty$, in particular where we also let $P \rightarrow \infty$, with $N/P  = \phi$ finite and non-vanishing? Here I am appealing to the description in [3] which likens N/P to a particle density.

In this limit, we could use the fact that $K$ has a fixed, closed form, allowing us to look at the asymptotics of its eigenspectrum.

Can we recover the existence of a jamming transition at some $\phi = \phi_c$? We would only be able to do this for some specified data distribution (e.g. uniform over the surface of the unit sphere in some number of dimensions), of course (and for some specific form of loss function $\mathcal L$ and choice of nonlinearity...), but I think it could be very interesting nonetheless. If neural kernels prove too unwieldly, we could potentially work with random Fourier features or truncated Taylor expansions of simple kernels like the Gaussian. I am sure there is at least some simplification of this problem formulation that is tractable..!

I have not delved too deeply into the disordered solids literature and am unsure what exactly a proof of the existence of a jamming transition looks like -- are there any pieces that we are missing besides the asymptotics of $K$ as a function of $\phi$?

**More prior work**:

[4] considers the relationship between the jamming transition and the generalization abilities of the network by looking at the NTK, but it doesn't explicitly derive $\phi_c$ from the kernel as I would be interested in doing.

[5] takes an interacting particle approach to neural network training dynamics, but it looks at the dynamics in weight space rather than function space. It may still prove to be useful.

**Additional information**:

It may also be useful to note the following characterization of the fluctuations of the kernel (more precisely, the Gramian matrix over the training data) from [4]:
$$
|K^P_{t=0} - K^P_{t=T}|_F = \mathcal{O}(P^{-1/2})
$$

$$
|K^P_{t=0} - K^\infty|_F = \mathcal{O}(P^{-1/4})
$$

where $K^P_{t=0}$ is the kernel or a randomly initialized network with width $P$, $K^P_{t=T}$ is the same kernel trained for some time $T$, and $K^\infty$ is the constant, asymptotic form of $K$. This means that the fluctuations due to random initialization of the network are the next largest source of noise in the NTK after the quenched disorder due to the data sampling distribution and fluctuations throughout the training process are relatively small.

I am not exactly sure how this description of the noise scales with the number of data in the large $N$ limit, but [4] seems to be an interesting starting point, in particular if we want to consider ensemble averages over large but finite kernels (this is potentially future work or something I can work on if I get lucky and have quick success with the inftyite with limit). [6] may also prove useful as it considers some finite-width corrections to the NTK.

## References

[0] Baity-Jesi, Marco, et al. "Comparing dynamics: Deep neural networks versus glassy systems." *Journal of Statistical Mechanics: Theory and Experiment* 2019.12 (2019): 124013.

[1] Spigler, S., et al. "A jamming transition from under-to over-parametrization affects generalization in deep learning." *Journal of Physics A: Mathematical and Theoretical* 52.47 (2019): 474001.

[2] Geiger, Mario, et al. "Jamming transition as a paradigm to understand the loss landscape of deep neural networks." *Physical Review E* 100.1 (2019): 012115.

[3] Lee, Jaehoon, et al. "Wide neural networks of any depth evolve as linear models under gradient descent." *Advances in neural information processing systems*. 2019.

[4] Geiger, Mario, et al. "Scaling description of generalization with number of parameters in deep learning." *Journal of Statistical Mechanics: Theory and Experiment* 2020.2 (2020): 023401.

[5] ROTSKOFF, GRANT M., and E. R. I. C. VANDEN-EIJNDEN. "TRAINABILITY AND  ACCURACY OF NEURAL NETWORKS: AN INTERACTING PARTICLE SYSTEM APPROACH." *stat* 1050 (2019): 30.

[6] Hanin, Boris, and Mihai Nica. "Finite depth and width corrections to the neural tangent kernel." *arXiv preprint arXiv:1909.05989* (2019).


## Further Thoughts

Are there (stable) fixed points of the dynamics that are not (local) minima of the loss function?

- There are fixed points of that are not minima of the loss iff the null-space of the Gram matrix is not trivial (a sufficient condition is P > N)
  - But are they stable?
  - Taylor expand $\dot f$ about $f^*$ where $\frac{\partial \mathcal L}{\partial f}|_{f=f^*} \in Ker(K)$ ?
