# Asymptotics of Generalization in Support Vector Machines with Gaussian Covariates

## Problem set-up
Let
\[
  x_1,\ldots,x_N \overset{iid}{\sim} \mathcal{N_d}(\mu, \Sigma)\,,\\
  y_i = sgn\left(\frac{a_0^Tx_i}{\sqrt d} + \sigma\epsilon_i\right)\,,
\]
with
\[
  a_0\in S^{d-1}(\sqrt{d}), \quad \sigma\geq 0\,,\\
  \langle \epsilon_i\rangle = 0, \quad \langle \epsilon_i^2\rangle = 1, \quad p(\epsilon) = p(-\epsilon), \quad x_i \perp \epsilon_i\,.
\]

We are interested in the asymptotic ($N,d \rightarrow \infty$, $N/d\rightarrow\psi$ finite) behavior of the L2-regularized L2-loss SVM estimator for $a_0$:
\[
  \hat a(\lambda) = \underset{a\in\mathbb{R}^d}{argmin} \left\{ \sum_{i=1}^N  \eta\left(y_i\frac{a^Tx_i}{\sqrt d} \right) + \frac{\lambda}{2}a^Ta\right\}\,
\]
where $\eta(x) = \max(0, 1-x)^2$ is the squared hinge loss. Let's write its derivatives here for future use:

- $\eta'(x) = 2 \cdot \min(0, x-1)$
- $\eta''(x) = 2\cdot \mathbf 1\{x < 1\}$.

Note that these derivatives are defined everywhere except at $x=1$. If this fact gives us trouble later, perhaps we can use a sequence of functions $\eta_k\rightarrow\eta$. [@liuExactHighdimensionalAsymptotics2019] did something similar for the (non-squared) hinge loss, which is arguably even more ill-behaved.

Also, note that we may need to re-scale some terms in our minimization problem in order to ensure they remain of the correct order when we take our limits. If things end up vanishing that really shouldn't (or vice-versa), we will know we have some problems and will have to correct accordingly.

Moving on. For some inverse temperature $\beta\geq 0$, we will define the distribution
\[
  p(a) = \frac{1}{Z}e^{-\beta H(a)}
\]
where
\[
  H(a) = \sum_{i=1}^N  \eta\left(y_i\frac{a^Tx_i}{\sqrt d} \right) + \frac{\lambda}{2}a^Ta\,,
\]

noting that as $\beta\rightarrow\infty$, $p(a)\rightarrow\delta(a - \hat a)$.


## Laplace approximation and a simplifying ansatz
Our Hamiltonian is highly nonlinear in $a$, but because we wish to take $\beta\rightarrow\infty$, we can perform a Laplace approximation:
\[
  p(a) \overset{\beta\rightarrow\infty} \sim \frac{1}{Z}\exp\left\{-\frac{\beta}{2} (a-\hat{a})^T H''(\hat{a})(a-\hat{a})\right\}\,,
\]
where $H''$ is the hessian of $H$. But we don't know what $\hat a$ is!

**Ansatz/approximation**: $\hat a = \alpha^*a_0$

- This holds pretty well in simulation for various values of $\sigma$ and $\lambda$
- This an *approximation* that should only hold on average, i.e. for typical but not arbitrary $\{x_i, y_i\}$
  - How do I interpret such an approximation? Is it too strong? Is there a way to account for the variability that I am averaging out?

We must then have that $\left.\frac{\partial}{\partial \alpha}\langle H(\alpha a_0)\rangle\right|_{\alpha=\alpha^*} = 0$ (where the average is over our disorder terms), giving us a consistency equation for $\alpha^*$:
\[
  \begin{align}
  \frac{\partial}{\partial \alpha}H(\alpha a_0) &= \frac{\partial}{\partial \alpha} \left[\sum_{i=1}^N  \eta\left(\alpha y_i\frac{a_0^Tx_i}{\sqrt d} \right) + \frac{\lambda}{2}\alpha^2a_0^Ta_0\right] \\
  &= \sum_{i=1}^N  \eta'\left(\alpha y_i\frac{a_0^Tx_i}{\sqrt d} \right)y_i\frac{a_0^Tx_i}{\sqrt d} + \lambda d \alpha\\
  \implies \alpha^* &=
-\left\langle\frac{1}{\lambda d} \sum_{i=1}^N \eta'\left(\alpha^* y_i\frac{a_0^Tx_i}{\sqrt d} \right)y_i\frac{a_0^Tx_i}{\sqrt d}\right\rangle
  \end{align}
\]

Note that all of the disorder comes from terms in the form $y_i\frac{a_0^Tx_i}{\sqrt d}$, the density for which we can pretty easily derive an expression given an explicit model of the noise ($\epsilon_i$) distribution -- although integrating this density and subsequently solving this fixed point equation may be no small feat. We should be able to work some of this out numerically or make some more asymptotic arguments here, though. More on this later.



## Hessian trouble
Let's go ahead and compute the Hessian of H:
\[
  \begin{align}
  H''(a) &= \frac{1}{d}\sum_{i=1}^N \underbrace{y_i^2}_{=1}\eta''\left(y_i\frac{a^Tx_i}{\sqrt d}\right)x_ix_i^T + \lambda  I_d\\
  &= \frac{1}{d}\sum_{i=1}^N \eta''\left(y_i\frac{a^Tx_i}{\sqrt d}\right)x_ix_i^T + \lambda  I_d \\
  \implies H''(\alpha^*a_0) &= \frac{1}{d}\sum_{i=1}^N \eta''\left(\alpha^*y_i\frac{a_0^Tx_i}{\sqrt d}\right)x_ix_i^T + \lambda  I_d \\
  \end{align}
\]

When we try and calculate expectations of observables $\langle f(a)\rangle$ (averaging over both our disorder and $p(a)$), we will have two options:

- adding some auxiliary terms to the partition function and computing the relevant derivatives of the corresponding free energy (where we will have to do some replica analysis in order to average over the disorder)
- finding the posterior distribution of $a$ directly (integrating out the densities of $x$ and $\epsilon$) and using that to compute the averages.

Whichever route we take, we will need to calculate an expectation of the form (with some additional indices/sums for our replicas?):
\[
  \left\langle \exp\left\{ -\frac{\beta}{2} (a-\alpha^*a_0)^T \left(\frac{1}{d}\sum_{i=1}^N \eta''\left(\alpha^*y_i\frac{a_0^Tx_i}{\sqrt d}\right)x_ix_i^T\right)(a-\alpha^*a_0) \right\}\right\rangle\,.
\]

What a nightmare! What can we do?

Here's a thought. Let $v_i = y_i\frac{a_0^Tx_i}{\sqrt d}$.  Since $\eta''(x) = 2\cdot \mathbf 1\{x < 1\}$, the term in the middle of our quadratic form becomes
\[
  \frac{2}{d}\sum_{i=1}^N \mathbf{1}\left\{v_i < \frac{1}{\alpha^*}\right\}x_ix_i^T\,,
\]
which is a sum of $\tilde N = N\cdot P\left(v < \frac{1}{\alpha}\right)$ non-zero terms (when $N\rightarrow\infty$) where the terms are outer products of terms $\tilde x\sim p\left(x\mid v < \frac{1}{\alpha^*}\right)$. Then our problem reduces to computing
\[
  \left\langle\exp\left\{ -\frac{\beta}{d}\sum_{i=1}^{\tilde N}\left((a-\alpha^*a_0)^T\tilde x_i\right)^2 \right\}\right\rangle\,,
\]
which I am hoping we can accomplish by approximating $(a-\alpha^*a_0)^T\tilde x_i$ as a univariate random normal variable with mean and variance dependent on $\alpha^*$. I can't exactly justify this, but something *similar* is done in [@canatarStatisticalMechanicsGeneralization2020] in the supplementary materials section *II.A*. This term is a sum of $d$ (going to infinity) terms--suggesting that the central limit theorem may be our friend here--but they are not independent due to the covariance $\Sigma$ of $x$ (probably not such a big deal) as well as the condition that $v < \frac{1}{\alpha^*}$ (probably a much bigger deal). Let's think a little harder about this later.

## Noise distributions and disorder terms
Recalling our definition
\[
  y_i = sgn\left(\frac{a_0^Tx_i}{\sqrt d} + \sigma\epsilon_i\right)\,,
\]
we can note that
\[
  \begin{align}
    P(y = 1 | x) &= P\left(\epsilon \geq -\frac{a_0^Tx}{\sigma\sqrt d}\right)\\
    &= P\left(\epsilon \leq \frac{a_0^Tx}{\sigma\sqrt d}\right) \text{($p(\epsilon)$ is even)}\\
    &\equiv F_\epsilon\left(\frac{a_0^Tx}{\sigma\sqrt d}\right)\,.
  \end{align}
\]
Any CDF with an even density will do as a specification for $F_\epsilon$, but we may be able to choose one (without too much loss of generality) that simplifies our problem a bit. Here is a short list of reasonable choices -- we should circle back here later.

1. $\epsilon \sim \text{Logistic}(0,1)$: $F_\epsilon(x) = \frac{1}{1 + e^{-x}}$
2. $\epsilon \sim \mathcal N(0,1)$: $F_\epsilon(x) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt 2}\right)\right]$
3. $\epsilon \sim \text{Laplace}(0,1)$: $F_\epsilon(x) = \frac{1}{2}\exp(x)\Theta(-x) + \left(1-\frac{1}{2}\exp(-x))\right)\Theta(x)$

Moving on. Let's work out the distribution of $v = y\frac{a_0^Tx}{\sqrt d}$, as it keeps popping up in our calculations.

Let's first denote $l = \frac{a_0^Tx}{\sqrt d} \sim \mathcal N\left(\frac{a_0^T\mu}{\sqrt d}, \frac{a_0^T\Sigma a_0}{d}\right)$. Note that if $\mu = 0$, $\Sigma = I_d$ (which we should probably consider first), $l \sim \mathcal N\left(0,1\right)$. Let's keep this more general for now and consider $l\sim\mathcal N(m, s^2)$ for scalar $m, s$.

We have
\[
\begin{align}
  p_{V|L}(v|l) &= \sum_{y'=\pm 1}\ P(y=y'|l) \delta (v - yl)\\
              &= F_\epsilon(l/\sigma) \delta (v - l) + (1-F_\epsilon(l/\sigma)) \delta (v + l)\\
              &= F_\epsilon(l/\sigma) \delta (v - l) + F_\epsilon(-l/\sigma) \delta (v + l)\quad \text{($p(\epsilon)$ is even)}\\
  \\
  p_{V,L}(v,l) &= p_{V|L}(v|l)p_L(l)\\
  \\
  p_V(v) &= \int dl \ p_{V,L}(v,l)\\
       &= \int dl \ F_\epsilon(l/\sigma)p_L(l)\delta(v-l) + \int dl \ F_\epsilon(-l/\sigma)p_L(l)\delta(v+l)\\
       &= F_\epsilon(v/\sigma)\left(p_L(v) + p_L(-v)\right)\\
  \\
\end{align}
\]

## Some potentially useful papers

[@liuExactHighdimensionalAsymptotics2019], [@dietrichStatisticalMechanicsSupport1999], [@canatarStatisticalMechanicsGeneralization2020]

# Citations
