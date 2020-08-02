# Why it is interesting

DNNs work. But why? SGD works -- why does it not get stuck in bad local minima?

Deep learning is effective, but training is slow. Characterizing the dynamics could be useful in improving their convergence rates and other properties 

# Background/general context

## Intro

- Similar to disordered systems:

  - "the loss function plays the role of the energy, the weights are the degrees of freedom, and the dataset corresponds to the parameters defining the energy function"
  - Randomness of data is akin to "quenched disorder"
- SGD lets weights evolve dynamically towards low loss values

  - Similar to a "quench" in physics
- Main model in literature for studying dynamics induced by a quench is based on "stochastic Langevin equations" (similar again to SGD)

  - Initial conditions high temperature/ random and uncorrelated with quenched disorder
- disordered systems display glassy dynamics during a quench
  - means the systems gets stuck in local minima for long periods of time
  - Suggests that deep neural networks should suffer this problem too
    - in particular, connections to p-spin models have been previously suggested -- Choromanska?
- HOWEVER: in practice, minima with perfect accuracy aren't too hard too find in a neural network
- Possible explanations for paradox:

  - Loss landscape rough but this doesn't damage performance

    - "gradient descent or stochastic versions of it tend without barrier crossing to the widest and the highest minima"
    - "there exist rare and wide minima which have large basins of attraction and are reached without any sub- stantial barrier crossing by the training dynamics"

  - DNNS work in a regime with no spurious local minima

    - "the loss function, despite being non-convex, is characterized by a connected level set"
    - Supported by large-width analyses (NTK, etc.)

  - Some work suggests that SGD avoids the bad minima even if they do exist

  - Other, noiseless processes (other than SGD) can achieve similar performance, suggesting there is no noise-induced barrier crossing

## Glassy Dynamics

Two main observables for characterizing slow dynamics:

- Energy as a function of time 
  - Decreases and slowly hits an asymptote
    - "Slow" in comparison to an exponential relaxation (**Fig 1**)
      - Power law of logarithm of time typical for glasses
- Two-time correlation function $\Delta (t_w, t_w + t)$
  - "aging" typical of low-temp quenching (**Fig 1**)
    - Time-scale controlling decorrelation times is a function of $t_w$
      - First the decorrelation times increase independently of $t_w$ (looks stationary)
      - Then hangs out in plateau for time that increases with $t_w$ (aging) 
    - The height/correlation value of the plateau ("Edwards-Anderson parameter") quantifies to what extent the system gets stuck in local minima
- Note the existence of LONG (exponential in # params) time-scale dynamics in which barrier crossing takes place

Slow dynamics and aging are distinctive features of glassy systems

- Due to increasing number of flat directions, not barriers
- Gradient descent: always confined to a single basin of attraction but number of negative eigenvalues of the Hessian decreases with time 

# What the paper is about

- Numerically analyze the training dynamics of deep neural networks using methods of glassy systems.

- two main issues/questions
	
	- (1) "the complexity of the loss landscape and of the dynamics within it"
	- (2) "to what extent DNNs share similarities with glassy systems"
	
	
	

# Methods

- (1) "probing the training dynamics through the measurement of one and two-point correlation functions, as done in physics, we infer properties of the loss landscape in which the system is evolving"
- (2) "comparing the results obtained for mean- field glasses to measurements performed for realistic DNNs we test the analogy between these systems"
- Analysis on several architectures and datasets:
  - Architectures (one toy model, and three typical networks):
    - Toy Model
      - 1 layer
      - large width, ($10^4$ hidden nodes)
      - large width -> non-existance of local minima
    - Fully Connected 
    - Small Net
    - ResNets
  - Datasets:
    - MNIST
    - CIFAR-10
    - CIFAR-100

# Main results

### Overview

- Standard/overparameterized: 
  - similar to glassy: absence of barrier crossing
  - distinctive dynamical behavior $\rightarrow$ not glassy 
- Behaviour is similar to glassy dynamics in underparameterized regime
- Suggests a phase transition ( confirmed in later works )

### Summary of experimantal findings:

#### Loss/energy function over time
- Initial exploration of high-loss configs
  - Loss stays relatively constant at beginning of training (until $t_1$)
- followed by descent that appears linear in log(t)
  - Much slower than the power law time of p-spin model w/ comparable degrees of freedom

- Drastically slows down at $t_2$ and asymptotically reaches minimum possible value (global minimum)
  - Very different from p-spin model, in which one of its highest and widest minima is reached instead
    - Convergence to global minima requires barrier crossing in p-spin and corresponds to exponential timescales
  - Evidence that no barrier-crossing is involved

Authors suggest that this is similar to p-spin dynamics but without big local/bad minima

- Is this a flaw in the argument? 
  - "p-spin but a little different" isn't p-spin...

#### Two-point correlation

 Mean square displacement of the degrees of freedom/weights $\Delta(t_w, t_w+t) = \frac{1}{M}\sum_{i=1}^M (w_i(t_w) - w_i(t_w + t))^2$

Three observed regimes:

- $t_w < t_1$: $\Delta(t_w, t_w+t) $ collapses to single curve 
- $t_1 < t_w < t_2$: $\Delta(t_w, t_w+t) $ clearly depends on $t_w$
  - characteristic time increases with $t_w$
  - $t > t_2 - t_w$: $\Delta(t_w, t_w+t) $ flat
- $t_2 < t_w$: characteristic time stops increasing with $t_w$

Large time dynamics can be largely explained by diffusion in weight space:

- In a diffusing system,  motion is purely driven by noise
- noise $D$ estimated with variance of loss function's gradient
- Larger $D$ implies larger $\Delta(t_w, t_w+t)$
  - So normalized $\Delta(t_w, t_w+t)/D(t_w)$ used in graphs
  - Once we normalize by $D(t_w)$, $t_w > t_2$ curves collapse into a single curve, suggesting a stationary regime driven by diffusion near the bottom of the loss landscape
    - This is just a heuristic -- in particular D(t_w) is changing, so $\Delta(t_w, t_w+t)/D(t_w)$ is only a good normalization for small-ish $t$

In a p-spin model, aging continues  even when energy approaches asymptotic value

- In training, aging stops at $t_2$ and becomes stationary except for changes in noise strength

In p-spin, all curves are the same for small $t$, no matter the $t_w$ -- this is not present in DNNs

No observed plateau for $t_w > t_2$, contrary to p-spins (fig 1b)

- Instead log-log plot of $\Delta(t_w, t_w+t)/D(t_w)$ vs $t$ is a straight line (except for change of $D(t_w)$ with time)
  - Characteristic of diffusion

#### Summary

- Similar initial exploration of high-loss space ($t < t_1$)
- Aging present for $t_w < t_w < t_2$ 
- Aging disappears for $t_w > t_2$ and diffusion dominates 
  - distinct from p-spin models

### Conclusions:

- "although the first regimes share similarities with the dynamics of mean-field glasses after a quench, **the final regime does not**"
  - No aging for $t_w > t_2$
  - DNNs can reach global minima, while p-spin models get stuck in local minima for exponentially long periods of time
  - Suggests the geometry and dynamics at the bottom of the landscape are different from p-spin models
- Slowness of dynamics does not seem to come from barrier crossing
  - toy model (with no expected barriers) behaves similarly to the more realistic models
- Over-parameterization of DNNs may play a role in making convergence to global minimum possible
  - Decreasing number of nodes in toy model leads to typical glassy behavior 
    - convergence to bad local minimum
    - 2-point correlation qualitatively similar to glassy systems
  - Authors conecture existence of phase transition from hard, glassy dynamics in the underparameterized regime to easy dynamics in the overparameterized regime
    - Easy regime characterized by many flat directions but no bad minima
    - Hard regime characterized by rough, glassy dynamics with high barriers
    - Similar to jamming transition in ordered solids 
      - Examined in further work

# Limitations and Approximations, inaccuracies or errors of judgement (if applicable)

- Conclusions reached through qualitative/visual analysis of plots

- $\Delta(t_w, t_w+t)/D(t_w)$ is only a good normalization for small-ish $t$
- plateau of $\Delta(t_w, t_w+t)/D(t_w)$ could exist for larger t than looked at in this paper (but how to account for changing D?)

# 



