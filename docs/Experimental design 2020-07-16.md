## Intro
In the so-called "lazy learning" regime of deep learning, neural network outputs evolve as a linear combination of random features that are fixed throughout training. These features are given by a first-order Taylor expansion of the model around its weights at initialization. As network width tends to infinity, the inner product in this feature space converges to a deterministic kernel known as the "Neural Tangent Kernel" [@jacotNeuralTangentKernel2018]. This obervation connects neural networks to the deep literature on kernel methods as well as finite-rank approximations to kernels using random features [@rahimiRandomFeaturesLargeScalea].

[@meiGeneralizationErrorRandom2019] studied precise asymptotics for the generalization error of linear models with random features.


## Experiment
Random (NTK, CK, Gaussian) features exhibit a double descent curve and a discontinuous jump in the fraction of unsatisfied constraints as the loss (potential energy) moves from zero to a finite value. By appealing to analogies between DNN training and disordered solids (in particular when using the hinge loss), \[@spiglerJammingTransitionOverparametrization2019, @geigerJammingTransitionParadigm2019] predicted and observed these same features when moving between the overparameterized (corresponding to a vanishing loss) and underparameterized (corresponding to a finite loss) regimes when training DNNs using full-batch gradient descent. They additionally predict (an observe) a discontinuous jump in the fraction of unsatisfied constraints as a function of the number of data per parameter, but so far __I have been unable to recover this phenomenon__.

TODO:

1. Determine whether my inability to recover this jump is due to my use of finite (but small) amounts of regularization. Any finite amount of regularization turns this into a convex optimization problem (but the solution should have very high variance over changes to the data). Going all the way to zero regularization will make the problem non-convex, and my initialization will start to matter (adding another variance term) -- maybe this is where the discontinuity is hiding?
   - __I went to zero regularization and did not see a jump!__
2. If we can't recover the discontinuity by going all the way to zero regularization, if seems that the only place that this phenomenology can be hiding is in feature learning (as opposed to lazy learning).
    - Train last layer with
      - a random initialization
      - a partially trained first layer
      - a mostly trained first layer
      - etc. (logarithmically spaced? features seem to emerge on exponential time-scales)
    - See if we start to see the discontinuity forming
    - Gotchas:
      - we might need to use NTK instead of CK to make strong claims about feature learning
        - then we need to be in feature learning regime rather than mean field
      -  How many parameters are there? Let's say we do CK with a random first layer. Then the only parameters are the second layer. But what if the first layer is partially trained? Do those parameters count now? What if we have a fully trained model? What then?



## Citations
