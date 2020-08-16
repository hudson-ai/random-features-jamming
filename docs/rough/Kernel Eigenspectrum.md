## Simplest Case

Let $x_i \sim \mathcal N_p(0, I)$, $X = (x_1, \ldots x_N)^T$ (shape is ($N\times P$)), $K(x, x') = x^Tx'$.

Then $K( X,  X)_{ij} = K(x_i, x_j) = x_i^Tx_j$, so $K( X,  X) =  X  X^T$ (shape is ($N\times N$))



SVD: $X = U D V^T$

$K( X,  X) =  X  X^T = UD^2U^T$

Scatter matrix $= X^TX = V D^2 V^T \sim W_p(I,N)$ (shape is ($P\times P$)),



Therefore our (Gramian) kernel has the same eigenspectrum as the scatter matrix, which has a well-studied Wishart distribution!

For large $P<N$, the $P$ nonsingular eigenvalues (diagonal entries of $D^2$) will be [Marchenko Pasteur](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution) distributed:

$\phi=N/P$

$\zeta_- = (1-\phi^2)^2\,, \quad \zeta_+ = (1+\phi^2)^2$

$\rho(\lambda) = \frac{1}{2\pi\lambda}\sqrt{(\lambda/P - \zeta_-)(\zeta_+ - \lambda/P)}\,, \quad \lambda \in [max(0,\zeta_-), \ \zeta_+]$

