- Model should look glassy then have a jamming transition then be linear in the thermodynamic limit.
- Why don't we use the limiting gaussian kernels? It seems that they asymptotically have the best generalization ability..?



Experiment on NTKs:

- For finite network, get NTK at initialization
- Solve for analytic gaussian process in the infinite limit (from finite/noisy NTK)
- Plot solution

- Do we see the process go from simple to complex and back to simple?



NTK variablility:

$|\Theta_N^{t=0} - \Theta_N^{t=T}|_F = O(N^{-1/2})$

$|\Theta_N^{t=0} - \Theta_\infin|_F = O(N^{-1/4})$

- Random matrix theory approach to dynamics of $f$? 
- Dependence on number of data points?



How does $\sum_i K(x, x_i)$  vary with P/N?



**Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent**:
$$
\dot f(x) = -\eta \sum_iK_t(x, x_i)\frac{\partial\mathcal L}{\partial f(x_i)}
$$
"The Geometry of Physics":
$$
(\nabla f)^i = -\sum_j g_{ij} \frac{\partial f}{\partial x_j}
$$

- Coordinates are not independent

$$
\dot f(x_i) = -\eta \ \frac{\partial H}{\partial f(x_i)} \\
H = \sum_{i,j} f(x_i) K_{ij} \frac{\partial \mathcal L}{\partial f(x_j)}
$$

E.g. hinge loss, $K_{ij} = 0$ (fix this later):

$\mathcal{L}(\{f(x_i)\}, \{y_i\}) = \sum_i \max(0, 1 - y_if(x_i))$

$\frac{\partial \mathcal L}{\partial f(x_j)} = -y_j \cdot \{y_jf(x_j) < 1\}$
$$
H = -\sum_{i,j} K_{ij} f(x_i)y_j  \{f(x_j)y_j < 1\} \\
= -\sum_i f(x_i) \underbrace{\sum_j y_j K_{ij}  \{f(x_j)y_j < 1\}}_{\phi_i}
$$

$\phi_i$ is a signed, weighted sum of unsatisfied constraints

Mean field?

- $\phi = \sum_i \phi_i $ 
- $H = - \phi/P \sum_i f(x_i)$

Alternatively, notice:
$$
\sum_j y_j K_{ij}  \{f(x_j)y_j < 1\}\\
= \underbrace{\left[\sum_{j|y_j = 1} K_{ij} \{f(x_j) < 1\}\right]}_{\phi_i^+} - \underbrace{\left[\sum_{j|y_j = -1} K_{ij} \{f(x_j) > -1\}\right]}_{\phi_i^-} \\
 = \phi_i^+ - \phi_i^- = \phi_i\\
 \implies \\
 H = - \sum_i f(x_i) (\phi_i^+ - \phi_i^-)
$$

Can we estimate the location of the jamming phase transition as a function of P for large N,P (coordinated limit)? (recall N is # params, P is data-points -- kernel can be explicitly found for N to infinity)

- K is a PxP matrix, and it has variance of O(N^-1/4)
  - Mean?
- should it depend on the distribution of $\{x\}$?
- "Additionally, we see that the error grows in the size of the dataset. Thus, although error grows with dataset this can be counterbalanced by a corresponding increase in the model size" - wide nns are linear



What if $f(x_i) = \pm 1$?

Max-ent:
$$
p[f] = \frac{1}{Z}e^{-H[f]}
$$

- can we estimate $\langle \phi \rangle$?
  - Is there a phase transition for certain values of N, P, limiting kernel $K^\infin$?







Deterministic Langevin:
$$
\dot x_i = \Gamma\sum_j J_{ij}(y_j 1\{y_j x_j < 1\})\\
= \Gamma\left[\sum_{j|y_j = +1}J_{ij}1\{x_j < 1\} -\sum_{j|y_j = -1}J_{ij}1\{-x_j < 1\}\right] \\
$$
