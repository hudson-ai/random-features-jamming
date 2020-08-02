# Kernels, Reproducing Kernel Hilbert Spaces, and You

What is an RKHS? A functional analysis approach: http://www.stats.ox.ac.uk/~sejdinov/teaching/atml14/Theory_2014.pdf



### Kernels

Let $\mathcal X$ be a non-empty set. The function $k:\mathcal X \times \mathcal X \rightarrow \R$ is said to be a kernel if there exists a real Hilbert space $\mathcal H$ and a map $\phi:\mathcal X \rightarrow \mathcal H$ such that $\forall x,y \in \mathcal X,$
$$
k(x,y) =\langle\phi (x),\phi (y) \rangle_{\mathcal H}\,.
$$

Such a map $\phi: \mathcal X \rightarrow \mathcal H$ is referred to as a feature map, and the space $\mathcal H$ is referred to as a feature space. For a given kernel, there may be more than one feature map.

### Reproducing Kernels

Let $\mathcal H$ be a Hilbert space of $\R$-valued functions defined on a non-empty set $\mathcal X$. A function $k:\mathcal X \times \mathcal X \rightarrow \R$ is called a reproducing kernel of $\mathcal H$ if it satisfies

- $\forall x \in \mathcal X, \ k(\cdot, x)\in \mathcal H$
- $\forall x \in X, \ \forall f \in \mathcal H, \ \langle f, k(\cdot,x)\rangle_{\mathcal H} = f(x)$ (the reproducing property).

In particular, for any $x,y\in\mathcal X$,
$$
k(x,y) = \langle k(\cdot,x), k(\cdot,y)\rangle_{\mathcal H}
$$

*Uniqueness*: for a given Hilbert space $\mathcal H$, if it exists, the reproducing kernel is unique.

### Moore–Aronszajn theorem

Let $k:\mathcal X\times \mathcal X\rightarrow\R$ be positive definite. There is a unique $\mathcal H\subset \R^{\mathcal X}$ with reproducing kernel k. $\mathcal H$ is called the reproducing kernel Hilbert space (**RKHS**) of $k$.

### Mercer’s theorem

We'll come back to this

### Bochner's theorem

### Representer theorem
