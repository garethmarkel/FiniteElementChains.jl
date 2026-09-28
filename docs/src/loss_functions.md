
### Introduction

This notebook shows how to use different loss functions with FiniteElementChains.jl. 

### Setup

First, activate the environment and load the required libraries.

```Julia
using Pkg
Pkg.activate(".")

using FiniteElementChains
using Gridap
using Flux
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using Gridap.Algebra
using Zygote
using LinearAlgebra
using ForwardDiff
using Distributions
using ChainRules
using Plots
using DelimitedFiles
using Optim
```

### Loss functions and neural network training

Loss functions can significantly influence the final quality of your neural network. It's important to choose the right loss function for your problem. To that end, FiniteElementChains provides functions and tooling to let you swap out a loss function of your choice.

### Prepackaged loss functions

This package defines a type `LossSetup` to provide packaged loss functions to our training methods. It takes a loss function, a preconditioner, and a regularization parameter (Lp norm) as arguments, though these can be filled with placeholders. Obviously this is a fairly limited construction (why these three arguments? why not 5 arguments? etc.), but should be fairly flexible without too much added complexity.

For example, if we want to take the raw L1 loss of the finite element residual, we would do the following.

```Julia
losseval = LossSetup(raw_residual_loss,1.0)
```

In training, that gets the loss and the tangent that needs to be passed to the pullback function of the finite element problem using an `evaluate` method.

```Julia
resid_norm, dlDR = evaluate(losseval, resid_vec)
```

#### L1 Loss

```Julia
losseval = LossSetup(raw_residual_loss,1.0)
```

#### L2 Loss

```Julia
losseval = LossSetup(raw_residual_loss,2.0)
```

#### L1 Loss of Riesz Representation

We may not want the los of the raw residual vector. Technically, that "finite element residual" we get from $$f(v)-a(u_h,v)$$ isn't measured in the same space or on the same scale as the error we're actually trying to control. For us to use the residual to construct a step relative to the values of the field $$u(x,y)$$ (which is implicitly required to step with the neural network parameters) we need to transform that residual into something called the "Riesz representation." We do this by premultiplying the raw residual vector with the "Gram matrix," which contains information on what is meant y size and proximity in your problem.

For example, in a Poisson problem, the Gram matrix is given by:

\begin{equation}
G^{\mathrm{P}}_{ij}=\int_\Omega \nabla \phi_j \cdot \nabla \phi_i\,dx
\end{equation}

In a Maxwell problem, the Gram matrix could be given by given by:

\begin{equation}
G^{\mathrm{M}}_{ij}=
\int_\Omega
\boldsymbol{\phi}_j\cdot\boldsymbol{\phi}_i\,dx
+
\int_\Omega
(\nabla\times\boldsymbol{\phi}_j)
\cdot
(\nabla\times\boldsymbol{\phi}_i)\,dx.
\end{equation}

These are easy to construct in Gridap. Here's an example for a Maxwell problem.

```Julia
a_gram(u,v) = ∫( u ⋅ v + curl(u) ⋅ curl(v) ) * dΩ
M_gram = assemble_matrix(a_gram, Ures, V)
M_gram_fact = cholesky(M_gram)
```

To take the L1 loss of the Riesz representor, construct your loss setup as follows:

```Julia
losseval = LossSetup(riesz_transformed_loss, M_gram_fact, 2)
```

#### L2 Loss of Riesz Representation

```Julia
losseval = LossSetup(riesz_transformed_loss, M_gram_fact, 2)
```

### Constructing a custom loss function

You can make any loss function you want. It just needs to return 1.) a loss, and 2.) a vector containing the gradient of the loss w.r.t. each entry into the residual vector. It also needs to take in 3 arguments.

```Julia
Ures = TrialFESpace(V, Res_exact)  
assem = SparseMatrixAssembler(U,V)


dcrfunc(r) =  ∫( (curl(r) ⋅ curl(v_fef)))*dΩ + ∫( (r ⋅ v_fef))*dΩ
ncrfunc(r) = sqrt(sum(∫(r ⋅ r)*dΩ) + sum(∫(curl(r)⋅curl(r))*dΩ))

function custom_norm_loss(resid_vec::R, M_gram_fact::M, l_x_norm::G) where {R,M,G}
    riesz = M_gram_fact \ resid_vec
    
    rfe = FEFunction(Ures,riesz)

    resid_norm = ncrfunc(rfe)
    dcrpb = assemble_vector(dcrfunc(rfe), assem, Ures) ./ (resid_norm) 
    dlDR = M_gram_fact \ dcrpb
    dlDR = 2 .* cvc

    return resid_norm, dlDR
end

losseval = LossSetup(custom_norm_loss, M_gram_fact, 2)

```

### Which loss function is right for my problem?

Who can say? Many practices we take as givens in machine learning are things that shouldn't work in theory, but empirically work very well. Vastly larger is the graveyard of ideas that are great on paper and go nowhere in practice. Every problem is different, and experimentation is your friend. That said, L1 loss of the Riesz representation seems to work well for a variety of problems. 

In a future post, I plan to conduct a systematic comparison of training strategies and loss functions. Please feel free to reach out or open a pull request if you have an interest in doing this in the meantime.


