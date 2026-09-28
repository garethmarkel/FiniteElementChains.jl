## Finite Element Methods 

In inverse modelling, we use finite element methods to accurately model physical systems by forcing the numerical solution to follow constraints defined by a PDE. This adds an amount of inductive bias, which restricts the space of possible solutions to ones following a certain structure. Even if this PDE doesn't perfectly match the "true" physical law, by including it we add information about our problem that could be hard or impossible to learn from the data. 

In many finite element applications, we use about $H^1$ elements. These force the divergence of the gradient of a field $u$ to be determined by sources and sinks in the field. But these aren't the only physical quantities we're worried about! For example, in electromagnetics, we are often worried about the curl of the field, and use an $H(curl)$ space which forces the curl of the curl of the field to be constrained by the sources and sinks.

For an illustration of what this means, take the following picture. The iron filings organize themselves into a curling field around the magnet. Changing the orientation of the curling filings would imply a change in the placement and position of the magnetic field. We apply these constraints to finite element problems using moment-based finite elements.

![Physics Magnetism Study Guide: Key Concepts & Laws | Notes](https://static.studychannel.pearsonprd.tech/study_guide_files/physics/sub_images/d41ac28f_image_7.png)

### What is a moment-based element?

A moment-based element sets each degree of freedom equal to an integral of some function of the underlying field.

$$f(u(x))= D_{freedom}$$

One example is the Nedelec edge element, which sets each degree of freedom equal to the tangential component of the field passing over a given edge.

$$\int_e \mathbf{u}\cdot\mathbf{t}\,ds = d_e$$

### How are these calculated?

We're interested in obtaining an estimate of an underlying field $\mathbf{u}$, mapping a $d$ dimensional coordinate set to a $m$ dimensional output vector.

$$\mathbf{u} : \mathbb{R}^d \to \mathbb{R}^m$$

The object we estimate, $\hat{\mathbf{u}}$, needs to do the same thing. In the finite element method, we fit $\hat{\mathbf{u}}$ by updating the degrees of freedom to minimize a finite element residual. In the most common FEM case, where the solution is in $H^1$, the degrees of freedom are directly $\hat{\mathbf{u}(x)}$. When we use moment-based elements, the degrees of freedom are functions of the estimated field, $f(\hat{\mathbf{u}(x)})$. These functions are defined by the element's geometry.

#### Setup

For each cell, there is a physical element $K$ and a reference element $\tilde{K}$. We will use the tilde to refer to anything in reference space--so a physical point would be $x$ and a reference point would be $\tilde{x}$.

```Julia
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

π2 = π^2

# Manufactured solution
E_exact((x,y)) = VectorValue(
    cos(π*x)*cos(π*y),
    sin(π*x)*sin(π*y)
)

Res_exact((x,y)) = VectorValue(
    0.0,
    0.0
)

# change HERE

kappa((x,y)) = 1/(1 + x^2 + y^2 + (x-1)^2 + (y-1)^2)
# Forcing term (derived analytically)
J(x) = (2*π2 + kappa(x)) * E_exact(x)

domain = (0,1,0,1)
partition = (50,50)
model = CartesianDiscreteModel(domain, partition)

order = 1
reffe = ReferenceFE(nedelec, 0)

V = TestFESpace(
    model,
    reffe;
    conformity=:HCurl,
    dirichlet_tags="boundary"
)

Ω = Triangulation(model)

momentbasedinfo, interior_coords = MomentBasedElementTools(V, Ω)
```

#### 1. Get the Jacobian of the transformation, for each element

Let's define a mapping:

$$F_K: \tilde{K} \to K, F_K(\tilde{x}) = x$$

And a jacobian:

$$J(\tilde{x}) = \frac{\partial x}{\partial \tilde{x}}$$

The Jacobian defines a mapping from $\mathbf{u}$ to $\tilde{\mathbf{u}}$. This is called the Piola transform. For Nedelec elements, the Piola transform used is the covariant Piola transform, and is given by the following equation.

$$ \mathbf{u}(x) = J^{-T} \tilde{\mathbf{u}}(\tilde{x})$$ 

This jacobian can be obtained with the following snippet in Gridap. We transpose it to get the value that gets multiplied to physical $\mathbf{u}$.

```Julia
jac_vec = lazy_map(x-> x.args[1].value', V.fe_dof_basis.cell_dof)
```
#### 2. Get the nodes of the points used for integration on each reference element

The moment-based element degrees of freedom are defined as an integral. This integral is defined by a quadrature rule. Gridap takes care of this in the background when you set up your problem. For each element, you can get the relavant nodes with the following snippet.
```Julia
node_vec = lazy_map(x-> x.dofs.predofs.nodes, V.fe_dof_basis.cell_dof)
```

This will give you the points in the reference space. Let's map them to points in the physical space using this snippet:

```Julia
cell_map_vec = get_cell_map(Ω)
xkvec = lazy_map((x,y)-> x(y), cell_map_vec,node_vec)
xkvec=vcat(collect(xkvec)...)
```
We'll also create a vector denoting which indices of xkvec correspond to which element.

```Julia
cell_indices = lazy_map(n -> ((n-1)*length(node_vec[1]) + 1):(n*length(node_vec[1])),collect(1:length(node_vec)))
```

#### 3. Get the matrix $M$ to describe how reference-space degrees of freedom are blended to compute physical degrees of freedom

Woah, reference-space degrees of freedom? What? Don't worry, we haven't talked about those yet. This post is code forward, so before we calculate anything, we're going to make sure we have every piece of the puzzle. 

In later steps, we're going to calculate the degrees of freedom on the reference element. These reference degrees of freedom might contribute to multiple physical degrees of freedom, depending on the order and type of your element. If $D$ is a degree of freedom, we define the following mapping:

$$ D = M\tilde{D}$$

We can get this for each element using the following snippet.

```Julia
map_vec = lazy_map(x-> x.dofs.values', V.fe_dof_basis.cell_dof)
```

#### 4. Get the weights of each integration point's contribution for each $\tilde{D}$ in each cell

Gridap nicely prepackages the weights for the integrals in the moment-based degrees of freedom for you. Our goal is to extract these, and create a matrix $B$ that can be multiplied with a vector of $\tilde{\mathbf{u}(x)}$ evaluated at the integration points from step 2. We do this with the following function:

```Julia
function create_B_matrix(V::FESpace)
    nodelist = V.fe_dof_basis.cell_dof[1].dofs.predofs.nodes
    
    BigB = zeros(
        VectorValue{2, Float64}, 
        length(V.fe_dof_basis.cell_dof[1].dofs.predofs), 
        length(nodelist)
    )
    
    n_faces = length(V.fe_dof_basis.cell_dof[1].dofs.predofs.face_own_moms)
    
    @inbounds for i in 1:n_faces
        n_moments_face = length(V.fe_dof_basis.cell_dof[1].dofs.predofs.face_own_moms[i])
        
        node_mapping = V.fe_dof_basis.cell_dof[1].dofs.predofs.face_nodes[i]
        
        moments = V.fe_dof_basis.cell_dof[1].dofs.predofs.face_moments[i]
        
        @inbounds for j in 1:n_moments_face
            rowid = V.fe_dof_basis.cell_dof[1].dofs.predofs.face_own_moms[i][j]
            
            BigB[rowid,node_mapping] .= moments[:,j]
        end
    end
    
    return BigB
end
```

#### 5. Map each element's matrix $B$ to the physical space by multiplying with the element's jacobian matrix $J$

As mentioned above, we want to multiply $B$ with a vector of $\tilde{\mathbf{u}}(\tilde{x})$ evaluated at the integration points from step 2. However, there's one problem with this. The field we're estimating, $\tilde{\mathbf{u}}(\tilde{x})$, is a map over physical space, not reference space. We could do this by applying the Piola transform to $\mathbf{u}(x)$ before multiplying with $B$. But this is inneficient--$B$ isn't changing, $J$ isn't changing, so doing it in two steps just wastes computations in every iteration of our process.

Instead, we can acheive the same result by pre-computing and storing the product of $B$ and $J$:

```Julia
BigBprodJ = lazy_map(x -> BigB .⋅ x,jac_vec)
```

#### 6. Multiply BprodJ with the mapping matrix $M$

Same logic as the last section--let's precompute this product up front and save ourselves some time in our eventual optimization loop.

```Julia
BigMprodBprodJ = collect(lazy_map((x,y) -> vector_value_mat_mul(x, y) ,map_vec,BigBprodJ))
```

`vector_value_mat_mul` is a convenience function for dealing with Gridap's `VectorValue` type. It's included in this package.

#### 7. Leave room for Dirichlet constraints

In most problems, we constrain some degrees of freedom to have specific values. Since these are not free degrees of freedom, we need to exclude them from our calculations. This is easy, as we just assign those 0 weight.

```Julia
BigPos = lazy_map(x->[ifelse(i > 0, 1.0,0.0) for i in x],V.cell_dofs_ids)
BigPBD = collect(lazy_map((x,y) -> x .* y,BigMprodBprodJ,BigPos))
```

#### 8. Get predictions at physical points

Let's say we have some function $\hat{\mathbf{u}}$, and a set of physical integration points $x_k$ representing the physical coordinates of the points on the reference cells used to calculate the integrals involved in the degrees of freedom. In this step, we calculate $\hat{\mathbf{u}}(x_k)$. If we have a function `upredfunc`, this is as simple as:

```Julia
preds = upredfunc.(interior_coords)
```
#### 9. Put it all together

We now just need to multiply each cell's prediction vector with it's weight matrix, using the following chunk.

```Julia
dofvals = collect(lazy_map((x,y) -> matvec_dot(x,@view preds[y]), BigMBJ, cell_indices))
```

`matvec_dot` is another convenience function for Gridap types.

### Common kinds of moment-based elements


| Name | Sobolev Space | Moment Formula | Applications |
|-------|---------------|----------------|--------------|
| Nédélec | $H(\mathrm{curl})$ | $\displaystyle \int_e (\mathbf{v}\cdot\mathbf{t})\,q\,ds,\quad q\in P_k(e)$ | Maxwell equations, electromagnetics, wave propagation |
| Raviart–Thomas | $H(\mathrm{div})$ | $\displaystyle \int_F (\mathbf{v}\cdot\mathbf{n})\,q\,dS,\quad q\in P_k(F)$ | Mixed Poisson, Darcy flow, flux-conservative problems |
| Brezzi–Douglas–Marini | $H(\mathrm{div})$ | $\displaystyle \int_F (\mathbf{v}\cdot\mathbf{n})\,q\,dS,\quad q\in P_k(F)$ | Mixed FEM, Darcy flow, incompressible flow, flux approximation |


### Are they actually useful?

Short answer--yes! Check out [this post](https://fenics.readthedocs.io/projects/dolfin/en/2017.2.0/demos/maxwell-eigenvalues/python/demo_maxwell-eigenvalues.py.html) from FEnICs, a Python FEM software. The authors simulated a Maxwell eigenvalue problem in two dimensions with a known analytical solution, and compared whether the estimated solution converged to the analytical solution as the number of elements increased. Performance was significantly better when using Nedelec elements.
