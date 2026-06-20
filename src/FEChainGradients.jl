"""
    get_predictions(re, θ, coordmat::AbstractMatrix, U::FESpace) -> Tuple{FEFunction, Function}

Evaluates the neural network at the given coordinates and returns the resulting 
`FEFunction` along with its Zygote pullback.
"""
function get_predictions(re, θ::AbstractVector, coordmat::AbstractMatrix, U::FESpace)
    zp, zpull = Zygote.pullback(th -> view(re(th)(coordmat), 1, :), θ)
    return FEFunction(U, Float64.(zp)), zpull
end

"""
    get_cell_residual(U::FESpace, resid_norm_pb::Function) -> AbstractArray

Propagates the gradient of the residual norm back to the cell degrees of freedom (DoFs).
"""
function get_cell_residual(U::FESpace, resid_norm_pb::Function)
    # unthunk is required to convert ChainRules lazy ZeroTangents into real arrays
    dl_dr_global = ChainRules.unthunk(resid_norm_pb(1.0)[2])
    dl_dr_fef = FEFunction(U, dl_dr_global)
    return get_cell_dof_values(dl_dr_fef)
end

"""
    dl_dfef(local_dldr_vals::AbstractArray, upredfunc::FEFunction, res_fefunc::Function, assem::SparseMatrixAssembler, U::FESpace, Ω::Triangulation) -> AbstractVector

Computes the derivative of the loss with respect to the finite element function DoFs 
by chaining the residual Jacobian with the local residual gradients.
"""
function dl_dfef(local_dldr_vals::AbstractArray, upredfunc::FEFunction, res_u_fefunc::Function,assem::SparseMatrixAssembler, U::FESpace, Ω::Triangulation)

    dr_du_jacobian_dc = Gridap.FESpaces.jacobian(res_u_fefunc, upredfunc)

    # Note: Accessing .dict.vals[1] relies on Gridap's internal dictionary structure for DomainContributions
    dldu_dc = cell_level_chain_rule((a,b) -> vec(a' * b), local_dldr_vals, dr_du_jacobian_dc.dict.vals[1],Ω)
    dldu_vec = assemble_vector(dldu_dc,assem,U)
    return dldu_vec
    
end

"""
    cell_level_chain_rule(mapfunc::Function, i::AbstractArray, j::AbstractArray, trian::Triangulation) -> DomainContribution

Applies a chain rule mapping function across cell-level arrays `i` and `j` using lazy evaluation,
returning the resulting `DomainContribution`.
"""
function cell_level_chain_rule(mapfunc::Function, i::AbstractArray,j::AbstractArray, trian::Triangulation)
    dldu_dc = DomainContribution()
    dldu_arr = lazy_map(mapfunc, i, j)
    add_contribution!(dldu_dc,trian,dldu_arr)
    return dldu_dc
end


"""
    get_error_loss(known_coords::AbstractArray, known_values::AbstractVector, dofmap::AbstractArray, upredfunc::FEFunction, zpb::Function) -> Tuple

Computes the L2 norm of the error between sensor data and network predictions, 
and calculates the gradient of this loss with respect to the network parameters.

# Arguments
- `known_coords`: Coordinates of the sensor locations.
- `known_values`: Ground truth values at the sensor locations.
- `dofmap`: Degree of freedom mapping for the cells.
- `upredfunc::FEFunction`: The finite element function representing the `u` predictions.
- `zpb::Function`: The Zygote pullback function for the neural network.

# Returns
- `Tuple`: The gradient vector `dl_de_dtheta` and the scalar `normloss`.
"""
function get_error_loss(known_coords::AbstractArray, known_values::AbstractVector, dofmap::AbstractArray, upredfunc::FEFunction, zpb::Function)
    
    interp_preds = upredfunc(known_coords)
    
    errorvec, errorpull = ChainRules.rrule(-, known_values, interp_preds)
    normloss, normlosspull = ChainRules.rrule(norm, errorvec)
    
    lbar = normlosspull(1.0)[2]
    ebar = errorpull(lbar)[3]

    n_free = length(upredfunc.free_values)
    global_sums = zeros(Float64, n_free)
    
    deriv_vals = lazy_map(y -> upredfunc.fe_space.space.fe_basis(y), known_coords)
    
    # 2. Iterate through your data
    for (i, cell_dofs) in enumerate(dofmap)
        
        for (j, dof) in enumerate(cell_dofs)
            if dof > 0
                global_sums[dof] += ebar[i]*deriv_vals[i][j]
            end
        end
    end
    dl_de_dtheta = zpb(global_sums)[1]

    return dl_de_dtheta, normloss
end