
"""
    get_coord_mat(crd::AbstractArray) -> Matrix{Float64}

Converts a vector of coordinate arrays into a dense 2xN coordinate matrix.

# Arguments
- `coords::AbstractVector`: A vector where each element is an array or tuple representing `(x, y)` coordinates.

# Returns
- `Matrix{Float64}`: A 2-row matrix containing the horizontally stacked coordinates.

# Examples
```julia-repl
julia> coords = [[1.0, 2.0], [3.0, 4.0]];

julia> get_coord_mat(coords)
2×2 Matrix{Float64}:
 1.0  3.0
 2.0  4.0
"""
function get_coord_mat(crd::AbstractArray)
    
    coord_mat = Matrix{Float64}(undef, 2, length(crd))
    
    @inbounds for i in eachindex(crd)
        coord_mat[:,i] = crd[i]
    end

    return coord_mat
end


"""
    assemble_vector(dc::DomainContribution, assem::SparseMatrixAssembler, V::FESpace) -> AbstractVector

Allocates and assembles a global vector from a `DomainContribution` over a given finite element space.
"""
function assemble_vector(dc::DomainContribution, assem::SparseMatrixAssembler, V::FESpace)
    rs = collect_cell_vector(V, dc)
    vec = allocate_vector(assem, rs)
    assemble_vector!(vec, assem, rs)
    return vec
end

"""
    get_cell_ids_field(Ω::Triangulation) -> CellField

Gets the node ids for each cell 
"""
function get_cell_ids_field(Ω)
    return CellField(collect(1:num_cells(Ω)), Ω)
end


"""
    get_dof_map(U::FESpace,cell_ids_field::CellField,known_coords::AbstractArray)

Gets the cell ids for each coordinates
"""
function get_dof_map(U::FESpace,cell_ids_field::CellField,known_coords::AbstractArray)
    return lazy_map(x -> U.space.cell_dofs_ids[cell_ids_field(x)], known_coords)
end


function vector_value_mat_mul(
        A::TA,
        X::AbstractMatrix{T},
    ) where {TA, T}

    m, n = size(A)
    _, p = size(X)

    Y = fill(zero(T), m, p)

    @inbounds for j in 1:p
        for k in 1:n
            xkj = X[k,j]
            for i in 1:m
                Y[i,j] += A[i,k] * xkj
            end
        end
    end

    return Y
end

function matvec_dot(
        A::TA, 
        x::TB
    ) where {TA,TB}
    
    y = zeros(eltype(A[1,1] ⋅ x[1]), size(A,1))
    @inbounds for i in axes(A,1)
        s = zero(eltype(y))
        for j in axes(A,2)
            s += A[i,j] ⋅ x[j]
        end
        y[i] = s
    end
    y
end

function create_dof_to_cell_map(V::FESpace)
    cell_dofs = get_cell_dof_ids(V)

    n_dofs = num_free_dofs(V)
    dof_to_cell = zeros(Int, n_dofs)
    dof_to_local = zeros(Int, n_dofs)
    
    for (cell_id, dofs) in enumerate(cell_dofs)
        for (local_id, dof) in enumerate(dofs)
            if dof > 0
                if dof_to_cell[dof] == 0
                    dof_to_cell[dof] = cell_id
                    dof_to_local[dof] = local_id
                end
            end
        end
    end

    return dof_to_cell,dof_to_local
end

function evaluate(loss::LossSetup{T,M,N},resid_vec::J) where {T,M,N,J}
    loss.loss_target(resid_vec, loss.M_gram_factorized,loss.l_x_norm)
end

function riesz_transformed_loss(resid_vec::R, M_gram_fact::M, l_x_norm::G) where {R,M,G}
    riesz = M_gram_fact \ resid_vec
    resid_norm, resid_norm_pb = ChainRules.rrule(norm, riesz, l_x_norm)
    
    dlDR = ChainRules.unthunk(resid_norm_pb(1.0)[2])

    dlDR = M_gram_fact \ dlDR

    return resid_norm, dlDR
end

function raw_residual_loss(resid_vec::R, M_gram_fact::M, l_x_norm::G) where {R,M,G}
    resid_norm, resid_norm_pb = ChainRules.rrule(norm, resid_vec, l_x_norm)
    dlDR = ChainRules.unthunk(resid_norm_pb(1.0)[2])
    return dlDR, resid_norm
end
