
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


function get_cell_ids_field(Ω)
    return CellField(collect(1:num_cells(Ω)), Ω)
end

function get_dof_map(U::FESpace,cell_ids_field::CellField,known_coords::AbstractArray)
    return lazy_map(x -> U.space.cell_dofs_ids[cell_ids_field(x)], known_coords)
end


