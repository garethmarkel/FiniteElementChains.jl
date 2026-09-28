function get_moment_weights(V::FESpace,Ω::Triangulation)
    
    jac_vec = lazy_map(x-> x.args[1].value', V.fe_dof_basis.cell_dof)
    node_vec = lazy_map(x-> x.dofs.predofs.nodes, V.fe_dof_basis.cell_dof)
    map_vec = lazy_map(x-> x.dofs.values', V.fe_dof_basis.cell_dof)

    dim_mom = size(map_vec[1])[1]
    dim_p = length(node_vec[1])
    
    cmvec = get_cell_map(Ω)
    cell_indices = lazy_map(n -> ((n-1)*dim_p + 1):(n*dim_p),collect(1:length(node_vec)))
    BigB = create_B_matrix(V)
    BigBprodJ = lazy_map(x -> BigB .⋅ x,jac_vec)

    BigMprodBprodJ = collect(lazy_map((x,y) -> vector_value_mat_mul(x, y) ,map_vec,BigBprodJ))
    BigPos = lazy_map(x->[ifelse(i > 0, 1.0,0.0) for i in x],V.cell_dofs_ids)
    BigPBD = collect(lazy_map((x,y) -> x .* y,BigMprodBprodJ,BigPos))

    coords_vec = lazy_map((x,y)-> x(y), cmvec,node_vec)
    coords_vec=vcat(collect(coords_vec)...)

    return BigMprodBprodJ,BigPBD,coords_vec,cell_indices
    
end

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

function get_dof_value_vec(
        dofvals::Vector{Vector{T}}, 
        V::FESpace,
        dof_to_cell::Vector{Int64},
        dof_to_local::Vector{Int64}
    ) where {T}
    
    dofvec = zeros(T, V.nfree)
    
    @inbounds for i in 1:V.nfree
        dofvec[i] = dofvals[dof_to_cell[i]][dof_to_local[i]]
    end
    
    return dofvec
end

function get_predictions(
        re::T, 
        θ::AbstractVector, 
        coordmat::AbstractMatrix, 
        U::FESpace, 
        momentbasedfeinfo::MomentBasedElementTools{M, P, I, C, L, A}
        ) where {T, M, P, I, C, L, A}

    BigPBD = momentbasedfeinfo.dof_pullback_weights
    BigMBJ = momentbasedfeinfo.dof_prediction_weights
    cell_indices = momentbasedfeinfo.cell_indices
    pbvec = momentbasedfeinfo.pullbackmat
    dof_to_cell = momentbasedfeinfo.dof_to_cell
    dof_to_local = momentbasedfeinfo.dof_to_local
    
    zp, zpull = Zygote.pullback(th -> view(re(th)(coordmat), 1:2, :), θ)
    
    ghvec = reinterpret(VectorValue{2, eltype(zp)}, vec(zp))
    
    dofvals = collect(lazy_map((x,y) -> matvec_dot(x,@view ghvec[y]), BigMBJ, cell_indices))
    
    dofvec = get_dof_value_vec(dofvals, U.space, dof_to_cell, dof_to_local)
    
    return FEFunction(U, dofvec), zpull
end

function construct_pullback_tangent!(
            dldu_vec::AbstractVector, 
            momentbasedfeinfo::MomentBasedElementTools{M, P, I, C, L, A}
    ) where {M, P, I, C, L, A}

    BigPBD = momentbasedfeinfo.dof_pullback_weights
    BigMBJ = momentbasedfeinfo.dof_prediction_weights
    cell_indices = momentbasedfeinfo.cell_indices
    pbvec = momentbasedfeinfo.pullbackmat
    dof_to_cell = momentbasedfeinfo.dof_to_cell
    dof_to_local = momentbasedfeinfo.dof_to_local
    
    fill!(pbvec,0.0)
    @inbounds for i in 1:length(dof_to_cell)
        pbvec[:,cell_indices[dof_to_cell[i]]] .+= dldu_vec[i] .* reinterpret(Float64, BigMBJ[dof_to_cell[i]][dof_to_local[i],:]')
    end
end

# function get_moment_based_cell_residual(U::FESpace, dl_dr_global::T) where {T}
#     cell_to_dofs = get_cell_dof_ids(U)
    
#     # Return a generator instead of an allocated array
#     function safe_gather_generator(ids)
#         return (id > 0 ? dl_dr_global[id] : zero(eltype(dl_dr_global)) for id in ids)
#     end
    
#     return lazy_map(safe_gather_generator, cell_to_dofs)
# end

function get_moment_based_cell_residual(U::FESpace, dl_dr_global::T) where {T}
    cell_to_dofs = get_cell_dof_ids(U)
    
    # map cleanly allocates and fills a properly-typed array for each cell
    function safe_gather_array(ids)
        return map(id -> id > 0 ? dl_dr_global[id] : zero(eltype(dl_dr_global)), ids)
    end
    
    return lazy_map(safe_gather_array, cell_to_dofs)
end

