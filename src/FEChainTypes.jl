"""
    NNSetup{S, A, B}

Contains the neural network restructurers and parameters used for the FEINN.
"""
Base.@kwdef struct NNSetup{S, A, B}
    re_u::S          
    re_k::A       
    θ_u::B     
    θ_k::B     
end

"""
    NNSetup{S, A, B}

Contains the information on what loss to use
"""
Base.@kwdef struct LossSetup{T,M,N}
    loss_target::T  
    M_gram_factorized::M
    l_x_norm::N  
end

"""
    LossSetup(loss_target::T, l_x_norm::N) where {T,N}

Returns a LossSetup object in cases that don't use a riesz projection
"""
function LossSetup(loss_target::T, l_x_norm::N) where {T,N}
    return LossSetup(loss_target, 1.0,l_x_norm)
end

"""
    PDESetup{S, D, A, C}

Contains the finite element spaces, assemblers, and coordinate matrices 
required for the residual calculation.
"""
Base.@kwdef struct PDESetup{S, D, A, C}
    U_u::S          # FE space for u
    U_kap::D        # FE space for kappa
    assem_u::A      # Assembler for u
    assem_k::A      # Assembler for kappa
    coords_u::C     # Coordinate matrix for u
    coords_k::C     # Coordinate matrix for kappa
end

"""
    SensorData{V, C, M, I, B}

Holds the experimental/sensor data used for the loss function.
"""
Base.@kwdef struct SensorData{V, C, M, I, B}
    values::V
    coords::C
    dofmap::M
    idmap::I
    basismap::B
end


"""
    SensorData(space::FESpace, values::V, coords::C) where {V,C}

Function to construct a SensorData object from the finite element space, and known values + measurement locations.
"""
function SensorData(space::FESpace, values::V, coords::C) where {V,C}
    
    cell_ids_field = CellField(collect(1:num_cells(space.fe_basis.trian)), space.fe_basis.trian)
    dofmap = collect(lazy_map(x -> space.cell_dofs_ids[cell_ids_field(x)], coords))
    idmap = collect(lazy_map(x -> cell_ids_field(x), coords))
    basis_map = collect(lazy_map((y) -> reinterpret(Float64,space.fe_basis(y)'), coords))

    return SensorData(values, coords, dofmap, idmap, basis_map)
end

"""
    MomentBasedElementTools{M, P, I, C, L, A}

Contains the information required to reconstruct moment based elements from a neural network's predictions.
"""
Base.@kwdef struct MomentBasedElementTools{M, P, I, C, L, A}
    dof_prediction_weights::M
    dof_pullback_weights::P
    cell_indices::I
    dof_to_cell::C        
    dof_to_local::L
    pullbackmat::A
end

"""
    MomentBasedElementTools(V::FESpace, Ω::Triangulation)

Function to construct the Nedelec reconstruction object from the FESpace info.
"""
function MomentBasedElementTools(V::FESpace, Ω::Triangulation)
    BigMBJ,BigPBD,coords_vec,cell_indices = get_moment_weights(V,Ω)
    dof_to_cell,dof_to_local = create_dof_to_cell_map(V)
    pbvec = Matrix{Float64}(undef,2,length(coords_vec))

    coords_vec = [get_array(i) for i in coords_vec]

    return MomentBasedElementTools(BigMBJ,BigPBD,cell_indices,dof_to_cell,dof_to_local,pbvec), coords_vec
end