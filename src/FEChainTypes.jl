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
    SensorData{V, C, M}

Holds the experimental/sensor data used for the loss function.
"""
Base.@kwdef struct SensorData{V, C, M}
    values::V
    coords::C
    dofmap::M
end
