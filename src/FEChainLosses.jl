"""
    fe_residual_loss(residual_function::Function, nnsetup::NNSetup, pdesetup::PDESetup,sensordata::SensorData) -> Tuple

Orchestrates the forward pass and automatic differentiation of the PDE residual using positional tuples.

# Arguments
- `residual_function::Function`: The Gridap function defining the physics residual.
- `nnsetup::NNSetup`: Struct containing neural network parameters `(θ_u, θ_k)` and architectures `(re_u, re_k)`
- `pdesetup::PDESetup`: Struct containing finite element spaces `(U, U_kap)`, finite element assemblers `(assem, assem_k)`, coordinate matrices `(U_u_coords, U_kap_coords)`.
- `sensordata::SensorData`: Struct containing sensor data arrays `(known_values, known_coords, dofmap)`.

# Returns
- `Tuple`: The gradients `dldtheta_u`, `dldtheta_k`, and the scalar `resid_norm`.
"""
function fe_residual_loss(residual_function::Function, nnsetup::NNSetup, pdesetup::PDESetup,sensordata::SensorData)

    # unpack parameters
    θ_u = nnsetup.θ_u
    θ_k = nnsetup.θ_k
    re_u=nnsetup.re_u
    re_k=nnsetup.re_k
    
    U_u_coords = pdesetup.coords_u 
    U_kap_coords = pdesetup.coords_k
    U = pdesetup.U_u
    U_kap = pdesetup.U_kap
    assem = pdesetup.assem_u
    assem_k = pdesetup.assem_k
    known_values = sensordata.values
    known_coords = sensordata.coords
    dofmap = sensordata.dofmap
    
    # get ancillary params 
    Ω = pdesetup.U_u.space.fe_basis.trian

    # get predictions
    upredfunc, zpbu = get_predictions(re_u,θ_u, U_u_coords,U)
    kpredfunc, zpbk = get_predictions(re_k,θ_k, U_kap_coords,U_kap)

    # create functions to get jacobian of 
    res_u_fefunc(uvals) = residual_function(kpredfunc,uvals)
    res_k_fefunc(uvals) = residual_function(uvals,upredfunc)
    
    dc_resid = residual_function(kpredfunc, upredfunc)
    
    resid_vec = assemble_vector(dc_resid,assem,U)
    
    resid_norm, resid_norm_pb = ChainRules.rrule(norm, resid_vec,1)
    
    local_dldr_vals = get_cell_residual(U,resid_norm_pb)
    
    dldu_vec = dl_dfef(local_dldr_vals, upredfunc, res_u_fefunc,assem, U,Ω)
    dldk_vec = dl_dfef(local_dldr_vals, kpredfunc, res_k_fefunc,assem_k, U_kap,Ω)
    
    dldtheta_u = zpbu(dldu_vec)[1]
    dldtheta_k = zpbk(dldk_vec)[1]
    
    return dldtheta_u,dldtheta_k, resid_norm
end


"""
    fe_error_loss(residual_function::Function, nnsetup::NNSetup, pdesetup::PDESetup,sensordata::SensorData) -> Tuple

Evaluates the sensor error loss and computes its gradient with respect to the `u` network parameters using positional tuples.

# Arguments
- `residual_function::Function`: The Gridap function defining the physics residual.
- `nnsetup::NNSetup`: Struct containing neural network parameters `(θ_u, θ_k)` and architectures `(re_u, re_k)`
- `pdesetup::PDESetup`: Struct containing finite element spaces `(U, U_kap)`, finite element assemblers `(assem, assem_k)`, coordinate matrices `(U_u_coords, U_kap_coords)`.
- `sensordata::SensorData`: Struct containing sensor data arrays `(known_values, known_coords, dofmap)`.

# Returns
- `Tuple`: The gradients `dldtheta_u`, `dldtheta_k`, and the scalar `resid_norm`.
"""

function fe_error_loss(residual_function::Function, nnsetup::NNSetup, pdesetup::PDESetup,sensordata::SensorData)

    # unpack parameters
    θ_u = nnsetup.θ_u
    θ_k = nnsetup.θ_k
    re_u=nnsetup.re_u
    re_k=nnsetup.re_k
    
    U_u_coords = pdesetup.coords_u 
    U_kap_coords = pdesetup.coords_k
    U = pdesetup.U_u
    U_kap = pdesetup.U_kap
    assem = pdesetup.assem_u
    assem_k = pdesetup.assem_k
    known_values = sensordata.values
    known_coords = sensordata.coords
    dofmap = sensordata.dofmap
    
    # get ancillary params 
    Ω = pdesetup.U_u.space.fe_basis.trian

    # get predictions
    upredfunc, zpbu = get_predictions(re_u,θ_u, U_u_coords,U)
    
    dl_de_dtheta, errornormloss = get_error_loss(known_coords,known_values,dofmap,upredfunc, zpbu)

    return dl_de_dtheta, errornormloss
end