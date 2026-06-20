"""
    train_on_error!(iterations, residual_function::Function, nnsetup::NNSetup, pdesetup::PDESetup, sensordata::SensorData)

Trains the FEINN using the L2 norm of the data fitting error
"""
function train_on_error!(iterations, residual_function::Function, nnsetup::NNSetup, pdesetup::PDESetup, sensordata::SensorData)
    # --- STAGE 1: Data-Loss Only (Optimizing u parameters) ---
    function fg!(F, G, w)
        nnsetup.θ_u .= w
        dldthetau_e, errornorm = fe_error_loss(residual_function, nnsetup, pdesetup, sensordata)
        
        if G !== nothing
            G .= dldthetau_e
            return errornorm
        end
        if F !== nothing
            return errornorm
        end
    end
    res = Optim.optimize(Optim.only_fg!(fg!), deepcopy(nnsetup.θ_u),BFGS(), Optim.Options(iterations=iterations, store_trace=true))
    nnsetup.θ_u .= Optim.minimizer(res)
    return res
end

"""
    train_on_residual!(iterations, residual_function::Function, nnsetup::NNSetup, pdesetup::PDESetup, sensordata::SensorData)

Trains the FEINN using the l1 norm of the FEM residual error
"""
function train_on_residual!(iterations, residual_function::Function, nnsetup::NNSetup, pdesetup::PDESetup, sensordata::SensorData)
    # --- STAGE 1: Data-Loss Only (Optimizing u parameters) ---
    function fg!(F,G,w)
        nnsetup.θ_k .= w
        dldthetau_r, dldthetak_r, residnorm = fe_residual_loss(residual_function, nnsetup, pdesetup,sensordata)
        if !isnothing(G)
            copy!(G, dldthetak_r)
            return residnorm
        end
        if !isnothing(F)
            return residnorm
        end
    end
    
    res = Optim.optimize(Optim.only_fg!(fg!), deepcopy(nnsetup.θ_k),BFGS(), Optim.Options(iterations=iterations, store_trace=true))
    nnsetup.θ_k .= Optim.minimizer(res)

    return res
end

"""
    train_on_joint_loss!(iterations, residual_function::Function, nnsetup::NNSetup, pdesetup::PDESetup, sensordata::SensorData)

Trains the FEINN using the l2 norm of the data fitting error and the l1 norm of the FEM residual error, with said l1 error weighted by parameter alpha
"""
function train_on_joint_loss!(iterations, α, residual_function::Function, nnsetup::NNSetup, pdesetup::PDESetup, sensordata::SensorData)

    len_u = length(nnsetup.θ_u)
    
    function fg!(F,G,w)
        nnsetup.θ_u .= view(w,1:len_u)
        nnsetup.θ_k .= view(w,(len_u + 1):length(w))
        dldthetau_r, dldthetak_r, residnorm = fe_residual_loss(residual_function, nnsetup, pdesetup,sensordata)
        dldthetau_e, errornorm = fe_error_loss(residual_function, nnsetup, pdesetup,sensordata)
            
        if !isnothing(G)
            G[1:len_u] .= dldthetau_e .+ α .* dldthetau_r
            G[(len_u + 1):end] .= α .* dldthetak_r
            return α*residnorm + errornorm
        end
        if !isnothing(F)
            return α*residnorm + errornorm
        end
    end
    
    res = Optim.optimize(Optim.only_fg!(fg!), vcat(deepcopy(nnsetup.θ_u), deepcopy(nnsetup.θ_k)),BFGS(), Optim.Options(iterations=iterations, store_trace=true))
    best = Optim.minimizer(res)
    nnsetup.θ_u .= best[1:length(nnsetup.θ_u)]
    nnsetup.θ_k .= best[(length(nnsetup.θ_u) + 1):end]

    return res
end

"""
    train_feinn!(iter_pattern, joint_alphas, residual_function, nnsetup, pdesetup, sensordata)

Trains the FEINN. iter_pattern should be a length 3 vector with [iterations for data fitting, iterations for initial residual fitting, iteration for joint fitting stages].
joint_alphas should be a vector with all alphas desired for the joint data fitting step.
"""
function train_feinn!(iter_pattern, joint_alphas, residual_function, nnsetup, pdesetup, sensordata)

    print("Training on error...\n\n")

    train_on_error!(iter_pattern[1], residual_function, nnsetup, pdesetup, sensordata)

    print("Training on FEM residual...\n\n")

    train_on_residual!(iter_pattern[2], residual_function, nnsetup, pdesetup, sensordata)

    for α in joint_alphas
        print("Training on residual and error with alpha = $α ...\n\n")
        train_on_joint_loss!(iter_pattern[3], α, residual_function, nnsetup, pdesetup, sensordata)
    end

    print("Training complete.\n\n")

end