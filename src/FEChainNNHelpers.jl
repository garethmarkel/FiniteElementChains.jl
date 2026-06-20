"""
    setup_nn(model_u,model_kap) -> NNSetup

Destructures neural networks for u and kappa
"""
function setup_nn(model_u,model_kap)
    θ_u, re_u = destructure(model_u)
    θ_k, re_k = destructure(model_kap)

    return NNSetup(re_u,re_k,θ_u,θ_k)
end

"""
    initialize_networks(input_size::Int,output_size::Int,nlayers::Int,neurons::Int,inner_activation::Function,k_activation::Function) -> NNSetup

Initializes stock neural networks for u and kappa, with input size = spatial dim, output size = output dim of your FEM problem, nlayers layers, `neurons` neurons per hidden layer, inner_activation activation function, and k_activation for your k network.
"""
function initialize_networks(input_size::Int,output_size::Int,nlayers::Int,neurons::Int,inner_activation::Function,k_activation::Function)
    layers = []

    # input → first hidden
    push!(layers, Dense(input_size, neurons, inner_activation))

    # add n-1 hidden layers
    for i in 2:nlayers
        push!(layers, Dense(neurons, neurons, inner_activation))
    end

    k_layers = deepcopy(layers)
    
    push!(layers, Dense(neurons, output_size))
    push!(k_layers, Dense(neurons, output_size,k_activation))
    
    model_u = Chain(
        layers...
    )
    model_kap = Chain(
        k_layers...
    )

    return setup_nn(model_u,model_kap)
end

"""
    nitialize_networks(input_size::Int,output_size::Int,nlayers::Int,neurons::Int) -> NNSetup

Initializes stock neural networks for u and kappa, with input size = spatial dim, output size = output dim of your FEM problem, nlayers layers, `neurons` neurons per hidden layer, a softplus activation function for hidden layers, and a rectabs output function for the kappa network.
"""
function initialize_networks(input_size::Int,output_size::Int,nlayers::Int,neurons::Int)
    return initialize_networks(input_size,output_size,nlayers,neurons,softplus,rect_abs)
end

"""
    initialize_networks() -> NNSetup

Initializes stock neural networks for u and kappa, with input size = 2 , output size = 2, 2 layers, 20 neurons per hidden layer, a softplus activation function for hidden layers, and a rectabs output function for the kappa network.
"""
function initialize_networks()
    return initialize_networks(2,1,2,20,softplus,rect_abs)
end


