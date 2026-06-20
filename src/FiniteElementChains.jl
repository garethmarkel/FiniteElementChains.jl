module FiniteElementChains

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


include("FEChainTypes.jl")
include("FEChainUtils.jl")
include("FEChainGradients.jl")
include("FEChainLosses.jl")
include("FEChainTraining.jl")
include("FEChainNNStructure.jl")
include("FEChainNNHelpers.jl")

export NNSetup, PDESetup, SensorData,
       train_feinn!, train_on_error!, train_on_residual!, train_on_joint_loss!,
       initialize_networks, get_coord_mat, get_cell_ids_field, get_dof_map
       get_predictions

end