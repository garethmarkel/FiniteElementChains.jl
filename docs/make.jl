using Documenter, FiniteElementChains

makedocs(
    format = Documenter.HTML(),
    modules = [FiniteElementChains],
    sitename = "FiniteElementChains.jl",
    authors = "Gareth Markel",
    linkcheck = false,
    pages = [
    "Home" => "index.md",
    "What are FEINNs?" => "feinn_introduction.md",
    "Tutorials" => [
         "First steps" => "first_steps.md",
         "Loss functions" => "loss_functions.md",
         "Working with moment based elements" => "moment_based_elements.md",
         "Electromagnetics & FEINNs" => "nedelec_elements.md"
         ],
    "API" => "api.md",
    "Citations" => "citations.md"
     ],
    remotes = nothing,
 )

deploydocs(
    repo = "github.com/garethmarkel/FiniteElementChains.jl.git",
    devbranch = "main",
)