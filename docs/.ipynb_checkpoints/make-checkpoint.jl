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
         "first_steps.md",
         ],
    "API" => "api.md"
     ],
    remotes = nothing,
 )