function siren_initial_layer(x)
    sin(10*x)
end
function rect_abs(x)
    abs(x) + 0.01
end

struct FourierFeatures{T}
    B::T
end

function FourierFeatures(in_dim::Int, nfeat::Int; σ=10f0)
    B = σ .* randn(Float32, nfeat, in_dim)
    FourierFeatures(B)
end

function (ff::FourierFeatures)(x)
    proj = 2f0*pi .* (ff.B * x)
    vcat(sin.(proj), cos.(proj))
end

Flux.@layer FourierFeatures

init_siren(infeat,outf) = rand(Float32,infeat,outf) .* 2.0*sqrt(6/infeat) .+ -1.0.*sqrt(6/infeat)
init_siren_init(infeat,outf) = rand(Float32,infeat,outf) .* 2.0/infeat .+ -1.0/infeat
