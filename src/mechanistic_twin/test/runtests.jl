using Test
using MechanisticTwin

@testset "MechanisticTwin.jl" begin
    include("test_aggregation.jl")
    include("test_neuron_death.jl")
    include("test_synthetic.jl")
end
