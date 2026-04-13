using Test
using MechanisticTwin
using DifferentialEquations

@testset "Coupled System Synthetic Validation" begin
    p = CoupledParams()
    u0 = default_initial_state(p)
    u0[3] = 0.01  # Seed fibrils

    @testset "Initial state is valid" begin
        @test all(u0 .>= 0)
        @test u0[1] > 0   # Monomers present
        @test u0[4] == p.nd.N_0  # Full neuron count
    end

    @testset "Short integration (1 year)" begin
        t_span = (0.0, years_to_hours(1.0))
        prob = ODEProblem(coupled_system_ode!, u0, t_span, p)
        sol = solve(prob, Tsit5(); reltol=1e-6, abstol=1e-8)

        # Solution should complete
        @test sol.retcode == ReturnCode.Success

        # All states remain non-negative
        for i in 1:length(u0)
            @test all(sol[i, :] .>= -1e-10)  # Allow tiny numerical noise
        end

        # Neurons should decline (even slightly) over 1 year
        @test sol[4, end] <= u0[4]
    end
end
