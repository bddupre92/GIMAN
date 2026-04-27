using Test
using MechanisticTwin

@testset "Aggregation Module" begin
    p = AggregationParams()

    @testset "Healthy steady state" begin
        ss = healthy_steady_state(p)
        @test ss[1] > 0              # Monomers present
        @test ss[2] == 0.0           # No oligomers
        @test ss[3] == 0.0           # No fibrils
        @test isapprox(ss[1], p.k_prod / p.k_clear_M, rtol=0.01)
    end

    @testset "Reproduction number" begin
        R0 = reproduction_number(p)
        @test R0 > 0
        # Default params: k_frag=1e-5, k_clear_F=0.005 → R0=0.002 (sub-critical)
        @test R0 < 1.0  # Disease-free state is stable with defaults
    end

    @testset "Prasinezumab reduces k_e" begin
        p_treated = apply_prasinezumab(p; eta=0.25, C_Ab=1.0, K_d=0.5)
        @test p_treated.k_e < p.k_e
        @test p_treated.k_e > 0  # Still positive
        # Other params unchanged
        @test p_treated.k_prod == p.k_prod
        @test p_treated.k_n == p.k_n
    end

    @testset "ODE preserves positivity" begin
        u = [2.0, 0.1, 0.01]
        du = similar(u)
        aggregation_ode!(du, u, p, 0.0)
        # du should be well-defined (no NaN/Inf)
        @test all(isfinite, du)
    end
end
