module PkgTest

using BenchmarkTools, Random, Statistics, Test, OTSM
using COSMO, SCS#, MosekTools

@testset "cat" begin
A = Matrix{Matrix{Int}}(undef, 2, 2)
A[1, 1] = [1 1 1; 1 1 1]
A[1, 2] = [2 2; 2 2]
A[2, 1] = [3 3 3]
A[2, 2] = [3 3]
@test cat(A) == [1 1 1 2 2; 1 1 1 2 2; 3 3 3 3 3]
end

@testset "otsm_ba" begin
A, S, ts_optim = portwine_data()
for r in 1:3
    for init_fun in [init_eye, init_sb, init_tb]        
        @info "rank = $r, init = $init_fun"
        @time Ô_ba, ts_ba, = otsm_pba(S, r; verbose = true, O = init_fun(S, r))
        @show ts_ba
        @test ts_ba ≈ ts_optim[r]
        @test test_optimality(Ô_ba, S)[1] == 1
    end
end
# bm = @benchmark tsm_ba(S, 1)
# display(bm); println()
end

@testset "SDP relaxation" begin
A, S, ts_optim = portwine_data()
r = 1
for r in 1:3
    for solver in [
        #Mosek.Optimizer(LOG=0),
        COSMO.Optimizer(max_iter=5000, verbose=false),
        SCS.Optimizer(verbose=0)
        ]
        @info "rank = $r, solver = $solver"
        @time Ô_sdp, ts_sdp, = otsm_sdp(S, r, solver = solver)
        @show ts_sdp
        @test ts_sdp ≈ ts_optim[r]
        @test test_optimality(Ô_sdp, S)[1] == 1
    end
end
end

@testset "Procrustes" begin
Random.seed!(123)
n, d, m = 10, 3, 50
A, S, Ā, A_manopt = generate_procrustes_data(m, n, d)
@info "Proximal block ascent algorithm:"
for init_fun in [init_eye, init_sb, init_tb]        
    @info "init = $init_fun"
    @time Ô_ba, ts_ba, = otsm_pba(S, d; verbose = true, O = init_fun(S, d))
    @show ts_ba
    @test test_optimality(Ô_ba, S)[1] == 1
end
@info "SDP relaxation (failed to find global solution):"
for solver in [
    #Mosek.Optimizer(LOG=0),
    COSMO.Optimizer(max_iter=5000, verbose=false),
    SCS.Optimizer(verbose=0)
    ]
    @info "solver = $solver"
    @time Ô_sdp, ts_sdp, = otsm_sdp(S, d, solver = solver)
    @show ts_sdp
    @test test_optimality(Ô_sdp, S)[1] == -1
end
end

end # PkgTest module
