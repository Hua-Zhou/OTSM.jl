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

@testset "otsm_pba (port wine, MAXDIFF)" begin
A, maxdiff_optim, _ = portwine_data()
Smaxdiff = [A[i]'A[j] for i in 1:4, j in 1:4]
OTSM.verify_input_data!(Smaxdiff, 3, true) # set S[i, i] to 0
for r in 1:3
    for init_fun in [init_eye, init_sb, init_tb]
        @info "rank = $r, init = $init_fun"
        @time Ô_ba, ts_ba, = otsm_pba(Smaxdiff, r; 
            verbose = true, O = init_fun(Smaxdiff, r))
        @show ts_ba
        @test ts_ba ≈ maxdiff_optim[r]
        @show test_optimality(Ô_ba, Smaxdiff)
        # @test z1 == :global_optimal
        # @test z2 == :global_optimal
    end
end
# bm = @benchmark tsm_ba(S, 1)
# display(bm); println()
end

@testset "otsm_pba (port wine, MAXBET)" begin
A, _, maxbet_optim = portwine_data()
Smaxbet = [A[i]'A[j] for i in 1:4, j in 1:4]
for r in 1:3
    for init_fun in [init_eye, init_sb, init_tb, init_lww1]
        @info "rank = $r, init = $init_fun"
        @time Ô_ba, ts_ba, = otsm_pba(Smaxbet, r; 
            verbose = true, O = init_fun(Smaxbet, r))
        @show ts_ba
        if r == 3 && init_fun ∈ [init_eye, init_lww1]
            # r=3, init=init_eye or init_lww1 leads to an inferior local maximum
            @test ts_ba ≈ 818.7063749651282
        else
            @test ts_ba ≈ maxbet_optim[r]
        end
        # check global optimality certificate
        @show test_optimality(Ô_ba, Smaxbet)
    end
end
# bm = @benchmark tsm_ba(S, 1)
# display(bm); println()
end

@testset "SDP relaxation (port wine, MAXDIFF)" begin
A, maxdiff_optim, _ = portwine_data()
Smaxdiff = [A[i]'A[j] for i in 1:4, j in 1:4]
OTSM.verify_input_data!(Smaxdiff, 3, true) # set S[i, i] to 0
for r in 1:3
    for solver in [
        #Mosek.Optimizer(LOG=0),
        COSMO.Optimizer(max_iter=5000, verbose=false),
        SCS.Optimizer(verbose=0)
        ]
        @info "rank = $r, solver = $solver"
        @time Ô_sdp, ts_sdp, = otsm_sdp(Smaxdiff, r, solver = solver)
        @show ts_sdp
        @test ts_sdp ≈ maxdiff_optim[r]
        @show test_optimality(Ô_sdp, Smaxdiff)
    end
end
end

@testset "Procrustes (random data)" begin
Random.seed!(123)
n, d, m = 10, 3, 50
A, S, Ā, A_manopt = generate_procrustes_data(m, n, d)
@info "Proximal block ascent algorithm:"
for init_fun in [init_eye, init_sb, init_tb]
    @info "init = $init_fun"
    @time Ô_ba, ts_ba, = otsm_pba(S, d; verbose = true, O = init_fun(S, d))
    @show ts_ba
    @show test_optimality(Ô_ba, S)
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
    @show test_optimality(Ô_sdp, S)
end
end

@testset "MaxBet (Liu-Wang-Wang Example 5.1)" begin
A = [4.3299 2.3230 -1.3711 -0.0084 -0.7414;
2.3230 3.1181 1.0959 0.1285 0.0727;
-1.3711 1.0959 6.4920 -1.9883 -0.1878;
-0.0084 0.1285 -1.9883 2.4591 1.8463;
-0.7414 0.0727 -0.1878 1.8463 5.8875]
ns    = [2, 3]
m     = length(ns)
r     = 1
start = [1; cumsum(ns)[1:end-1] .+ 1]
stop  = cumsum(ns)
S     = [A[start[i]:stop[i], start[j]:stop[j]] for i in 1:m, j in 1:m]
@info "Proximal block ascent algorithm:"
for init_fun in [init_eye, init_sb, init_tb, init_lww1]
    @info "init = $init_fun"
    @time Ô_ba, ts_ba, = otsm_pba(S, r; verbose = true, O = init_fun(S, r))
    @show test_optimality(Ô_ba, S)
    if init_fun == init_eye
        # init_eye leads to a inferior local maximum
        @test abs(ts_ba - 14.10123) < 1e-3
    elseif init_fun == init_sb
        # the test matrix by Shapiro-Botha (init_sb) is singular with largest eigenvalue
        # 0 (with multiplicity 3). Different machines give different init_sb. Some of 
        # them lead to inferior local maximum.
        @test abs(ts_ba - 14.10123) < 1e-3 || abs(ts_ba - 14.73022) < 1e-3
    else
        @test abs(ts_ba - 14.73022) < 1e-3
    end
end
end

@testset "MaxBet (Liu-Wang-Wang Example 5.2)" begin
A = Float64.([45 -20 5 6 16 3;
-20 77 -20 -25 -8 -21;
5 -20 74 47 18 -32;
6 -25 47 54 7 -11;
16 -8 18 7 21 -7;
3 -21 -32 -11 -7 70])
ns    = [2, 2, 2]
m     = length(ns)
r     = 1
start = [1; cumsum(ns)[1:end-1] .+ 1]
stop  = cumsum(ns)
S     = [A[start[i]:stop[i], start[j]:stop[j]] for i in 1:m, j in 1:m]
@info "Proximal block ascent algorithm:"
for init_fun in [init_eye, init_sb, init_tb, init_lww1]
    @info "init = $init_fun"
    @time Ô_ba, ts_ba, = otsm_pba(S, r; verbose = true, O = init_fun(S, r))
    @test abs(ts_ba - 378.9624) < 1e-3
    @show test_optimality(Ô_ba, S)
end
end

end # PkgTest module
