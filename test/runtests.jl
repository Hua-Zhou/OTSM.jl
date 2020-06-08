module PkgTest

using BenchmarkTools, Statistics, Test, TraceSumMaximization


function portwine_data()
    A1 = [
    7 5 7 5 5 6 5 6;
    0 6 2 7 7 8 4 6;
    5 6 5 7 6 6 10 6;
    8 3 5 4 4 1 3 5
    ]
    A2 = [
    4 3 3 1 2 1 0 2;
    0 6 3 6 5 5 4 6;
    5 5 7 3 5 4 2 4
    ]
    A3 = [
    7 2 6 2 5 3 2 4;
    4 0 3 0 1 0 0 0;
    2 6 4 6 5 5 4 4;
    6 6 7 4 6 5 3 5
    ]
    A4 = [
    9 8 10 7 8 8 6 8;
    7 6 6 7 7 8 5 9;
    9 7 7 8 8 10 10 10
    ]
    m = 4
    A = [A1', A2', A3', A4']
    A = map(M -> M .- mean(M, dims=1), A)
    S = [A[i]'A[j] for i in 1:m, j in 1:m]
    for i in 1:m
        fill!(S[i, i], 0)
    end
    tsm_optim = [419.6513374038389, 542.3275506362339, 568.1929048530802]
    S, tsm_optim
end

@testset "cat" begin
A = Matrix{Matrix{Int}}(undef, 2, 2)
A[1, 1] = [1 1 1; 1 1 1]
A[1, 2] = [2 2; 2 2]
A[2, 1] = [3 3 3]
A[2, 2] = [3 3]
@test cat(A) == [1 1 1 2 2; 1 1 1 2 2; 3 3 3 3 3]
end

@testset "tsm_ba" begin

S, tsm_optim = portwine_data()
for r in 1:3
    for init_fun in [init_eye, init_sb, init_tb]        
        @info "rank = $r, init = $init_fun"
        @time Ô_ba, obj_ba, = tsm_ba(S, r; verbose = true, O = init_fun(S, r))
        @test obj_ba ≈ tsm_optim[r]
        @test testoptimality(Ô_ba, S)[1] == 1
    end
end

# bm = @benchmark tsm_ba(S, 1)
# display(bm); println()
end

end # PkgTest module
