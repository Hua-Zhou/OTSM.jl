using Statistics

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
    ts_optim = [419.65133740484237, 542.3275506383522, 568.19290485317]
    A, S, ts_optim
end
