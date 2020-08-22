var documenterSearchIndex = {"docs":
[{"location":"#OTSM.jl","page":"OTSM.jl","title":"OTSM.jl","text":"","category":"section"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"OTSM.jl implements algorithms for solving the orthogonal trace sum maximization (OTSM) problem","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"operatornamemaximize sum_ij=1^m operatornametr (O_i^T S_ij O_j)","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"subject to orthogonality constraint O_i^T O_i = I_r. Here S_ij in mathbbR^d_i times d_j, 1 le i j le m, are data matrices. S_ii are symmetric and S_ij = S_ji^T. Many problems such as canonical correlation analysis (CCA) with m ge 2 data sets, Procrustes analysis with m ge 2 images, orthogonal least squares, and MaxBet are special cases of OSTM. ","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"Details on OTSM are described in paper: ","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"Joong-Ho Won, Hua Zhou, and Kenneth Lange. (2018) Orthogonal trace-sum maximization: applications, local algorithms, and global optimality, arXiv. ","category":"page"},{"location":"#Installation","page":"OTSM.jl","title":"Installation","text":"","category":"section"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"OTSM.jl requires Julia v1.0 or later. The package has not yet been registered and must be installed using the repository location. Start julia and use the ] key to switch to the package manager REPL","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"(@v1.4) pkg> add https://github.com/Hua-Zhou/OTSM.jl","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"Use the backspace key to return to the Julia REPL.","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"versioninfo()","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"Julia Version 1.4.2\nCommit 44fa15b150* (2020-05-23 18:35 UTC)\nPlatform Info:\n  OS: macOS (x86_64-apple-darwin18.7.0)\n  CPU: Intel(R) Core(TM) i7-6920HQ CPU @ 2.90GHz\n  WORD_SIZE: 64\n  LIBM: libopenlibm\n  LLVM: libLLVM-8.0.1 (ORCJIT, skylake)\nEnvironment:\n  JULIA_EDITOR = code\n  JULIA_NUM_THREADS = 4","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"# for use in this tutorial\nusing OTSM","category":"page"},{"location":"#Algorithms","page":"OTSM.jl","title":"Algorithms","text":"","category":"section"},{"location":"#Proximal-block-ascent-algorithm","page":"OTSM.jl","title":"Proximal block ascent algorithm","text":"","category":"section"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"The otsm_pba() function implements an efficient local search algorithm for solving OTSM.","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"For documentation of the otsm_pba() function, type ?otsm_bpa in Julia REPL.","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"otsm_pba","category":"page"},{"location":"#OTSM.otsm_pba","page":"OTSM.jl","title":"OTSM.otsm_pba","text":"otsm_pba(S, r)\n\nMaximize the trace sum sum_{i,j} trace(Oi' * S[i, j] * Oj) = trace(O' * S * O)  subject to  orthogonality constraint Oi' * Oi == I(r) by a  proximal block ascent algorithm. Each of S[i, j] for i < j is a di x dj matrix, S[i, i] are symmetric, and S[i, j] = S[j, i]'. Note  otsm_pba is mutating in the sense the keyword argument O=O_init is updated  with the final solution.\n\nPositional arguments\n\nS       :: Matrix{Matrix}: S[i, j] is a di x dj matrix, S[i, i] are\n\nsymmetric, and S[j, i] = S[i, j]'.\n\nr       :: Integer       : rank of solution.\n\nKeyword arguments\n\nαinv    :: Number: proximal update constant 1α, default is1e-3`.\nmaxiter :: Integer: maximum number of iterations, default is 50000.\ntolfun  :: Number: tolerance for objective convergence, default is 1e-10.\ntolvar  :: Number: tolerance for iterate convergence, default is 1e-8.\nverbose :: Bool  : verbose display, default is false.\nO       :: Vector{Matrix}: starting point, default is init_tb(S, r).\nlog     :: Bool: record iterate history or not, defaut is false.\n\nOutput\n\nO       : result, O[i] has dimension di x r.\ntracesum: objective value evaluated at final O.\nobj     : final objective value from PBA algorithm, should be same as tracesum.\nhistory : iterate history.\n\n\n\n\n\n","category":"function"},{"location":"#Semidefinite-programming-(SDP)-relaxation","page":"OTSM.jl","title":"Semidefinite programming (SDP) relaxation","text":"","category":"section"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"The otsm_sdp() function implements an SDP relaxation approach for solving OTSM.","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"For documentation of the otsm_sdp() function, type ?otsm_sdp in Julia REPL.","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"otsm_sdp","category":"page"},{"location":"#OTSM.otsm_sdp","page":"OTSM.jl","title":"OTSM.otsm_sdp","text":"otsm_sdp(S, r)\n\nMaximize the trace sum sum_{i,j} trace(Oi' * S[i, j] * Oj) subject to the orthogonality constraint Oi'Oi = I(r) using an SDP relaxation (P-SDP in  the manuscript). Each of S[i, j] for i < j is a di x dj matrix and  S[i, j] = S[j, i]'.\n\nOutput\n\nO       : solution.\ntracesum: trace sum at solution.\nobj     : SDP relaxation objective value at solution.\nisexaxt : true means global optimality.\n\n\n\n\n\n","category":"function"},{"location":"#Start-point","page":"OTSM.jl","title":"Start point","text":"","category":"section"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"Different strategies for starting point are implemented.","category":"page"},{"location":"#Initialize-O_i-by-I_r","page":"OTSM.jl","title":"Initialize O_i by I_r","text":"","category":"section"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"init_eye","category":"page"},{"location":"#OTSM.init_eye","page":"OTSM.jl","title":"OTSM.init_eye","text":"init_eye(S)\n\nInitialize O[i] by an di x r diagonal matrix with ones on diagonal.\n\n\n\n\n\n","category":"function"},{"location":"#Initialize-O_i-by-a-strategy-by-Ten-Berge-(default)","page":"OTSM.jl","title":"Initialize O_i by a strategy by Ten Berge (default)","text":"","category":"section"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"This is the default for the proximal block ascent algorithm otsm_pba.","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"init_tb","category":"page"},{"location":"#OTSM.init_tb","page":"OTSM.jl","title":"OTSM.init_tb","text":"init_tb(S)\n\nCompute initial point following Ten Berge's second upper bound  (https://doi.org/10.1007/BF02294053, p. 273). This is same as the  Liu-Wang-Wang starting point strategy 2 (https://doi.org/10.1137/15M100883X,  Algorithm 3.2, p. 1495). Take the eigenvectors corresponding to the r largest  eigenvalues of S. This is a D x r orthogonal matrix, D=sum_i=1^m d_i. For each di x r block, project to the Stiefel manifold. These blocks constitute an initial point.\n\n\n\n\n\n","category":"function"},{"location":"#Initialize-O_i-by-a-strategy-by-Liu-Wang-Wang","page":"OTSM.jl","title":"Initialize O_i by a strategy by Liu-Wang-Wang","text":"","category":"section"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"init_lww1","category":"page"},{"location":"#OTSM.init_lww1","page":"OTSM.jl","title":"OTSM.init_lww1","text":"init_lww1(S, r)\n\nCompute initial point following Liu-Wang-Wang starting point strategy 1  (https://doi.org/10.1137/15M100883X, Algorithm 3.1, p. 1494). Set O[1] to  the top r eigenvectors of S[1, 1]. Then Ok = Uk * Qk, where Uk is the  top r eigenvectors of S[k, k] and Qk is the Q factor in the QR decomposition  of Uk * sum_{j<k} S[k, j] * O[j].\n\n\n\n\n\n","category":"function"},{"location":"#Initialize-O_i-by-a-strategy-by-Shapiro-Botha-and-Won-Zhou-Lange","page":"OTSM.jl","title":"Initialize O_i by a strategy by Shapiro-Botha and Won-Zhou-Lange","text":"","category":"section"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"init_sb","category":"page"},{"location":"#OTSM.init_sb","page":"OTSM.jl","title":"OTSM.init_sb","text":"init_sb(S, r)\n\nCompute initial point by Shapiro-Botha (https://doi.org/10.1137/0609032, p. 380).  Also see the extension by Won-Zhou-Lange paper (Lemma B.1). Replace diagonal  blocks S[i, i] by S_ii - sum_j P_ij * D_ij * P_ij, where  P_ij * D_ij * Q_ij is the SVD of S_ij. The resulting matrix is  negative semidefinite. Take the eigenvectors corresponding to the r largest  eigenvalues. This is a D x r orthogonal matrix, D=sum_i=1^m d_i. For each di x r block, project to the Stiefel manifold. These blocks constitute an initial point.\n\n\n\n\n\n","category":"function"},{"location":"#Example-data-Port-Wine","page":"OTSM.jl","title":"Example data - Port Wine","text":"","category":"section"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"The package contains the port wine example data set from the Hanafi and Kiers (2006) paper. It can be retrieved by the portwine_data() function.","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"A, _, _ = portwine_data();","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"Data matrices A1, A2, A3, A4 record the ratings (centered at 0) of m=4 accessors on 8 port wines in d_1=4, d_2=3, d_3=4, and d_4=3 aspects respectively. ","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"for i in 1:4\n    display(A[i])\nend","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"8×4 Array{Float64,2}:\n  1.25  -5.0  -1.375   3.875\n -0.75   1.0  -0.375  -1.125\n  1.25  -3.0  -1.375   0.875\n -0.75   2.0   0.625  -0.125\n -0.75   2.0  -0.375  -0.125\n  0.25   3.0  -0.375  -3.125\n -0.75  -1.0   3.625  -1.125\n  0.25   1.0  -0.375   0.875\n\n\n\n8×3 Array{Float64,2}:\n  2.0  -4.375   0.625\n  1.0   1.625   0.625\n  1.0  -1.375   2.625\n -1.0   1.625  -1.375\n  0.0   0.625   0.625\n -1.0   0.625  -0.375\n -2.0  -0.375  -2.375\n  0.0   1.625  -0.375\n\n\n\n8×4 Array{Float64,2}:\n  3.125   3.0  -2.5   0.75\n -1.875  -1.0   1.5   0.75\n  2.125   2.0  -0.5   1.75\n -1.875  -1.0   1.5  -1.25\n  1.125   0.0   0.5   0.75\n -0.875  -1.0   0.5  -0.25\n -1.875  -1.0  -0.5  -2.25\n  0.125  -1.0  -0.5  -0.25\n\n\n\n8×3 Array{Float64,2}:\n  1.0   0.125   0.375\n  0.0  -0.875  -1.625\n  2.0  -0.875  -1.625\n -1.0   0.125  -0.625\n  0.0   0.125  -0.625\n  0.0   1.125   1.375\n -2.0  -1.875   1.375\n  0.0   2.125   1.375","category":"page"},{"location":"#MAXDIFF","page":"OTSM.jl","title":"MAXDIFF","text":"","category":"section"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"The MAXDIFF approach for CCA seeks the rotations of A_i that achieve the maximal agreement","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"operatornamemaximize 2 sum_i  j operatornametr (O_i^T A_i^T A_j O_j)","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"subject to constraint O_i^T O_i = I_r. This corresponds to an OTSM problem with S_ij = A_i^T A_j and S_ii = 0.","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"Smaxdiff = [A[i]'A[j] for i in 1:4, j in 1:4]\nfor i in 1:4\n    fill!(Smaxdiff[i, i], 0)\nend\ndisplay.(Smaxdiff);","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"4×4 Array{Float64,2}:\n 0.0  0.0  0.0  0.0\n 0.0  0.0  0.0  0.0\n 0.0  0.0  0.0  0.0\n 0.0  0.0  0.0  0.0\n\n\n\n3×4 Array{Float64,2}:\n  5.0   -15.0  -12.0     13.0\n -9.25   36.0    5.875  -20.375\n  5.75  -11.0  -14.125    7.625\n\n\n\n4×4 Array{Float64,2}:\n  9.75  -26.0  -14.625   21.125\n  8.0   -27.0  -10.0     18.0\n -6.0    21.0    2.5    -13.5\n  4.5    -8.0  -12.75     6.75\n\n\n\n3×4 Array{Float64,2}:\n 6.0   -11.0  -12.0     8.0\n 1.75    9.0   -6.625   1.125\n 0.25    3.0    6.125  -2.625\n\n\n\n4×3 Array{Float64,2}:\n   5.0   -9.25     5.75\n -15.0   36.0    -11.0\n -12.0    5.875  -14.125\n  13.0  -20.375    7.625\n\n\n\n3×3 Array{Float64,2}:\n 0.0  0.0  0.0\n 0.0  0.0  0.0\n 0.0  0.0  0.0\n\n\n\n4×3 Array{Float64,2}:\n 13.0  -21.625  14.375\n 11.0  -21.0    11.0\n -5.0   16.5    -2.5\n 10.0   -5.75   13.25\n\n\n\n3×3 Array{Float64,2}:\n  9.0  -8.0    12.0\n  1.0   4.375   0.375\n -6.0  -0.875  -8.875\n\n\n\n4×4 Array{Float64,2}:\n   9.75     8.0   -6.0    4.5\n -26.0    -27.0   21.0   -8.0\n -14.625  -10.0    2.5  -12.75\n  21.125   18.0  -13.5    6.75\n\n\n\n3×4 Array{Float64,2}:\n  13.0     11.0  -5.0  10.0\n -21.625  -21.0  16.5  -5.75\n  14.375   11.0  -2.5  13.25\n\n\n\n4×4 Array{Float64,2}:\n 0.0  0.0  0.0  0.0\n 0.0  0.0  0.0  0.0\n 0.0  0.0  0.0  0.0\n 0.0  0.0  0.0  0.0\n\n\n\n3×4 Array{Float64,2}:\n 13.0    10.0  -4.0  10.0\n  2.875  -2.0  -0.5   1.25\n -2.375  -4.0  -4.5  -7.25\n\n\n\n4×3 Array{Float64,2}:\n   6.0   1.75    0.25\n -11.0   9.0     3.0\n -12.0  -6.625   6.125\n   8.0   1.125  -2.625\n\n\n\n3×3 Array{Float64,2}:\n  9.0  1.0    -6.0\n -8.0  4.375  -0.875\n 12.0  0.375  -8.875\n\n\n\n4×3 Array{Float64,2}:\n 13.0   2.875  -2.375\n 10.0  -2.0    -4.0\n -4.0  -0.5    -4.5\n 10.0   1.25   -7.25\n\n\n\n3×3 Array{Float64,2}:\n 0.0  0.0  0.0\n 0.0  0.0  0.0\n 0.0  0.0  0.0","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"Proximal block ascent algorithm for finding a rank r=2 solution to MAXDIFF.","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"Ô_pba, ts_pba, obj, history = otsm_pba(Smaxdiff, 2; verbose = true);","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"iter = 1, obj = 539.8501989834106\niter = 2, obj = 542.2346791607897\niter = 3, obj = 542.326755374587\niter = 4, obj = 542.3275270111226\niter = 5, obj = 542.327550329459\niter = 6, obj = 542.3275506362339\niter = 7, obj = 542.3275506383457\niter = 8, obj = 542.3275506383522","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"The test_optimality() function attempts to certify whether a local solution O::Vector{Matrix} is a global solution. The first output indicates the solution is :infeasible, :suboptimal, :stationary_point, or :global_optimal.","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"# proximal block ascent yields the global solution\ntest_optimality(Ô_pba, Smaxdiff)[1]","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":":global_optimal","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"For documentation of the test_optimality() function, type ?test_optimality in Julia REPL.","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"test_optimality","category":"page"},{"location":"#OTSM.test_optimality","page":"OTSM.jl","title":"OTSM.test_optimality","text":"test_optimality(O::Vector{Matrix}, S::Matrix{Matrix}; tol=1e-3, method=:wzl)\n\nTest if the vector of orthogonal matrices O is globally optimal. The O is  assumed to satisfy the first-order optimality conditions. Each of O[i] is a  di x r matrix. Each of S[i, j] for i < j is a di x dj matrix.  Tolerance tol is used to test the positivity of the smallest eigenvalue of the test matrix.\n\nPositional arguments\n\nO :: Vector{Matrix}: A point satisifying O[i]'O[i] = I(r).\nS :: Matrix{Matrix}: Data matrix.\n\nKeyword arguments\n\ntol     :: Number: Tolerance for testing psd of the test matrix.  \nmethod  :: Symbol: :wzl (Theorem 3.1 of Won-Zhou-Lange https://arxiv.org/abs/1811.03521)    or :lww (Theorem 2.4 of Liu-Wang-Wang https://doi.org/10.1137/15M100883X)\n\nOutput\n\nstatus :: Symbol: A certificate that \n:infeasible: orthogonality constraints violated\n:suboptimal: failed the first-order optimality (stationarity) condition  \n:stationary_point: satisfied first-order optimality; may or may not be global optimal\n:nonglobal_stationary_point: satisfied first-order optimality; must not be global optimal\n:global_optimal: certified global optimality\nΛ      :: Vector{Matrix}: Lagrange multipliers.\nC      :: Matrix{Matrix}: Certificate matrix corresponding to method.\nλmin   :: Number: The minimum eigenvalue of the certificate matrix.\n\n\n\n\n\n","category":"function"},{"location":"#MAXBET","page":"OTSM.jl","title":"MAXBET","text":"","category":"section"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"The MAXBET approach for CCA seeks the rotations of A_i that achieve the maximal agreement","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"operatornamemaximize sum_ij operatornametr (O_i^T A_i^T A_j O_j)","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"subject to constraint O_i^T O_i = I_r. This corresponds to an OTSM problem with S_ij = A_i^T A_j.","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"Smaxbet = [A[i]'A[j] for i in 1:4, j in 1:4]\ndisplay.(Smaxbet);","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"4×4 Array{Float64,2}:\n   5.5   -12.0  -6.25     7.25\n -12.0    54.0   6.0    -31.0\n  -6.25    6.0  17.875   -9.375\n   7.25  -31.0  -9.375   28.875\n\n\n\n3×4 Array{Float64,2}:\n  5.0   -15.0  -12.0     13.0\n -9.25   36.0    5.875  -20.375\n  5.75  -11.0  -14.125    7.625\n\n\n\n4×4 Array{Float64,2}:\n  9.75  -26.0  -14.625   21.125\n  8.0   -27.0  -10.0     18.0\n -6.0    21.0    2.5    -13.5\n  4.5    -8.0  -12.75     6.75\n\n\n\n3×4 Array{Float64,2}:\n 6.0   -11.0  -12.0     8.0\n 1.75    9.0   -6.625   1.125\n 0.25    3.0    6.125  -2.625\n\n\n\n4×3 Array{Float64,2}:\n   5.0   -9.25     5.75\n -15.0   36.0    -11.0\n -12.0    5.875  -14.125\n  13.0  -20.375    7.625\n\n\n\n3×3 Array{Float64,2}:\n  12.0  -10.0    11.0\n -10.0   29.875  -7.125\n  11.0   -7.125  15.875\n\n\n\n4×3 Array{Float64,2}:\n 13.0  -21.625  14.375\n 11.0  -21.0    11.0\n -5.0   16.5    -2.5\n 10.0   -5.75   13.25\n\n\n\n3×3 Array{Float64,2}:\n  9.0  -8.0    12.0\n  1.0   4.375   0.375\n -6.0  -0.875  -8.875\n\n\n\n4×4 Array{Float64,2}:\n   9.75     8.0   -6.0    4.5\n -26.0    -27.0   21.0   -8.0\n -14.625  -10.0    2.5  -12.75\n  21.125   18.0  -13.5    6.75\n\n\n\n3×4 Array{Float64,2}:\n  13.0     11.0  -5.0  10.0\n -21.625  -21.0  16.5  -5.75\n  14.375   11.0  -2.5  13.25\n\n\n\n4×4 Array{Float64,2}:\n  26.875   20.0  -13.5  12.25\n  20.0     18.0  -11.0   9.0\n -13.5    -11.0   12.0  -2.0\n  12.25     9.0   -2.0  11.5\n\n\n\n3×4 Array{Float64,2}:\n 13.0    10.0  -4.0  10.0\n  2.875  -2.0  -0.5   1.25\n -2.375  -4.0  -4.5  -7.25\n\n\n\n4×3 Array{Float64,2}:\n   6.0   1.75    0.25\n -11.0   9.0     3.0\n -12.0  -6.625   6.125\n   8.0   1.125  -2.625\n\n\n\n3×3 Array{Float64,2}:\n  9.0  1.0    -6.0\n -8.0  4.375  -0.875\n 12.0  0.375  -8.875\n\n\n\n4×3 Array{Float64,2}:\n 13.0   2.875  -2.375\n 10.0  -2.0    -4.0\n -4.0  -0.5    -4.5\n 10.0   1.25   -7.25\n\n\n\n3×3 Array{Float64,2}:\n 10.0   2.0    -5.0\n  2.0  10.875   4.625\n -5.0   4.625  11.875","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"Proximal block ascent algorithm for finding a rank r=2 solution to MAXBET.","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"Ô_pba, ts_pba, obj, history = otsm_pba(Smaxbet, 2; verbose = true);","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"iter = 1, obj = 769.5257682063867\niter = 2, obj = 778.9896186367976\niter = 3, obj = 779.6705236544665\niter = 4, obj = 779.7414861674778\niter = 5, obj = 779.7533232474\niter = 6, obj = 779.756174278359\niter = 7, obj = 779.7569547275209\niter = 8, obj = 779.7571745803215\niter = 9, obj = 779.7572368336034\niter = 10, obj = 779.7572544704886\niter = 11, obj = 779.7572594658141\niter = 12, obj = 779.7572608802018\niter = 13, obj = 779.757261280583\niter = 14, obj = 779.757261393906\niter = 15, obj = 779.7572614259773\niter = 16, obj = 779.7572614350538\niter = 17, obj = 779.7572614376222\niter = 18, obj = 779.7572614383489\niter = 19, obj = 779.7572614385546\niter = 20, obj = 779.7572614386127\niter = 21, obj = 779.7572614386298\niter = 22, obj = 779.7572614386339\niter = 23, obj = 779.7572614386349\niter = 24, obj = 779.7572614386353\niter = 25, obj = 779.7572614386356","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"This solution is certified to be global optimal.","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":"# proximal block ascent yields the global solution\ntest_optimality(Ô_pba, Smaxbet)[1]","category":"page"},{"location":"","page":"OTSM.jl","title":"OTSM.jl","text":":global_optimal","category":"page"}]
}
