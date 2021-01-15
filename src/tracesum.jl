using Convex, IterativeSolvers, LinearAlgebra, SCS
import Base: cat
import LinearAlgebra: BlasReal

"""
	cat(A::Matrix{Matrix{T}})

Concatenate an array of matrices into a single matrix. Similar to `cell2mat` in
Matlab.
"""
function Base.cat(A::VecOrMat{Matrix{T}}) where T
    m   = map(M -> size(M, 1), A[:, 1]) # row counts
    n   = map(M -> size(M, 2), A[1, :]) # column counts
    out = zeros(T, sum(m), sum(n))
    @inbounds for j in 1:length(n)
        jr = (sum(n[1:j-1]) + 1):sum(n[1:j])
        for i in 1:length(m)
            ir = (sum(m[1:i-1]) + 1):sum(m[1:i])
            out[ir, jr] = A[i, j]
        end
    end
    out
end

"""
	verify_input_data!(S, r, make_Sii_zero = true)

`S` is an array of matrices. Enforce `S[j, i] = S[i, j]'`, `i < j`, (copy 
upper triangular blocks `S[i, j]`, `i < j`, to lower triangular blocks) and
check `r` is less than `size(S[i, j], 1)`. If `make_Sii_zero = true`, fill 
diagonal blocsk `S[i, i]` by zeros; this is useful in certain applications.
"""
function verify_input_data!(
    S             :: AbstractMatrix{<:AbstractMatrix},
    r             :: Integer,
    make_Sii_zero :: Bool = true
    )
    @assert size(S, 1) == size(S, 2) "size(S, 1) should be equal to size(S, 2)"
    m = size(S, 1)
    @inbounds for i in 1:m
        @assert r ≤ size(S[i, i], 1) "rank r > size(S[$i, $i], 1) !"
        make_Sii_zero == true && fill!(S[i, i], 0)
        for j in (i + 1):m
            S[j, i] .= transpose(S[i, j])
        end
    end
    S, r
end

"""
	identify!(O::Vector{Matrix})

Fix identifiability of rectangular orthogonal matrices `O[1],..., O[end]`, where
`O[i]'O[i] = I(r)`, by reducing `O[end]` into a lower triangular matrix with
positive diagonal entries. When `O[end]` is a square matrix, it will be
reduced to the identity matrix.
"""
function identify!(O::Array{Matrix{T}}) where T
    m   = length(O)
    tmp = [similar(O[i]) for i in 1:m]
    if size(O[end], 1) == size(O[end], 2) # O[end] is orthogonal matrix
        @inbounds for i in 1:(length(O) - 1)
            copyto!(tmp[i], O[i])
            mul!(O[i], tmp[i], transpose(O[end]))
        end
        fill!(O[end], 0)
        @inbounds for i in 1:size(O[end], 1)
            O[end][i, i] = 1
        end
    elseif size(O[end], 1) > size(O[end], 2) # O[end] is rectangular
        L, Q = lq(O[end])      # LQ decomposition of Q[end]
        ds = [L[i, i] ≥ 0 ? 1 : -1 for i in 1:size(O[end], 2)]
        rmul!(L, Diagonal(ds)) # make diagonal entries of L to be positive
        @inbounds for i in 1:(m - 1)
            mul!(O[i], copyto!(tmp[i], O[i]), transpose(Q))
            rmul!(O[i], Diagonal(ds))
        end
        copyto!(O[end], L)
    else
        error("O[end] should have more rows than columns")
    end
    O
end

"""
    otsm_pba(S, r)

Maximize the trace sum `sum_{i,j} trace(Oi' * S[i, j] * Oj) = trace(O' * S * O)` 
subject to  orthogonality constraint `Oi' * Oi == I(r)` by a 
proximal block ascent algorithm. Each of `S[i, j]` for `i < j` is a
`di x dj` matrix, `S[i, i]` are symmetric, and `S[i, j] = S[j, i]'`. Note 
`otsm_pba` is mutating in the sense the keyword argument `O=O_init` is updated 
with the final solution.

# Positional arguments
- `S       :: Matrix{Matrix}`: `S[i, j]` is a `di x dj` matrix, `S[i, i]` are
symmetric, and `S[j, i] = S[i, j]'`.
- `r       :: Integer`       : rank of solution.

# Keyword arguments
- `αinv      :: Number`: proximal update constant ``1/α```, default is `1e-3`.
- `maxiter   :: Integer`: maximum number of iterations, default is `50000`.
- `tolfun    :: Number`: tolerance for objective convergence, default is `1e-10`.
- `tolvar    :: Number`: tolerance for iterate convergence, default is `1e-8`.
- `verbose   :: Bool`  : verbose display, default is `false`.
- `O         :: Vector{Matrix}`: starting point, default is `init_tb(S, r)`.
- `log       :: Bool`: record iterate history or not, defaut is `false`.
- `soconstr` :: Bool`: constrain solution `Oi` to be in `SO(d)`, ie, `det(Oi)=1`.

# Output
- `O`       : result, `O[i]` has dimension `di x r`.
- `tracesum`: objective value evaluated at final `O`.
- `obj`     : final objective value from PBA algorithm, should be same as `tracesum`.
- `history` : iterate history.
"""
function otsm_pba(
    S        :: Matrix{Matrix{T}},
    r        :: Integer;
    αinv     :: Number  = 1e-3,
    maxiter  :: Integer = 50000,
    tolfun   :: Number  = 1e-10,
    tolvar   :: Number  = 1e-8,
    verbose  :: Bool    = false,
    log      :: Bool    = false,
    O        :: Vector{Matrix{T}} = init_tb(S, r),
    soconstr :: Bool    = false
    ) where T <: BlasReal
    m = size(S, 1)
    soconstr && any(i -> size(S[i, i], 1) ≠ r, 1:m) && 
    error("when socontr=true, it must be true di=r for all i")
    # record iterate history if requested
    history          = ConvergenceHistory(partial = !log)
    history[:tolfun] = tolfun
    history[:tolvar] = tolvar
    IterativeSolvers.reserve!(T, history, :tracesum, maxiter)
    IterativeSolvers.reserve!(T, history, :vchange , maxiter)
    IterativeSolvers.reserve!(Float64, history, :itertime, maxiter)
    # initial objective value
    tic     = time()
    SO      = S * O # SO[i] = sum_{j} S_{ij} O_j
    obj::T  = dot(O, SO)
    toc     = time()
    IterativeSolvers.nextiter!(history)
    push!(history, :tracesum, obj)
    push!(history, :itertime, toc - tic)
    # pre-allocate intermediate arrays
    ΔO  = [similar(O[i]) for i in 1:m]
    tmp = [similar(O[i]) for i in 1:m]
    Λi  = Matrix{T}(undef, r, r) # Lagrange multipliers
    for iter in 1:maxiter-1
        IterativeSolvers.nextiter!(history)
        verbose && println("iter = $iter, obj = $obj")
        # block update
        tic = time()
        @inbounds for i in 1:m
            # update Oi
            # tmp[i] = SO[i] + αinv * O[i]
            BLAS.axpy!(αinv, O[i], copyto!(tmp[i], SO[i]))
            Fi = svd!(tmp[i]; full = false)
            copyto!(ΔO[i], O[i])
            mul!(O[i], Fi.U, Fi.Vt)
			if soconstr && det(O[i]) < 0
				# modification to projection onto SO(r)
				# O[i] <- O[i] - 2 * Fi.U[:, end] * Fi.Vt[end, :]
				@views BLAS.ger!(T(-2), Fi.U[:, r], Fi.Vt[r, :], O[i])
			end            
            ΔO[i] .= O[i] .- ΔO[i]
            # update SO[j] for j > i
            for j in (i+1):m
                BLAS.gemm!('N', 'N', T(1), S[j, i], ΔO[i], T(1), SO[j])
            end
        end
        objold  = obj
        obj     = dot(O, mul!(SO, S, O))
        toc     = time()
        vchange = sum(norm, ΔO) / m # mean Frobenius norm of variable change        
        push!(history, :tracesum,       obj)
        push!(history, :vchange ,   vchange)
        push!(history, :itertime, toc - tic)
        (vchange < tolvar) && 
        (abs(obj - objold) < tolfun * abs(objold + 1)) &&
        IterativeSolvers.setconv(history, true) &&
        break
    end
    # fix identifiability and compute final trace sum objective
    identify!(O)
    tracesum::T = dot(O, mul!(SO, S, O))
    log && IterativeSolvers.shrink!(history)
    O, tracesum, obj, history
end

"""
    otsm_sdp(S, r)

Maximize the trace sum `sum_{i,j} trace(Oi' * S[i, j] * Oj)` subject to the
orthogonality constraint `Oi'Oi = I(r)` using an SDP relaxation (P-SDP in 
the manuscript). Each of `S[i, j]` for `i < j` is a `di x dj` matrix and 
`S[i, j] = S[j, i]'`.

# Output
- `O`       : solution.
- `tracesum`: trace sum at solution.
- `obj`     : SDP relaxation objective value at solution.
- `isexaxt` : `true` means global optimality.
"""
function otsm_sdp(
    S       :: Matrix{Matrix{T}},
    r       :: Integer;
    verbose :: Bool = false,
    O       :: Union{Nothing, Vector{Matrix{T}}} = nothing,
    solver  = SCS.Optimizer()
    ) where T <: BlasReal
    m = size(S, 1)
    d = [size(S[i, i], 1) for i in 1:m] # (d[i], d[j]) = size(S[i, j])
    S̃ = cat(S) # flattened matrix
    # form and solve the SDP
    U       = Convex.Semidefinite(sum(d), sum(d))
    problem = Convex.maximize((m / 2) * tr(S̃ * U))
    for i in 1:m
        ir = (sum(d[1:i-1]) + 1):sum(d[1:i])
        #problem.constraints += U[ir, ir] ⪯ I(d[i]) / m
        problem.constraints += eigmax(U[ir, ir]) <= inv(m)
        problem.constraints += tr(U[ir, ir]) == r / m
    end
    if O === nothing
        Convex.solve!(problem, () -> solver, verbose = verbose)
    else
        E   = cat(O)
        E .*= √m 
        mul!(U.value, E, transpose(E))
        Convex.solve!(problem, () -> solver, verbose = verbose, warmstart = true)
    end
    obj = problem.optval
    # retrieve solution to trace maximization problem
    # quick return if NAN in SDP solution
    if any(isnan.(U.value))
        O = [fill(T(NaN), d[i], r) for i in 1:m]
        return O, obj, T(NaN), false
    end
    Uchol = cholesky!(Symmetric(U.value), Val(true); # Cholesky with pivoting
    check = false, tol = 1e-3maximum(diag(U.value)))
    if rank(Uchol) ≤ r
        isexact = true
        for i in 1:m
            ir      = (sum(d[1:i-1]) + 1):sum(d[1:i])
            Uii     = Symmetric(U.value[ir, ir])
            Uiichol = cholesky(Uii, Val(true); 
            check = false, tol = 1e-3maximum(diag(Uii)))
            if rank(Uiichol) > r
                isexact = false; break
            end
        end
    else
        isexact = false
    end
    if verbose
        if isexact
            println("SDP solution solves the trace sum max problem exactly")
        else
            println("SDP solution solves the trace sum max problem approximately")
        end
    end
    E = Uchol.L[invperm(Uchol.piv), 1:r]
    O = [Matrix{T}(undef, d[i], r) for i in 1:m]
    # impove accuracy by SVD projection
    for i in 1:m
        ir = (sum(d[1:i-1]) + 1):sum(d[1:i])
        Eisvd = svd!(E[ir, :]; full = false)
        mul!(O[i], Eisvd.U, Eisvd.Vt)
    end
    # fix identifiability and compute final trace sum objective
    identify!(O)
    O, dot(O, S * O), obj, isexact
end

"""
	test_optimality(O::Vector{Matrix}, S::Matrix{Matrix}; tol=1e-3, method=:wzl)

Test if the vector of orthogonal matrices `O` is globally optimal. The `O` is 
assumed to satisfy the first-order optimality conditions. Each of `O[i]` is a 
`di x r` matrix. Each of `S[i, j]` for `i < j` is a `di x dj` matrix. 
Tolerance `tol` is used to test the positivity of the smallest eigenvalue
of the test matrix.

# Positional arguments
- `O :: Vector{Matrix}`: A point satisifying `O[i]'O[i] = I(r)`.
- `S :: Matrix{Matrix}`: Data matrix.

# Keyword arguments
- `tol     :: Number`: Tolerance for testing psd of the test matrix.  
- `method  :: Symbol`: `:wzl` (Theorem 3.1 of Won-Zhou-Lange <https://arxiv.org/abs/1811.03521>) 
    or `:lww` (Theorem 2.4 of Liu-Wang-Wang <https://doi.org/10.1137/15M100883X>)

# Output
- `status :: Symbol`: A certificate that 
    - `:infeasible`: orthogonality constraints violated
    - `:suboptimal`: failed the first-order optimality (stationarity) condition  
    - `:stationary_point`: satisfied first-order optimality; may or may not be global optimal
    - `:nonglobal_stationary_point`: satisfied first-order optimality; must not be global optimal
    - `:global_optimal`: certified global optimality
- `Λ      :: Vector{Matrix}`: Lagrange multipliers.
- `C      :: Matrix{Matrix}`: Certificate matrix corresponding to `method`.
- `λmin   :: Number`: The minimum eigenvalue of the certificate matrix.
"""
function test_optimality(
    O      :: Vector{Matrix{T}},
    S      :: Matrix{Matrix{T}};
    tol    :: Number = 1e-3,
    method :: Symbol = :wzl
    ) where T <: BlasReal
    r  = size(O[1], 2)
    m  = size(S, 1)
    # SO[i] = sum_{j} S_{ij} O_j
    SO = S * O 	
    # Langrange multiplier
    Λ  = [transpose(O[i]) * SO[i] for i in 1:m]
    storage_rr = Matrix{T}(undef, r, r)
    # check constraint satisfication and first order optimality
    @inbounds for i in 1:m
        # check constraint satisfication O[i]'O[i] = I(r)
        mul!(storage_rr, transpose(O[i]), O[i])
        storage_rr ≈ I || (return :infeasible, Λ, Matrix{Matrix{T}}(undef, m, m), T(NaN))
        # check symmetry of Λi (first order optimality condition)
        δi = check_symmetry(Λ[i])
        if δi > abs(tol)
            @warn "Λ$i not symmetric; norm(Λ - Λ') = $δi; " *
                "first order optimality not satisfied"
            return :suboptimal, Λ, Matrix{Matrix{T}}(undef, m, m), T(NaN)
        end
    end
    # certify global optimality
    if method == :wzl
        # Won-Zhou-Lange certificate matrix (Theorem 3.1)
        C  = copy(-S)
        @inbounds for i in 1:m
            ni = size(O[i], 1)
            # if eigmin(Λi) < 0, then it cannot be global optimal
            λmin = minimum(eigvals!(Symmetric(copyto!(storage_rr, Λ[i]))))
            if λmin < -abs(tol)
                return :nonglobal_stationary_point, Λ, Matrix{T}(undef, 0, 0), T(NaN)
            end
            # update certificate matrix by Won-Zhou-Lange
            copyto!(storage_rr, Λ[i])
            for j in 1:r
                storage_rr[j, j] -= λmin
            end
            C[i, i] .+= O[i] * Symmetric(storage_rr) * transpose(O[i])
            for k in 1:ni
                C[i, i][k, k] += λmin
            end
        end
        λmin = eigmin(Symmetric(cat(C)))
        if λmin > -abs(tol)
            status = :global_optimal
        elseif m == 2 && r == 1 
            # Won-Zhou-Lange certificate is necessary for m=2, r=1
            status = :nonglobal_stationary_point
        elseif m == 2 && all([S[i, i] ≈ 0I for i in 1:m])
            # Won-Zhou-Lange certificate is necessary for m=2, MAXDIFF
            status = :nonglobal_stationary_point
        else
            status = :stationary_point
        end
        return status, Λ, C, λmin
    elseif method == :lww
        # certificate by Liu-Wang-Wang 2015 (Theorem 2.4)
        C  = [-kron(S[i, j], Matrix(I, r, r)) for i in 1:m, j in 1:m]
        @inbounds for i in 1:m
            ni = size(O[i], 1)
            C[i, i] += kron(Matrix(I, ni, ni), Λ[i])
        end
        λmin = eigmin(Symmetric(cat(C)))
        if λmin > -abs(tol)
            status = :global_optimal
        elseif m == 2 && r == 1
            # Won-Zhou-Lange certificate is necessary for m=2, r=1
            status = :nonglobal_stationary_point
        else
            status = :stationary_point
        end
        return status, Λ, C, λmin
    else
        error("unrecognized certificate method $method; should be `:wzl` or `:lww`")
    end
end

"""
    check_symmetry(A)

Return `norm(A - A')`. If the return value is close to 0, `A` is nearly symmetric. 
"""
function check_symmetry(A::AbstractMatrix)
    @assert size(A, 1) == size(A, 2) "A is not square"
    δ = zero(eltype(A))
    @inbounds for j in 2:size(A, 2), i in 1:j-1
        δ += abs2(A[i, j] - A[j, i])
    end
    sqrt(δ)
end

# """
#     mts_evu(S, r)

# Maximize the trace sum `2 * sum_{i<j} trace(Oi' * S[i, j] * Oj)` subject to
# orthogonality constraint `Oi' * Oi == eye(r)` using the unconstrained
# eigenvalue minimization formulation. Each of `S[i, j]` for `i < j` is a
# `di x dj` matrix.
# """
# function mts_evu(
#     S::Matrix{Matrix{T}},
#     r::Integer;
#     verbose::Bool = false
#     ) where T <: Union{Float32, Float64}

#     m = size(S, 1)
#     d = [size(S[i, i], 1) for i in 1:m] # (d[i], d[j]) = size(S[i, j])
#     # check (1) Sji = Sij, (2) Sii = 0
#     verify_input_data!(S, r)

#     # form and solve the SDP
#     H = Convex.Variable(sum(d), sum(d))
#     obj = m * sumlargesteigs(H, r)
#     for i in 1:m
#         ir = (sum(d[1:i-1]) + 1):sum(d[1:i])
#         obj += sumlargesteigs(- H[ir, ir], r)
#     end
#     problem = Convex.minimize(obj)
#     for j in 1:m
#         jr = (sum(d[1:j-1]) + 1):sum(d[1:j])
#         for i in 1:(j - 1)
#             ir = (sum(d[1:i-1]) + 1):sum(d[1:i])
#             problem.constraints += H[ir, jr] == S[i, j]
#         end
#     end
#     solve!(problem)
#     obj = problem.optval

#     # retrieve solution to trace maximization problem
#     Heig = eigfact(Symmetric(H.value))
#     #@show Heig.values
#     # check eigenvalue gaps
#     if abs(Heig[:values][end - r + 1] - Heig[:values][end - r]) > 1e-3Heig[:values][end]
#         isexact = true
#         for i in 1:m
#             ir = (sum(d[1:i-1]) + 1):sum(d[1:i])
#             Xiev = eigvals(Symmetric(H.value[ir, ir]))
#             if r < d[i] && abs(Xiev[r + 1] - Xiev[r]) < 1e-3Xiev[end]
#                 isexact = false; break
#             end
#             if r == d[i] && abs(Xiev[r]) < 1e-3Xiev[end]
#                 isexact = false; break
#             end
#             # Xieig = eigfact(Symmetric(H.value[ir, ir]), 1:r)
#             # @show Xieig[:vectors]
#         end
#     else
#         isexact = false
#     end
#     if verbose
#         if isexact
#             info("SDP solution solves the trace sum max problem exactly")
#         else
#             info("SDP solution solves the trace sum max problem approximately")
#         end
#     end
#     O = [zeros(T, d[i], r) for i in 1:m]
#     # columns corresponding to largest r eigenvalues
#     jr = (sum(d) - r + 1):sum(d)
#     for i in 1:m
#         ir = (sum(d[1:i-1]) + 1):sum(d[1:i])
#         Ei = Heig.vectors[ir, jr]
#         Eisvd = svdfact!(Ei; thin = true)
#         A_mul_B!(O[i], Eisvd.U, Eisvd.Vt)
#         # O[i] = √m * E[ir, :]
#         # @show O[i]'O[i]
#         # Xieig = eigfact(Symmetric(H.value[ir, ir]))
#         # @show Xieig[:values]
#         # @show vecnorm(E[ir, :]'Xieig[:vectors][:, r+1:end])
#         # @show vecnorm(E[ir, :] * E[ir, :]' - Xieig[:vectors][:, 1:r] * Xieig[:vectors][:, 1:r]' / m)
#     end
#     # fix identifiability and compute final trace sum objective
#     identify!(O)
#     O, obj, dot(O, S * O), isexact
# end

# """
#     mts_evc(S, r)

# Maximize the trace sum `sum_{i<j} trace(Oi' * S[i, j] * Oj)` subject to
# orthogonality constraint `Oi' * Oi == eye(r)` using Shapiro-Botha SDP relaxation.
# Each of `S[i, j]` for `i < j` is a `d x d` matrix.
# """
# function mts_evc(
#     S::Matrix{Matrix{T}},
#     r::Integer;
#     verbose::Bool = false
#     ) where T <: Union{Float32, Float64}

#     m = size(S, 1)
#     d = [size(S[i, i], 1) for i in 1:m] # (d[i], d[j]) = size(S[i, j])
#     # check (1) Sji = Sij, (2) Sii = 0
#     verify_input_data!(S, r)

#     # form and solve the SDP
#     H = Convex.Semidefinite(sum(d), sum(d))
#     obj = 0
#     for i in 1:m
#         ir = (sum(d[1:i-1]) + 1):sum(d[1:i])
#         obj += sumlargesteigs(H[ir, ir], r)
#     end
#     problem = Convex.minimize(obj)
#     for j in 1:m
#         jr = (sum(d[1:j-1]) + 1):sum(d[1:j])
#         for i in 1:(j - 1)
#             ir = (sum(d[1:i-1]) + 1):sum(d[1:i])
#             problem.constraints += H[ir, jr] == - S[i, j]
#         end
#     end
#     solve!(problem)
#     obj = problem.optval

#     # retrieve solution to trace maximization problem
#     Heig = eigfact(Symmetric(- H.value))
#     #@show Heig.values
#     # check eigenvalue gaps
#     if abs(Heig[:values][end - r + 1] - Heig[:values][end - r]) > 1e-3Heig[:values][end]
#         isexact = true
#         for i in 1:m
#             ir = (sum(d[1:i-1]) + 1):sum(d[1:i])
#             Xiev = eigvals(- Symmetric(H.value[ir, ir]))
#             if r < d[i] && abs(Xiev[r + 1] - Xiev[r]) < 1e-3Xiev[end]
#                 isexact = false; break
#             end
#             if r == d[i] && abs(Xiev[r]) < 1e-3Xiev[end]
#                 isexact = false; break
#             end
#             # Xieig = eigfact(Symmetric(H.value[ir, ir]), 1:r)
#             # @show Xieig[:vectors]
#         end
#     else
#         isexact = false
#     end
#     if verbose
#         if isexact
#             info("SDP solution solves the trace sum max problem exactly")
#         else
#             info("SDP solution solves the trace sum max problem approximately")
#         end
#     end
#     O = [zeros(T, d[i], r) for i in 1:m]
#     # columns corresponding to largest r eigenvalues
#     E = view(Heig.vectors, :, (sum(d) - r + 1):sum(d))
#     for i in 1:m
#         ir = (sum(d[1:i-1]) + 1):sum(d[1:i])
#         Eisvd = svdfact!(E[ir, :]; thin = true)
#         A_mul_B!(O[i], Eisvd.U, Eisvd.Vt)
#         # O[i] = √m * E[ir, :]
#         # @show O[i]'O[i]
#         Xieig = eigfact(- Symmetric(H.value[ir, ir]))
#         # @show Xieig[:values]
#         # @show vecnorm(E[ir, :]'Xieig[:vectors][:, r+1:end])
#         # @show vecnorm(E[ir, :] * E[ir, :]' - Xieig[:vectors][:, 1:r] * Xieig[:vectors][:, 1:r]' / m)
#     end
#     # fix identifiability and compute final trace sum objective
#     identify!(O)
#     O, obj, dot(O, S * O), isexact
# end
