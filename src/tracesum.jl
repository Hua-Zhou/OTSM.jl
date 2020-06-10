using Convex, IterativeSolvers, LinearAlgebra, SCS
import Base: cat
import LinearAlgebra: BlasReal

"""
	cat(A::Matrix{Matrix{T}})

Concatenate an array of matrices into a single matrix. Similar to `cell2mat` in
Matlab.
"""
function Base.cat(A::VecOrMat{Matrix{T}}) where T
	m = map(M -> size(M, 1), A[:, 1]) # row counts
	n = map(M -> size(M, 2), A[1, :]) # column counts
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
	verify_input_data!(S, r)

`S` is an array of matrices. Enforce `S[i, j] = S[j, i]'` and `S[i, i] == 0`
and check `r` is less than `size(S[i, j], 1)`.
"""
function verify_input_data!(
	S :: AbstractMatrix{<:AbstractMatrix},
	r :: Integer
	)
	@assert size(S, 1) == size(S, 2) "size(S, 1) should be equal to size(S, 2)"
	m = size(S, 1)
	@inbounds for i in 1:m
		@assert r ≤ size(S[i, i], 1) "rank r > size(S[$i, $i], 1) !"
		fill!(S[i, i], 0)
        for j in (i + 1):m
            S[j, i] .= transpose(S[i, j])
		end
	end
	S, r
end

"""
	identify!(O::Vector{Matrix})

Fix identifiability of rectangular orthogonal matrices `O[1], ..., O[end]`, where
`O[i]' * O[i] = eye(r)`, by reducing `O[end]` into a lower triangular matrix with
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
        ds   = sign.(diag(L))  # sign of diagonal of L
        rmul!(L, Diagonal(ds)) # make diagonal entries of L to be positive
        @inbounds for i in 1:(m - 1)
            copyto!(tmp[i], O[i])
            mul!(O[i], tmp[i], transpose(Q))
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

Maximize the trace sum `2 sum_{i<j} trace(Oi' * S[i, j] * Oj) = sum_{i≠j}
trace(Oi' * S[i, j] * Oj)` subject to orthogonality constraint `Oi' * Oi
== I(r)` by a proximal block ascent algorithm. Each of `S[i, j]` for `i < j` is a
`di x dj` matrix.

# Positional arguments
- `S       :: Matrix{Matrix}`: `S[i, j]` is a `di x dj` matrix.
- `r       :: Integer`       : rank of solution.

# Keyword arguments
- `αinv    :: Number`: proximal update constant ``1/α```, default is `1e-3`.
- `maxiter :: Integer`: maximum number of iterations, default is `1000`.
- `tolfun  :: Number`: tolerance for objective convergence, default is `1e-8`.
- `tolvar  :: Number`: tolerance for iterate convergence, default is `1e-6`.
- `verbose :: Bool`  : verbose display, default is `false`.
- `O       :: Vector{Matrix}`: starting point, default is `O[i] = eye(di, r)`.
- `log     :: Bool`: record iterate history or not, defaut is `false`.

# Output
- `O`       : result, `O[i]` has dimension `di x r`.
- `tracesum`: objective value evaluated at final `O`.
- `obj`     : final objective value from PBA algorithm, should be same as `obj`.
- `history` : iterate history.
"""
function otsm_pba(
    S       :: Matrix{Matrix{T}},
    r       :: Integer;
	αinv    :: Number  = 1e-3,
    maxiter :: Integer = 1000,
    tolfun  :: Number  = 1e-8,
    tolvar  :: Number  = 1e-6,
    verbose :: Bool    = false,
    log     :: Bool    = false,
    O       :: Vector{Matrix{T}} = init_tb(S, r)
    ) where T <: BlasReal
    m = size(S, 1)
    d = [size(S[i, i], 1) for i in 1:m] # (d[i], d[j]) = size(S[i, j])
    # record iterate history if requested
    history          = ConvergenceHistory(partial = !log)
    history[:tolfun] = tolfun
    history[:tolvar] = tolvar
    IterativeSolvers.reserve!(T, history, :tracesum, maxiter)
    IterativeSolvers.reserve!(T, history, :vchange , maxiter)
    # check (1) Sji = Sij', (2) Sii = 0
    verify_input_data!(S, r)
    # initial objective value
    SO      = S * O # SO[i] = sum_{j≠i} S_{ij} O_j
    obj::T  = dot(O, SO)
    IterativeSolvers.nextiter!(history)
    push!(history, :tracesum, obj)
    # pre-allocate intermediate arrays
    ΔO  = [similar(O[i]) for i in 1:m]
    tmp = [similar(O[i]) for i in 1:m]
    Λi  = Matrix{T}(undef, r, r) # Lagrange multipliers
    for iter in 1:maxiter-1
        IterativeSolvers.nextiter!(history)
        verbose && println("iter = $iter, obj = $obj")
        # block update
        @inbounds for i in 1:m
            # update Oi
            # tmp[i] = SO[i] + αinv * O[i]
            BLAS.axpy!(αinv, O[i], copyto!(tmp[i], SO[i]))
            Fi = svd!(tmp[i]; full = false)
            copyto!(ΔO[i], O[i])
            mul!(O[i], Fi.U, Fi.Vt)
            ΔO[i] .= O[i] .- ΔO[i]
            # update storage[j], j ≠ i
            for j in 1:m
                j ≠ i &&
                BLAS.gemm!('N', 'N', T(1), S[j, i], ΔO[i], T(1), SO[j])
            end
        end
        objold  = obj
        obj     = dot(O, SO)
        vchange = sum(norm, ΔO) / m # mean Frobenius norm of variable change
        push!(history, :tracesum,     obj)
        push!(history, :vchange , vchange)
        (vchange < tolvar) && 
        (abs(obj - objold) < tolfun * abs(objold + 1)) &&
        IterativeSolvers.setconv(history, true) &&
        break
    end
    # fix identifiability and compute final trace sum objective
    identify!(O)
    mul!(SO, S, O)
    tracesum::T = dot(O, SO)
    log && IterativeSolvers.shrink!(history)
    O, tracesum, obj, history
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

"""
    otsm_sdp(S, r)

Maximize the trace sum `2sum_{i<j} trace(Oi' * S[i, j] * Oj)` subject to the
orthogonality constraint `Oi' * Oi == I(r)` using an SDP relaxation (P-SDP in 
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
    # check (1) Sji = Sij, (2) Sii = 0
    verify_input_data!(S, r)
    S̃ = cat(S) # flattened matrix
    # form and solve the SDP
    U       = Convex.Semidefinite(sum(d), sum(d))
    problem = Convex.maximize(m * tr(S̃ * U))
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
	test_optimality(O::Vector{Matrix}, S::Matrix{Matrix}; tol)

Test if the vector of orthogonal matrices `O` is globally optimal. The `O` is 
assumed to satisfy the first-order optimality conditions. Each of `O[i]` is a 
`di x r` matrix. Each of `S[i, j]` for `i < j` is a `di x dj` matrix. 
Tolerance `tol` is used to test the positivity of the smallest eigenvalue
of the test matrix.

# Positional arguments
- `O :: Vector{Matrix}`: a point satisifying the 1st order optimality.
- `S :: Matrix{Matrix}`: data matrix.

# Output
- `z`  : 1=globally optimal; 0=undecided; -1=suboptimal.
- `val`: smallest eigenvalue of the test matrix if `z=1` or `0`; `-Inf` otherwise.
"""
function test_optimality(
    O   :: Vector{Matrix{T}},
    S   :: Matrix{Matrix{T}},
	tol :: Number = 1e-3
    ) where T <: BlasReal
    r  = size(O[1], 2)
    verify_input_data!(S, r) # check (1) Sji = Sij', (2) Sii = 0
	L  = copy(-S) # test matrix
    m  = size(S, 1)
    SO = S * O 	# SO[i] = sum_{j≠i} S_{ij} O_j
	Λi = Matrix{T}(undef, r, r)
    @inbounds for i in 1:m
        mul!(Λi, transpose(O[i]), SO[i])
        # check symmetry of Λi (first order optimality condition)
        δi = check_symmetry(Λi) 
        δi > abs(tol) && 
            @warn "Λ$i not symmetric; norm(Λ - Λ') = $δi; " *
            "first order optimality not satisfied"
		λmin = eigmin(Symmetric(Λi))
        λmin < -abs(tol) && (return -1, -Inf)
        for j in 1:r
            Λi[j, j] -= λmin
        end
        L[i, i] .+= O[i] * Symmetric(Λi) * transpose(O[i])
        for k in 1:size(O[i], 1)
            L[i, i][k, k] += λmin
        end
	end
	λmin = eigmin(Symmetric(cat(L)))
	z = λmin > -abs(tol) ? 1 : 0
	return z, λmin
end

"""
	init_eye(S)

Initialize `O[i]` by an `di x r` diagonal matrix with ones on diagonal.
"""
function init_eye(
    S :: Matrix{Matrix{T}},
    r :: Integer
    ) where T <: BlasReal
    [Matrix{T}(I, size(S[i, i], 1), r) for i in 1:size(S, 1)]
end

"""
	init_tb(S)

Compute initial point following Ten Berge's second upper bound (p. 273).
Take the eigenvectors corresponding to the r largest eigenvalues of S.
This is a `D x r` orthogonal matrix, ``D=\\sum_{i=1}^m d_i``.
For each `di x r` block, project to the Stiefel manifold.
These blocks constitute an initial point.
"""
function init_tb(
    S :: Matrix{Matrix{T}},
    r :: Integer
    ) where T <: BlasReal
    m = size(S, 1)
    # only need evecs corresponding to largest r evals
	V = (eigen!(Symmetric(cat(S))).vectors)[:, end:-1:end-r+1]
	rowoffset = 0
	O = [Matrix{T}(undef, size(S[i, i], 1), r) for i in 1:m]
	for i in 1:m
		di         = size(S[i, i], 1)
		rowidx     = rowoffset+1:rowoffset+di
		Visvd      = svd!(V[rowidx, :])
		mul!(O[i], Visvd.U, Visvd.Vt)
		rowoffset += di
	end
	return O
end

"""
	init_sb(S, r)

Compute initial point following Shapiro-Botha (p. 380). Fill in the diagonal 
block of `S` with ``-\\sum_{j \\neq i} P_{ij} * D_{ij} * P_{ij}``, where 
``P_{ij} * D_{ij} * Q_{ij}`` is the SVD of ``S_{ij}``. The resulting matrix is 
negative semidefinite. Take the eigenvectors corresponding to the `r` largest 
eigenvalues. This is a `D x r` orthogonal matrix, ``D=\\sum_{i=1}^m d_i``.
For each `di x r` block, project to the Stiefel manifold.
These blocks constitute an initial point.
"""
function init_sb(
    S :: Matrix{Matrix{T}},
    r :: Integer
    ) where T <: BlasReal
	m    = size(S, 1)
    SS   = copy(S) # note SS[i,j] and S[i,j] point to same data
    Ssvd = [svd(S[i, j]) for i in 1:m, j in 1:m]
    for i in 1:m
        # now SS[i,i] points to diff data than S[i,i], so we are not worrried
        # changing data in S[i, i]
        SS[i, i] = zeros(T, size(SS[i, i])) 
		for j in 1:m
            if j ≠ i 
                svdij = Ssvd[i, j]
				SS[i, i] .-= svdij.U * Diagonal(svdij.S) * transpose(svdij.U)
			end
		end
	end
	# only need evecs corresponding to largest r evals
	V = (eigen!(Symmetric(cat(SS))).vectors)[:, end:-1:end-r+1]
	rowoffset = 0
	O = [Matrix{T}(undef, size(S[i, i], 1), r) for i in 1:m]
	for i in 1:m
		di = size(SS[i,i], 1)
		rowidx     = rowoffset+1:rowoffset+di
		Visvd      = svd!(V[rowidx, :])
		mul!(O[i], Visvd.U, Visvd.Vt)
		rowoffset += di
	end
	return O
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
