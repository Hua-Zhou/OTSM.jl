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

Compute initial point following Ten Berge's second upper bound (p. 273). This is
same as  the Liu-Wang-Wang starting point strategy 2 (Algorithm 3.2, p. 1495).
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
    O = Vector{Matrix{T}}(undef, m)
	for i in 1:m
		di     = size(S[i, i], 1)
		rowidx = rowoffset+1:rowoffset+di
		Visvd  = svd!(V[rowidx, :])
		O[i]   = Visvd.U * Visvd.Vt
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
These blocks constitute an initial point. This starting strategy only applies 
to cases where `A[i, i] = 0`.
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
		di      = size(SS[i,i], 1)
		rowidx  = rowoffset+1:rowoffset+di
		Visvd   = svd!(V[rowidx, :])
		mul!(O[i], Visvd.U, Visvd.Vt)
		rowoffset += di
	end
	return O
end

"""
	init_lww1(S, r)

Compute initial point following Liu-Wang-Wang starting point strategy 1 
(Algorithm 3.1, p. 1494). This starting strategy applies 
to cases where `A[i, i]` is pd, or equivalently `make_Sii_zero = false`.
"""
function init_lww1(
    S :: Matrix{Matrix{T}},
    r :: Integer
    ) where T <: BlasReal
    m = size(S, 1)
    O = Vector{Matrix{T}}(undef, m)
    for k in 1:m
        if k == 1
            O[k] = eigen(Symmetric(S[k, k])).vectors[:, end:-1:end-r+1]
        else
            Uk = eigen(Symmetric(S[k, k])).vectors[:, end:-1:end-r+1]
            @views O[k] = Uk * qr(transpose(Uk) * (S[k:k, 1:k-1] * O[1:k-1])[1]).Q
        end
    end
    return O
end
