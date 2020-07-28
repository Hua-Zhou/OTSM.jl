using LinearAlgebra, Random

"""
    generate_procrustes_data(m, n, d, σ=0.1, T=Float64)

Generate random (generalized) Procrustes data.

# Input
- `m :: Integer`: number of images.
- `n :: Integer`: number of image landmarks.
- `d :: Integer`: dimension of image (landmark points).
- `σ :: Number = 0.1`: noise level adding to each rotated image.
- `T :: DataType = Float64`: data type.

# Output
- `A :: Vector{Matrix{T}}`: `m` rotated images.
- `S :: Matrix{Matrix{T}}`: matrix of `S[i, j]` for OTSM problem.
- `Ā :: Matrix{T}`: the true/center image.
- `A_manopt :: Array{T, 3}`: 3D array of data suitable for Manopt software.
"""
function generate_procrustes_data(
    m :: Integer, 
    n :: Integer,
    d :: Integer, 
    σ :: Number   = 0.1, 
    T :: DataType = Float64
    )
    # true center image
    Ā = randn(T, n, d)
    # randomly rotated images
    A = Vector{Matrix{T}}(undef, m)
    for k in 1:m
        # Haar measure (uniform) on O(n)
        # M with iid std normal entries
        # M = QR
        Q, R = qr!(randn(T, d, d))
        O = Matrix(Q)
        # In Julia, diagonal entries of R not necessarily positive
        for j in 1:d
            R[j, j] < 0 && (O[:, j] *= -1)
        end
        # now O is uniform on O(n), not on SO(n)
        # if det(O)=-1 (reflection), swap the first two columns
        det(O) < 0 && (O[:, [1, 2]] = O[:, [2, 1]])
        # rotate each point (row) in A
        A[k] = Ā * transpose(O) + σ * randn(T, n, d)
    end
    # set S[i, j]
    S = [A[i]'A[j] for i in 1:m, j in 1:m]
    for i in 1:m
        fill!(S[i, i], 0)
    end
    A_manopt = Array{T, 3}(undef, d, n, m)
    for k in 1:m
        A_manopt[:, :, k] = transpose(A[k])
    end
    A, S, Ā, A_manopt
end
