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
        O = qr!(randn(T, d, d)).Q # Haar measure (uniform) on O(n)
        A[k] = Ā * O + σ * randn(n, d)
        # make sure to be Haar measure (uniform) on SO(n)
        # if det(O)=-1 (reflection), swap the first two columns
        det(O) < 0 && (A[k][:, [1, 2]] = A[k][:, [2, 1]])
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
