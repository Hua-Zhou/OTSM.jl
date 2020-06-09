# OTSM.jl

OTSM.jl implements algorithms for solving the orthogonal trace sum maximization (OTSM) problem
  
$\operatorname{maximize} \sum_{i,j=1}^m \operatorname{tr} (O_i^T S_{ij} O_j)$

subject to orthogonality constraint $O_i^T O_i = I_r$. Here $S_{i,j} \in \mathbb{R}^{d_i \times d_j}$, $1 \le i, j \le m$, are data matrices. Many problems such as canonical correlation analysis (CCA) with $m \ge 2$ data sets, Procrustes analysis with $m \ge 2$ images, and orthogonal least squares are special cases of OSTM. 

Details on OTSM are described in paper: 

* Joong-Ho Won, Hua Zhou, and Kenneth Lange. (2018) Orthogonal trace-sum maximization: applications, local algorithms, and global optimality, [arXiv](https://arxiv.org/abs/1811.03521). 

## Installation

This package requires Julia v1.0 or later, which can be obtained from
<https://julialang.org/downloads/> or by building Julia from the sources in the
<https://github.com/JuliaLang/julia> repository.

The package has not yet been registered and must be installed using the repository location.
Start julia and use the `]` key to switch to the package manager REPL
```julia
(@v1.4) pkg> add https://github.com/Hua-Zhou/OTSM.jl
```
Use the backspace key to return to the Julia REPL.


```julia
versioninfo()
```

    Julia Version 1.4.1
    Commit 381693d3df* (2020-04-14 17:20 UTC)
    Platform Info:
      OS: macOS (x86_64-apple-darwin18.7.0)
      CPU: Intel(R) Core(TM) i7-6920HQ CPU @ 2.90GHz
      WORD_SIZE: 64
      LIBM: libopenlibm
      LLVM: libLLVM-8.0.1 (ORCJIT, skylake)
    Environment:
      JULIA_EDITOR = code
      JULIA_NUM_THREADS = 4



```julia
# for use in this tutorial
using OTSM
```

## Example data

The package contains the port wine example data set from the [Hanafi and Kiers (200)](https://doi.org/10.1016/j.csda.2006.04.020) paper. It can be retrieved by the `portwine_data()` function.


```julia
A, S, = portwine_data();
```

Data matrices A1, A2, A3, A4 record the ratings (centered at 0) of $m=4$ accessors on 8 port wines in $d_1=4$, $d_2=3$, $d_3=4$, and $d_4=3$ aspects respectively. 


```julia
for i in 1:4
    display(A[i])
end
```


    8×4 Array{Float64,2}:
      1.25  -5.0  -1.375   3.875
     -0.75   1.0  -0.375  -1.125
      1.25  -3.0  -1.375   0.875
     -0.75   2.0   0.625  -0.125
     -0.75   2.0  -0.375  -0.125
      0.25   3.0  -0.375  -3.125
     -0.75  -1.0   3.625  -1.125
      0.25   1.0  -0.375   0.875



    8×3 Array{Float64,2}:
      2.0  -4.375   0.625
      1.0   1.625   0.625
      1.0  -1.375   2.625
     -1.0   1.625  -1.375
      0.0   0.625   0.625
     -1.0   0.625  -0.375
     -2.0  -0.375  -2.375
      0.0   1.625  -0.375



    8×4 Array{Float64,2}:
      3.125   3.0  -2.5   0.75
     -1.875  -1.0   1.5   0.75
      2.125   2.0  -0.5   1.75
     -1.875  -1.0   1.5  -1.25
      1.125   0.0   0.5   0.75
     -0.875  -1.0   0.5  -0.25
     -1.875  -1.0  -0.5  -2.25
      0.125  -1.0  -0.5  -0.25



    8×3 Array{Float64,2}:
      1.0   0.125   0.375
      0.0  -0.875  -1.625
      2.0  -0.875  -1.625
     -1.0   0.125  -0.625
      0.0   0.125  -0.625
      0.0   1.125   1.375
     -2.0  -1.875   1.375
      0.0   2.125   1.375


The MAXDIFF approach for CCA seeks the rotations of $A_i$ that achieve the maximal agreement

$\operatorname{maximize} \sum_{i < j} \operatorname{tr} (O_i^T A_i^T A_j O_j),$

subject to constraint $O_i^T O_i = I_r$. This corresponds to an OTSM problem with $S_{ij} = A_i^T A_j$ and $S_{ii} = 0$.


```julia
for i in 1:4, j in 1:i
    display(S[i, j])
end
```


    4×4 Array{Float64,2}:
     0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0



    3×4 Array{Float64,2}:
      5.0   -15.0  -12.0     13.0
     -9.25   36.0    5.875  -20.375
      5.75  -11.0  -14.125    7.625



    3×3 Array{Float64,2}:
     0.0  0.0  0.0
     0.0  0.0  0.0
     0.0  0.0  0.0



    4×4 Array{Float64,2}:
      9.75  -26.0  -14.625   21.125
      8.0   -27.0  -10.0     18.0
     -6.0    21.0    2.5    -13.5
      4.5    -8.0  -12.75     6.75



    4×3 Array{Float64,2}:
     13.0  -21.625  14.375
     11.0  -21.0    11.0
     -5.0   16.5    -2.5
     10.0   -5.75   13.25



    4×4 Array{Float64,2}:
     0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0



    3×4 Array{Float64,2}:
     6.0   -11.0  -12.0     8.0
     1.75    9.0   -6.625   1.125
     0.25    3.0    6.125  -2.625



    3×3 Array{Float64,2}:
      9.0  -8.0    12.0
      1.0   4.375   0.375
     -6.0  -0.875  -8.875



    3×4 Array{Float64,2}:
     13.0    10.0  -4.0  10.0
      2.875  -2.0  -0.5   1.25
     -2.375  -4.0  -4.5  -7.25



    3×3 Array{Float64,2}:
     0.0  0.0  0.0
     0.0  0.0  0.0
     0.0  0.0  0.0


## Proximal block ascent algorithm

The `otsm_pba()` function implements an efficient local search algorithm for solving OTSM.


```julia
Ô_pba, ts_pba, obj, history = otsm_pba(S, 2; verbose = true);
```

    iter = 1, obj = 110.25
    iter = 2, obj = 533.6042318034453
    iter = 3, obj = 542.2027792984238
    iter = 4, obj = 542.3265730402211
    iter = 5, obj = 542.3275463498419
    iter = 6, obj = 542.3275506295132
    iter = 7, obj = 542.327550638136


For documentation of the `otsm_pba()` function, type ?otsm_bpa in Julia REPL.
```@docs
otsm_pba
```

## Check global optimality of a local solution

The `test_optimality()` function attempts to certify whether a local solution `O::Vector{Matrix}` is a global solution. By a local solution, we mean a point that satifies the first order optimality condition:
$$
\Lambda_i = \sum_{j \ne i} O_i^T S_{ij} O_j
$$
is symmetric for $i=1,\ldots,m$. The first output indicates the solution is global optimal (1), or uncertain (0), or suboptimal (-1).


```julia
# proximal block ascent yields the global solution
test_optimality(Ô_pba, S)[1]
```




    1



For documentation of the `test_optimality()` function, type `?test_optimality` in Julia REPL.
```@docs
test_optimality
```
