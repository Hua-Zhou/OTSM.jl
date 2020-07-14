# OTSM.jl

OTSM.jl implements algorithms for solving the orthogonal trace sum maximization (OTSM) problem
  
$\operatorname{maximize} \sum_{i,j=1}^m \operatorname{tr} (O_i^T S_{ij} O_j)$

subject to orthogonality constraint $O_i^T O_i = I_r$. Here $S_{ij} \in \mathbb{R}^{d_i \times d_j}$, $1 \le i, j \le m$, are data matrices. $S_{ii}$ are symmetric and $S_{ij} = S_{ji}^T$. Many problems such as canonical correlation analysis (CCA) with $m \ge 2$ data sets, Procrustes analysis with $m \ge 2$ images, orthogonal least squares, and MaxBet are special cases of OSTM. 

Details on OTSM are described in paper: 

* Joong-Ho Won, Hua Zhou, and Kenneth Lange. (2018) Orthogonal trace-sum maximization: applications, local algorithms, and global optimality, [arXiv](https://arxiv.org/abs/1811.03521). 

## Installation

OTSM.jl requires Julia v1.0 or later. The package has not yet been registered and must be installed using the repository location. Start julia and use the `]` key to switch to the package manager REPL
```julia
(@v1.4) pkg> add https://github.com/Hua-Zhou/OTSM.jl
```
Use the backspace key to return to the Julia REPL.


```julia
versioninfo()
```

    Julia Version 1.4.2
    Commit 44fa15b150* (2020-05-23 18:35 UTC)
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

## Algorithms

### Proximal block ascent algorithm

The `otsm_pba()` function implements an efficient local search algorithm for solving OTSM.

For documentation of the `otsm_pba()` function, type `?otsm_bpa` in Julia REPL.
```@docs
otsm_pba
```

### Semidefinite programming (SDP) relaxation 

The `otsm_sdp()` function implements an SDP relaxation approach for solving OTSM.

For documentation of the `otsm_sdp()` function, type `?otsm_sdp` in Julia REPL.
```@docs
otsm_sdp
```

## Start point

Different strategies for starting point are implemented.

### Initialize $O_i$ by $I_r$

```@docs
init_eye
```

### Initialize $O_i$ by a strategy by Ten Berge (default)

This is the default for the proximal block ascent algorithm `otsm_pba`.

```@docs
init_tb
```

### Initialize $O_i$ by a strategy by Liu-Wang-Wang 

```@docs
init_lww1
```

### Initialize $O_i$ by a strategy by Shapiro-Botha and Won-Zhou-Lange

```@docs
init_sb
```

## Example data - Port Wine

The package contains the port wine example data set from the [Hanafi and Kiers (2006)](https://doi.org/10.1016/j.csda.2006.04.020) paper. It can be retrieved by the `portwine_data()` function.


```julia
A, _, _ = portwine_data();
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


## MAXDIFF

The MAXDIFF approach for CCA seeks the rotations of $A_i$ that achieve the maximal agreement

$\operatorname{maximize} 2 \sum_{i < j} \operatorname{tr} (O_i^T A_i^T A_j O_j),$

subject to constraint $O_i^T O_i = I_r$. This corresponds to an OTSM problem with $S_{ij} = A_i^T A_j$ and $S_{ii} = 0$.


```julia
Smaxdiff = [A[i]'A[j] for i in 1:4, j in 1:4]
for i in 1:4
    fill!(Smaxdiff[i, i], 0)
end
display.(Smaxdiff);
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



    4×4 Array{Float64,2}:
      9.75  -26.0  -14.625   21.125
      8.0   -27.0  -10.0     18.0
     -6.0    21.0    2.5    -13.5
      4.5    -8.0  -12.75     6.75



    3×4 Array{Float64,2}:
     6.0   -11.0  -12.0     8.0
     1.75    9.0   -6.625   1.125
     0.25    3.0    6.125  -2.625



    4×3 Array{Float64,2}:
       5.0   -9.25     5.75
     -15.0   36.0    -11.0
     -12.0    5.875  -14.125
      13.0  -20.375    7.625



    3×3 Array{Float64,2}:
     0.0  0.0  0.0
     0.0  0.0  0.0
     0.0  0.0  0.0



    4×3 Array{Float64,2}:
     13.0  -21.625  14.375
     11.0  -21.0    11.0
     -5.0   16.5    -2.5
     10.0   -5.75   13.25



    3×3 Array{Float64,2}:
      9.0  -8.0    12.0
      1.0   4.375   0.375
     -6.0  -0.875  -8.875



    4×4 Array{Float64,2}:
       9.75     8.0   -6.0    4.5
     -26.0    -27.0   21.0   -8.0
     -14.625  -10.0    2.5  -12.75
      21.125   18.0  -13.5    6.75



    3×4 Array{Float64,2}:
      13.0     11.0  -5.0  10.0
     -21.625  -21.0  16.5  -5.75
      14.375   11.0  -2.5  13.25



    4×4 Array{Float64,2}:
     0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0



    3×4 Array{Float64,2}:
     13.0    10.0  -4.0  10.0
      2.875  -2.0  -0.5   1.25
     -2.375  -4.0  -4.5  -7.25



    4×3 Array{Float64,2}:
       6.0   1.75    0.25
     -11.0   9.0     3.0
     -12.0  -6.625   6.125
       8.0   1.125  -2.625



    3×3 Array{Float64,2}:
      9.0  1.0    -6.0
     -8.0  4.375  -0.875
     12.0  0.375  -8.875



    4×3 Array{Float64,2}:
     13.0   2.875  -2.375
     10.0  -2.0    -4.0
     -4.0  -0.5    -4.5
     10.0   1.25   -7.25



    3×3 Array{Float64,2}:
     0.0  0.0  0.0
     0.0  0.0  0.0
     0.0  0.0  0.0


Proximal block ascent algorithm for finding a rank $r=2$ solution to MAXDIFF.


```julia
Ô_pba, ts_pba, obj, history = otsm_pba(Smaxdiff, 2; verbose = true);
```

    iter = 1, obj = 539.8501989834106
    iter = 2, obj = 542.2346791607897
    iter = 3, obj = 542.326755374587
    iter = 4, obj = 542.3275270111226
    iter = 5, obj = 542.327550329459
    iter = 6, obj = 542.3275506362339
    iter = 7, obj = 542.3275506383457
    iter = 8, obj = 542.3275506383522


The `test_optimality()` function attempts to certify whether a local solution `O::Vector{Matrix}` is a global solution. The first output indicates the solution is `:infeasible`, `:suboptimal`, `:stationary_point`, or `:global_optimal`.


```julia
# proximal block ascent yields the global solution
test_optimality(Ô_pba, Smaxdiff)[1]
```




    (:global_optimal, -3.507576956770025e-14)



For documentation of the `test_optimality()` function, type `?test_optimality` in Julia REPL.
```@docs
test_optimality
```

## MAXBET

The MAXBET approach for CCA seeks the rotations of $A_i$ that achieve the maximal agreement

$\operatorname{maximize} \sum_{i,j} \operatorname{tr} (O_i^T A_i^T A_j O_j),$

subject to constraint $O_i^T O_i = I_r$. This corresponds to an OTSM problem with $S_{ij} = A_i^T A_j$.


```julia
Smaxbet = [A[i]'A[j] for i in 1:4, j in 1:4]
display.(Smaxbet);
```


    4×4 Array{Float64,2}:
       5.5   -12.0  -6.25     7.25
     -12.0    54.0   6.0    -31.0
      -6.25    6.0  17.875   -9.375
       7.25  -31.0  -9.375   28.875



    3×4 Array{Float64,2}:
      5.0   -15.0  -12.0     13.0
     -9.25   36.0    5.875  -20.375
      5.75  -11.0  -14.125    7.625



    4×4 Array{Float64,2}:
      9.75  -26.0  -14.625   21.125
      8.0   -27.0  -10.0     18.0
     -6.0    21.0    2.5    -13.5
      4.5    -8.0  -12.75     6.75



    3×4 Array{Float64,2}:
     6.0   -11.0  -12.0     8.0
     1.75    9.0   -6.625   1.125
     0.25    3.0    6.125  -2.625



    4×3 Array{Float64,2}:
       5.0   -9.25     5.75
     -15.0   36.0    -11.0
     -12.0    5.875  -14.125
      13.0  -20.375    7.625



    3×3 Array{Float64,2}:
      12.0  -10.0    11.0
     -10.0   29.875  -7.125
      11.0   -7.125  15.875



    4×3 Array{Float64,2}:
     13.0  -21.625  14.375
     11.0  -21.0    11.0
     -5.0   16.5    -2.5
     10.0   -5.75   13.25



    3×3 Array{Float64,2}:
      9.0  -8.0    12.0
      1.0   4.375   0.375
     -6.0  -0.875  -8.875



    4×4 Array{Float64,2}:
       9.75     8.0   -6.0    4.5
     -26.0    -27.0   21.0   -8.0
     -14.625  -10.0    2.5  -12.75
      21.125   18.0  -13.5    6.75



    3×4 Array{Float64,2}:
      13.0     11.0  -5.0  10.0
     -21.625  -21.0  16.5  -5.75
      14.375   11.0  -2.5  13.25



    4×4 Array{Float64,2}:
      26.875   20.0  -13.5  12.25
      20.0     18.0  -11.0   9.0
     -13.5    -11.0   12.0  -2.0
      12.25     9.0   -2.0  11.5



    3×4 Array{Float64,2}:
     13.0    10.0  -4.0  10.0
      2.875  -2.0  -0.5   1.25
     -2.375  -4.0  -4.5  -7.25



    4×3 Array{Float64,2}:
       6.0   1.75    0.25
     -11.0   9.0     3.0
     -12.0  -6.625   6.125
       8.0   1.125  -2.625



    3×3 Array{Float64,2}:
      9.0  1.0    -6.0
     -8.0  4.375  -0.875
     12.0  0.375  -8.875



    4×3 Array{Float64,2}:
     13.0   2.875  -2.375
     10.0  -2.0    -4.0
     -4.0  -0.5    -4.5
     10.0   1.25   -7.25



    3×3 Array{Float64,2}:
     10.0   2.0    -5.0
      2.0  10.875   4.625
     -5.0   4.625  11.875


Proximal block ascent algorithm for finding a rank $r=2$ solution to MAXBET.


```julia
Ô_pba, ts_pba, obj, history = otsm_pba(Smaxbet, 2; verbose = true);
```

    iter = 1, obj = 769.5257682063867
    iter = 2, obj = 778.9896186367976
    iter = 3, obj = 779.6705236544665
    iter = 4, obj = 779.7414861674778
    iter = 5, obj = 779.7533232474
    iter = 6, obj = 779.756174278359
    iter = 7, obj = 779.7569547275209
    iter = 8, obj = 779.7571745803215
    iter = 9, obj = 779.7572368336034
    iter = 10, obj = 779.7572544704886
    iter = 11, obj = 779.7572594658141
    iter = 12, obj = 779.7572608802018
    iter = 13, obj = 779.757261280583
    iter = 14, obj = 779.757261393906
    iter = 15, obj = 779.7572614259773
    iter = 16, obj = 779.7572614350538
    iter = 17, obj = 779.7572614376222
    iter = 18, obj = 779.7572614383489
    iter = 19, obj = 779.7572614385546
    iter = 20, obj = 779.7572614386127
    iter = 21, obj = 779.7572614386298
    iter = 22, obj = 779.7572614386339
    iter = 23, obj = 779.7572614386349
    iter = 24, obj = 779.7572614386353
    iter = 25, obj = 779.7572614386356


This solution is certified to be global optimal.


```julia
# proximal block ascent yields the global solution
test_optimality(Ô_pba, Smaxbet)[1]
```




    (:global_optimal, 6.830358875212792e-15)


