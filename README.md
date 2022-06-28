# OTSM.jl

| **Documentation** | **Build Status** | **Code Coverage**  |
|-------------------|------------------|--------------------|
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://Hua-Zhou.github.io/OTSM.jl/stable) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://Hua-Zhou.github.io/OTSM.jl/dev) | [![Build Status](https://travis-ci.org/Hua-Zhou/OTSM.jl.svg?branch=master)](https://travis-ci.org/Hua-Zhou/OTSM.jl)  | [![Coverage Status](https://coveralls.io/repos/github/Hua-Zhou/OTSM.jl/badge.svg?branch=master)](https://coveralls.io/github/Hua-Zhou/OTSM.jl?branch=master) [![codecov](https://codecov.io/gh/Hua-Zhou/OTSM.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Hua-Zhou/OTSM.jl) |  


The Julia package OTSM.jl implements a few algorithms for the orthogonal trace sum maximization (OTSM) discussed in the following paper.

* Joong-Ho Won, Hua Zhou, and Kenneth Lange. (2021) Orthogonal trace-sum maximization: applications, local algorithms, and global optimality, [_SIAM Journal on Matrix Analysis and Applications_](https://doi.org/10.1137/20M1363388), 42(2):859-882. [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:WQUHiJFXdmkJ:scholar.google.com/&output=citation&scisdr=CgUnszo7EOCa8S6HAB4:AAGBfm0AAAAAYruBGB6grwLMQGRBDlqLvfst9-uM8h95&scisig=AAGBfm0AAAAAYruBGOcmm_E8eVKTBAFRXzLlXOEQepFj&scisf=4&ct=citation&cd=-1&hl=en)

* Joong-Ho Won, Teng Zhang, and Hua Zhou. (2022) Orthogonal trace-sum maximization: tightness of the semidefinite relaxation and guarantee of locally optimal solutions. [_SIAM Journal of Optimization_](). [arXiv](https://arxiv.org/abs/2110.05701)

OTSM.jl supports Julia v1.0 or later. See documentation for usage. It is not yet registered and can be installed, in the Julia Pkg mode, by
```{julia}
(@v1.4) Pkg> add https://github.com/Hua-Zhou/OTSM.jl
```
