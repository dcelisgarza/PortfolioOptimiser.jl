#src
#src Copywrite (c) 2025
#src Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
#src SPDX-License-Identifier: MIT

#=
# Risk Measures

This file contains the examples in the docstrings as runnable code.
=#

using PortfolioOptimiser, HiGHS, Clarabel, Pajarito, JuMP, StatsBase

## Randomly generated normally distributed returns.
ret = [0.670643    1.94045   -0.0896267   0.851535    -0.268234
       1.33575    -0.541003   2.28744    -0.157588    -1.45177
       -1.91694    -0.167745   0.920495    0.00677243  -1.29112
       0.123141    1.59841   -0.185076    2.58911     -0.250747
       1.92782     1.01679    1.12107     1.09731     -0.99954
       2.07114    -0.513216  -0.532891    0.917748    -0.0346682
       -1.37424    -1.35272   -0.628216   -2.76234     -0.112378
       1.3831      1.14021   -0.577472    0.224504     1.28137
       -0.0577619  -0.10658   -0.637011    1.70933      1.84176
       1.6319      2.05059   -0.21469    -0.640715     1.39879];

## Instantiate portfolio instance.
port = Portfolio(; ret = ret, assets = [:A, :B, :C, :D, :E],
                 solvers = PortOptSolver(; name = :PClGL,
                                         solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                                            "verbose" => false,
                                                                            "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                     MOI.Silent() => true),
                                                                            "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                        "verbose" => false,
                                                                                                                        "max_step_fraction" => 0.75))));

## Compute asset statistics.                                                
asset_statistics!(port)
## Clusterise assets (for hierarchical optimisations).
cluster_assets!(port)

#=
# Variance, [`Variance`](@ref)
=#

## If `sigma` is not `nothing` it must be a square matrix. This works at instantiation and runtime.
try
    Variance(; sigma = [1 0 0;
                        0 2 3])
catch err
    println(err)
end

try
    rm = Variance(;)
    rm.sigma = [1 0 0;
                0 2 3]
catch err
    println(err)
end

## Default formulation.
rm = Variance()
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
r1 = calc_risk(port; rm = rm)
## As a functor, we need to provide the covariance matrix to the risk measure directly.
rm.sigma = port.cov
isapprox(r1, rm(w1.weights))
## The value of :variance_risk is consistent with the risk calculation.
isapprox(r1, value(port.model[:variance_risk]))
## The [`SecondOrderCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Second-Order-Cone) constraint.
port.model[:constr_dev_soc]
## The variance risk.
port.model[:variance_risk]
## Variance risk is a [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr)
typeof(port.model[:variance_risk])

## Incompatible with [`NOC`](@ref)
w2 = optimise!(port, NOC(; rm = rm, str_names = true))

## Quadratic formulation.
rm = Variance(; formulation = Quad())
w3 = optimise!(port, Trad(; rm = rm, str_names = true))
port.model[:constr_dev_soc]
port.model[:variance_risk]
## Variance risk is a [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr)
typeof(port.model[:variance_risk])

## Incompatible with [`NOC`](@ref)
w4 = optimise!(port, NOC(; rm = rm, str_names = true))

## If either `network_adj` or `cluster_adj` field of the [`Portfolio`](@ref) instance is [`SDP`](@ref), the formulation has no effect because this constraint type requires a [`PSDCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Positive-Semidefinite-Cone) formulation of the variance.

A = connection_matrix(port)
B = cluster_matrix(port)

port.cluster_adj = SDP(; A = B)

rm = Variance()
w5 = optimise!(port, Trad(; rm = rm, str_names = true))
r5 = calc_risk(port; rm = rm)
port.model[:constr_M_PSD]
port.model[:variance_risk]
typeof(port.model[:variance_risk])

## Compatible with [`NOC`](@ref)
w6 = optimise!(port, NOC(; rm = rm, str_names = true))
## The risk of the [`NOC`](@ref) optimisation is higher than the optimal value.
r6 = calc_risk(port, :NOC; rm = rm)
r5 <= r6

## No adjacency constraints.
port.cluster_adj = NoAdj()

## Default formulation, with a variance upper bound of 10, standard deviaion of sqrt(10) = 3.1622776601683795.
rm = Variance(; settings = RMSettings(; ub = 10))
w7 = optimise!(port, Trad(; rm = rm, str_names = true))
## dev <= sqrt(10)
port.model[:dev_ub]

## Quadratic formulation, with a variance upper bound of 10, standard deviaion of sqrt(10) = 3.1622776601683795.
rm = Variance(; formulation = Quad(), settings = RMSettings(; ub = 10))
w8 = optimise!(port, Trad(; rm = rm, str_names = true))
## dev <= sqrt(10)
port.model[:dev_ub]

## We use an SDP constraint, this time on the network adjacency, to show that the formulation and risk upper bound change accordingly.
port.network_adj = SDP(; A = A)
rm = Variance(; formulation = Quad(), settings = RMSettings(; ub = 10))
w9 = optimise!(port, Trad(; rm = rm, str_names = true))
port.model[:constr_M_PSD]
port.model[:variance_risk]
typeof(port.model[:variance_risk])
## variance_risk <= 10
port.model[:variance_risk_ub]

## No adjacency constraints.
port.network_adj = NoAdj()

## Optimisations which use [`calc_risk`](@ref) to compute the risk have no [`JuMP`](https://github.com/jump-dev/JuMP.jl) model, therefore the formulation has no effect.
w10 = optimise!(port, HRP(; rm = rm))

#=
# Standard Deviation, [`SD`](@ref)
=#

## If `sigma` is not `nothing` it must be a square matrix. This works at instantiation and runtime.
try
    SD(; sigma = [1 0 0;
                  0 2 3])
catch err
    println(err)
end

try
    rm = SD(;)
    rm.sigma = [1 0 0;
                0 2 3]
catch err
    println(err)
end

## Setting the standard deviation upper bound to 10 (it's so high it has no effect on the optimisation).
rm = SD(; settings = RMSettings(; ub = 10))
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
r1 = calc_risk(port; rm = rm)
## As a functor, we need to provide the covariance matrix to the risk measure directly.
rm.sigma = port.cov
isapprox(r1, rm(w1.weights))
## The value of :sd_risk is consistent with the risk calculation.
isapprox(r1, value(port.model[:sd_risk]))
## The [`SecondOrderCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Second-Order-Cone) constraint.
port.model[:constr_sd_risk_soc]
## The variance risk.
port.model[:sd_risk]
## Variance risk is a [`VariableRef`](https://jump.dev/JuMP.jl/stable/api/JuMP/#VariableRef)
typeof(port.model[:sd_risk])
## sd_risk <= 10
port.model[:sd_risk_ub]

## Optimisations which use [`calc_risk`](@ref) to compute the risk have no [`JuMP`](https://github.com/jump-dev/JuMP.jl) model.
w2 = optimise!(port, HRP(; rm = rm))
