The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).

```@meta
EditURL = "../../../examples/0_risk_measure_examples.jl"
```

# Risk Measures

This file contains the examples in the docstrings as runnable code.

````@example 0_risk_measure_examples
using PortfolioOptimiser, HiGHS, Clarabel, Pajarito, JuMP, StatsBase

# Randomly generated normally distributed returns.
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

# Instantiate portfolio instance.
port = Portfolio(; ret = ret, assets = [:A, :B, :C, :D, :E],
                 solvers = PortOptSolver(; name = :PClGL,
                                         solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                                            "verbose" => false,
                                                                            "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                     MOI.Silent() => true),
                                                                            "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                        "verbose" => false,
                                                                                                                        "max_step_fraction" => 0.75))));

# Compute asset statistics.
asset_statistics!(port)
# Clusterise assets (for hierarchical optimisations).
cluster_assets!(port)
````

# Variance, [`Variance`](@ref)

````@example 0_risk_measure_examples
# If `sigma` is not `nothing` it must be a square matrix. This works at instantiation and runtime.
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

# Default formulation.
rm = Variance()
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
r1 = calc_risk(port; rm = rm)
# As a functor, we need to provide the covariance matrix to the risk measure directly.
rm.sigma = port.cov
isapprox(r1, rm(w1.weights))
# The value of :variance_risk is consistent with the risk calculation.
isapprox(r1, value(port.model[:variance_risk]))
# The [`SecondOrderCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Second-Order-Cone) constraint.
port.model[:constr_dev_soc]
# The variance risk.
port.model[:variance_risk]
# Variance risk is a [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr)
typeof(port.model[:variance_risk])

# Incompatible with [`NOC`](@ref)
w2 = optimise!(port, NOC(; rm = rm, str_names = true))

# Quadratic formulation.
rm = Variance(; formulation = Quad())
w3 = optimise!(port, Trad(; rm = rm, str_names = true))
port.model[:constr_dev_soc]
port.model[:variance_risk]
# Variance risk is a [`QuadExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#QuadExpr)
typeof(port.model[:variance_risk])

# Incompatible with [`NOC`](@ref)
w4 = optimise!(port, NOC(; rm = rm, str_names = true))

# If either `network_adj` or `cluster_adj` field of the [`Portfolio`](@ref) instance is [`SDP`](@ref), the formulation has no effect because this constraint type requires a [`PSDCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Positive-Semidefinite-Cone) formulation of the variance.

A = connection_matrix(port)
B = cluster_matrix(port)

port.cluster_adj = SDP(; A = B)

rm = Variance()
w5 = optimise!(port, Trad(; rm = rm, str_names = true))
r5 = calc_risk(port; rm = rm)
port.model[:constr_M_PSD]
port.model[:variance_risk]
typeof(port.model[:variance_risk])

# Compatible with [`NOC`](@ref)
w6 = optimise!(port, NOC(; rm = rm, str_names = true))
# The risk of the [`NOC`](@ref) optimisation is higher than the optimal value.
r6 = calc_risk(port, :NOC; rm = rm)
r5 <= r6

# No adjacency constraints.
port.cluster_adj = NoAdj()

# Default formulation, with a variance upper bound of 10, standard deviaion of sqrt(10) = 3.1622776601683795.
rm = Variance(; settings = RMSettings(; ub = 10))
w7 = optimise!(port, Trad(; rm = rm, str_names = true))
# dev <= sqrt(10)
port.model[:dev_ub]

# Quadratic formulation, with a variance upper bound of 10, standard deviaion of sqrt(10) = 3.1622776601683795.
rm = Variance(; formulation = Quad(), settings = RMSettings(; ub = 10))
w8 = optimise!(port, Trad(; rm = rm, str_names = true))
# dev <= sqrt(10)
port.model[:dev_ub]

# We use an SDP constraint, this time on the network adjacency, to show that the formulation and risk upper bound change accordingly.
port.network_adj = SDP(; A = A)
rm = Variance(; formulation = Quad(), settings = RMSettings(; ub = 10))
w9 = optimise!(port, Trad(; rm = rm, str_names = true))
port.model[:constr_M_PSD]
port.model[:variance_risk]
typeof(port.model[:variance_risk])
# variance_risk <= 10
port.model[:variance_risk_ub]

# No adjacency constraints.
port.network_adj = NoAdj()

# Optimisations which use [`calc_risk`](@ref) to compute the risk have no [`JuMP`](https://github.com/jump-dev/JuMP.jl) model, therefore the formulation has no effect.
w10 = optimise!(port, HRP(; rm = rm))
````

# Standard Deviation, [`SD`](@ref)

````@example 0_risk_measure_examples
# If `sigma` is not `nothing` it must be a square matrix. This works at instantiation and runtime.
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

# Setting the standard deviation upper bound to 10 (it's so high it has no effect on the optimisation).
rm = SD(; settings = RMSettings(; ub = 10))
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
r1 = calc_risk(port; rm = rm)
# As a functor, we need to provide the covariance matrix to the risk measure directly.
rm.sigma = port.cov
isapprox(r1, rm(w1.weights))
# The value of :sd_risk is consistent with the risk calculation.
isapprox(r1, value(port.model[:sd_risk]))
# The [`SecondOrderCone`](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Second-Order-Cone) constraint.
port.model[:constr_sd_risk_soc]
# The variance risk.
port.model[:sd_risk]
# Variance risk is a [`VariableRef`](https://jump.dev/JuMP.jl/stable/api/JuMP/#VariableRef)
typeof(port.model[:sd_risk])
# sd_risk <= 10
port.model[:sd_risk_ub]

# Optimisations which use [`calc_risk`](@ref) to compute the risk have no [`JuMP`](https://github.com/jump-dev/JuMP.jl) model.
w2 = optimise!(port, HRP(; rm = rm))
````

# Mean Absolute Deviation, [`MAD`](@ref)

````@example 0_risk_measure_examples
# Setting the mean absolute deviation upper bound to 10 (it's so high it has no effect on the optimisation).
rm = MAD(; settings = RMSettings(; ub = 10))
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
r1 = calc_risk(port; rm = rm)
# As a functor.
isapprox(r1, rm(port.returns, w1.weights))
# The value of :mad_risk is consistent with the risk calculation.
isapprox(r1, value(port.model[:mad_risk]))
# MAD risk is a [`AffExpr`](https://jump.dev/JuMP.jl/stable/api/JuMP/#AffExpr)
typeof(port.model[:mad_risk])
# mad_risk <= 10
port.model[:mad_risk_ub]
# `w` has no effect on the optimisation, but can lead to inconsistent values between the optimisation and functor.
ew1 = eweights(1:size(port.returns, 1), 0.5; scale = true)
rm = MAD(; w = ew1)
w2 = optimise!(port, Trad(; rm = rm, str_names = true))
# No effect int he optimisation.
isequal(w1.weights, w2.weights)
# Risk is not consistent with the one in `:mad_risk`, because `w` is not used in the optimisation.
r2 = calc_risk(port; rm = rm)
isapprox(r2, value(port.model[:mad_risk]))
# In order to make them consistent, we can compute the value of `mu` using `w`.
asset_statistics!(port; mu_type = MuSimple(; w = ew1))
w3 = optimise!(port, Trad(; rm = rm, str_names = true))
isapprox(w1.weights, w3.weights)
r3 = calc_risk(port; rm = rm)
isapprox(r3, value(port.model[:mad_risk]))
# Alternatively we can provide this value of `mu` to the risk measure, which takes precedence over the value in `port`. We reset the asset statistics.
rm = MAD(; mu = port.mu)
# Reset the asset statistics to show that the value in `rm` takes precedence.
asset_statistics!(port)
w4 = optimise!(port, Trad(; rm = rm, str_names = true))
r4 = calc_risk(port; rm = rm)
isapprox(r4, value(port.model[:mad_risk]))
# Using the value of `mu` leads to the same results as using the `w` used to compute it.
isapprox(w3.weights, w4.weights)
isapprox(r3, r4)
# We can use a different weights vector for the expected value of the absolute deviations.
ew2 = eweights(1:size(port.returns, 1), 0.7; scale = true)
rm = MAD(; we = ew2)
w5 = optimise!(port, Trad(; rm = rm, str_names = true))
r5 = calc_risk(port; rm = rm)
isapprox(r5, value(port.model[:mad_risk]))
# We can use both.
asset_statistics!(port; mu_type = MuSimple(; w = ew1))
rm = MAD(; mu = port.mu, we = ew2)
w6 = optimise!(port, Trad(; rm = rm, str_names = true))
r6 = calc_risk(port; rm = rm)
isapprox(r6, value(port.model[:mad_risk]))

# Hierarchical optimisation
asset_statistics!(port)
rm = MAD()
w6 = optimise!(port, HRP(; rm = rm))
r6 = calc_risk(port; rm = rm)
# Using `w`.
rm = MAD(; w = ew1)
w7 = optimise!(port, HRP(; rm = rm))
r7 = calc_risk(port; rm = rm)
# Using `mu` isntead.
asset_statistics!(port; mu_type = MuSimple(; w = ew1))
rm = MAD(; mu = port.mu)
w8 = optimise!(port, HRP(; rm = rm))
r8 = calc_risk(port; rm = rm)
````

* * *

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
