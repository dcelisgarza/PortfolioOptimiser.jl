#src
#src Copywrite (c) 2025
#src Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
#src SPDX-License-Identifier: MIT

#=
# Risk Measures

This file contains the examples in the docstrings as runnable code.
=#

using PortfolioOptimiser, Clarabel, JuMP, StatsBase

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
port = Portfolio(; ret = ret, assets = 1:size(ret, 2),
                 solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                  :check_sol => (allow_local = true,
                                                                 allow_almost = true),
                                                  :params => Dict("verbose" => false))));
## Compute asset statistics.                                                
asset_statistics!(port)
## Clusterise assets (for hierarchical optimisations).
cluster_assets!(port)

#=
## Standard Deviation, [`SD`](@ref)
=#

# Standard deviation.
rm = SD()
# Optimise portfolio.
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute the standard deviation.
r1 = calc_risk(port, :Trad; rm = rm)
# As a functor.
r1 == SD(; sigma = port.cov)(w1.weights)
# Check that the std risk exists as an SOC constraint.
port.model[:sd_risk]
#
port.model[:constr_sd_risk_soc]

# Hierarchical risk parity optimisation, no JuMP model.
w2 = optimise!(port, HRP(; rm = rm))
# Compute the standard deviation.
r2 = calc_risk(port, :HRP; rm = rm)
# Use SD as a functor.
r2 == SD(; sigma = port.cov)(w2.weights)

#=
## Mean Absolute Deviation, [`MAD`](@ref)
=#

# Vanilla mean absolute deviation.
rm = MAD()
# Optimise portfolio.
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute risk.
r1 = calc_risk(port; rm = rm)
# Check values are similar.
isapprox(r1, value(port.model[:mad_risk]))
# Exponential weights.
ew1 = eweights(1:size(ret, 1), 0.2; scale = true);
ew2 = eweights(1:size(ret, 1), 0.3; scale = true);
#=
Compute asset statistics, use ew1 in the `Trad` optimisation. This makes it consistent with the risk measure.
=#
asset_statistics!(port; mu_type = MuSimple(; w = ew1))
#=
Mean absolute deviation with different weights. w1 has no effect in the following optimisation in [`JuMP`](https://github.com/jump-dev/JuMP.jl)-based optimisations, so we account for it in the computation of `port.mu` above.
=#
rm = MAD(; w1 = ew1, w2 = ew2)
# Use the custom weights in the optimisation.
w2 = optimise!(port, Trad(; rm = rm, str_names = true))
#=
Using `w1` and `w2` may lead to inconsistent values between the functor and value in the model because the mean absolute deviation is formulated with slack constraints.
=#
r2_1 = calc_risk(port; rm = rm)
#
r2_2 = value(port.model[:mad_risk])
# Use a custom mu (added some random noise).
custom_mu = port.mu + [-0.0025545471368230766, -0.0047554044723918795, 0.010574122455999866,
                       0.0021521845052968917, -0.004417767086053032]
rm = MAD(; mu = custom_mu)
# Optimise with the custom mu.
w3 = optimise!(port, Trad(; rm = rm, str_names = true))
# Values don't match.
r3_1 = calc_risk(port; rm = rm)
#
r3_2 = value(port.model[:mad_risk])
# Vanilla mean absolute deviation.
rm = MAD()
# Hierarchical optimisation, no JuMP model.
w4 = optimise!(port, HRP(; rm = rm))
# Compute the mean absolute deviation.
r4 = calc_risk(port, :HRP; rm = rm)
# Use the risk measure as a functor.
r4 == rm(port.returns * w4.weights)
# Custom mu has no effect in the following optimisation.
rm = MAD(; mu = custom_mu)
# Hierarchical optimisation, no JuMP model.
w5 = optimise!(port, HRP(; rm = rm))
w4.weights == w5.weights
# Compute the mean absolute deviation.
r5 = calc_risk(port, :HRP; rm = rm)
# `w1` and `w2` both have effects.
rm = MAD(; w1 = ew1, w2 = ew2)
# Hierarchical optimisation, no JuMP model.
w6 = optimise!(port, HRP(; rm = rm))
# Compute the mean absolute deviation.
r6 = calc_risk(port, :HRP; rm = rm)

#=
## Semi Standard Deviation, [`SSD`](@ref)
=#

# Recompute asset statistics to get rid of the exponentially weighted expected returns vector.
asset_statistics!(port)
# Vanilla semi standard deviation.
rm = SSD()
# Optimise portfolio.
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute semi standard devation.
r1 = calc_risk(port; rm = rm)
# Values are consistent.
isapprox(r1, value(port.model[:sdev_risk]))
#=
Semi standard deviation with a returns threshold equal to the maximum return, this should make it equivalent to using the standard deviation.
=#
rm = SSD(; target = maximum(ret))
# Optimise portfolio using the semi standard deviation with a return threshold that includes all returns.
w2 = optimise!(port, Trad(; obj = MinRisk(), rm = rm, str_names = true))
# Optimise portfolio using the standard devation.
w3 = optimise!(port, Trad(; rm = SD(), str_names = true))
# Value are approximately equal.
isapprox(w2.weights, w3.weights; rtol = 5e-5)
# Exponential weights.
ew = eweights(1:size(ret, 1), 0.2; scale = true)
#=
Compute asset statistics, use `ew` in the `Trad` optimisation. This makes it consistent with the risk measure.
=#
asset_statistics!(port; mu_type = MuSimple(; w = ew))
#=
Semi standard deviation with exponential weights. `w` has no effect in the following optimisation, so we account for it in the computation of `port.mu` above.
=#
rm = SSD(; w = ew)
# Optimise using the exponential weight.
w4 = optimise!(port, Trad(; rm = rm, str_names = true))
#=
Since we used the same exponential weights to compute `port.mu` and passed it on to the functor, the risk computed by `calc_risk` will be consistent with the value in the `JuMP` model.
=#
r4 = calc_risk(port; rm = rm)
# Check they are approximately equal.
isapprox(r4, value(port.model[:sdev_risk]))
# Custom mu (added some random noise).
custom_mu = port.mu + [-0.0025545471368230766, -0.0047554044723918795, 0.010574122455999866,
                       0.0021521845052968917, -0.004417767086053032]
rm = SSD(; mu = custom_mu)
# Optimise portfolio using this custom mu.
w5 = optimise!(port, Trad(; rm = rm, str_names = true))
#=
Values don't match because the mean return is computed from the portfolio weights and returns matrix.
=#
r5_1 = calc_risk(port; rm = rm)
#
r5_2 = value(port.model[:sdev_risk])
# Vanilla semi standard deviation.
rm = SSD()
# Hierarchical optimisation, no JuMP model.
w6 = optimise!(port, HRP(; rm = rm))
# Compute the semi standard deviation.
r6 = calc_risk(port, :HRP; rm = rm)
# As a functor.
r6 == rm(port.returns * w6.weights)
# Custom mu has no effect in the following optimisation.
rm = SSD(; mu = custom_mu)
# Hierarchical optimisation, no JuMP model.
w7 = optimise!(port, HRP(; rm = rm))
w6.weights == w7.weights # true
# Compute the semi standard deviation.
r7 = calc_risk(port, :HRP; rm = rm)
# `w` has an effect in the following optimisation.
rm = SSD(; w = ew)
# Hierarchical optimisation, no JuMP model.
w8 = optimise!(port, HRP(; rm = rm))
# Compute the semi standard deviation.
r8 = calc_risk(port, :HRP; rm = rm)

#=
# First Lower Partial Moment, [`FLPM`](@ref)
=#

# Recompute asset statistics to get rid of the exponentially weighted expected returns vector.
asset_statistics!(port)
# Vanilla first lower partial moment.
rm = FLPM()
# Optimise portfolio.
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute first lower partial moment.
r1 = calc_risk(port; rm = rm)
# Values are consistent.
isapprox(r1, value(port.model[:flpm_risk]))
#=
First lower partial moment with a returns threshold equal to `Inf` will use `rm.mu` (which in this case is zero) in optimisations using [`JuMP`](https://github.com/jump-dev/JuMP.jl) models, and compute the mean of the returns vector when used in the functor.
=#
rm = FLPM(; target = Inf)
# Optimise portfolio using the first lower partial moment with a return threshold that includes all returns.
w2 = optimise!(port, Trad(; obj = MinRisk(), rm = rm, str_names = true))
# The risks do not match. This is because when using the functor, `mu` has no effect, and if `isinf(target)`, it will be set to the expected value of the returns vector. Whereas [`PortfolioOptimiser.set_rm`](@ref) took the value to be `target = range(; start = mu, stop = mu, length = N)`, where `N` is the number of assets, and `mu == 0` in this case.
r2_1 = calc_risk(port; rm = rm)
#
r2_2 = value(port.model[:flpm_risk])
# If we set `rm.target = 0`, then `calc_risk` will compute the correct risk.
rm.target = 0
isapprox(r2_2, calc_risk(port; rm = rm))
#=
# First lower partial moment with a returns threshold equal to `Inf`, will use `port.mu`in optimisations using [`JuMP`](https://github.com/jump-dev/JuMP.jl) models, and compute the mean of the returns vector when used in the functor.
=#
rm = FLPM(; target = Inf, mu = Inf)
# Value are approximately equal.
w3 = optimise!(port, Trad(; obj = MinRisk(), rm = rm, str_names = true))
# Exponential weights.
ew = eweights(1:size(ret, 1), 0.2; scale = true)
#=
Compute asset statistics, use `ew` in the `Trad` optimisation. This makes it consistent with the risk measure.
=#
asset_statistics!(port; mu_type = MuSimple(; w = ew))
#=
First lower partial moment with exponential weights. `w` has no effect in the following optimisation, so we account for it in the computation of `port.mu` above.
=#
rm = FLPM(; w = ew)
# Optimise using the exponential weight.
w4 = optimise!(port, Trad(; rm = rm, str_names = true))
#=
Since we used the same exponential weights to compute `port.mu` and passed it on to the functor, the risk computed by `calc_risk` will be consistent with the value in the `JuMP` model.
=#
r4 = calc_risk(port; rm = rm)
# Check they are approximately equal.
isapprox(r4, value(port.model[:flpm_risk]))
# Custom mu (added some random noise).
custom_mu = port.mu + [-0.0025545471368230766, -0.0047554044723918795, 0.010574122455999866,
                       0.0021521845052968917, -0.004417767086053032]
rm = FLPM(; mu = custom_mu)
# Optimise portfolio using this custom mu.
w5 = optimise!(port, Trad(; rm = rm, str_names = true))
#=
Values don't match because the mean return is computed from the portfolio weights and returns matrix.
=#
r5_1 = calc_risk(port; rm = rm)
#
r5_2 = value(port.model[:flpm_risk])
# Vanilla first lower partial moment.
rm = FLPM()
# Hierarchical optimisation, no JuMP model.
w6 = optimise!(port, HRP(; rm = rm))
# Compute the first lower partial moment.
r6 = calc_risk(port, :HRP; rm = rm)
# As a functor.
r6 == rm(port.returns * w6.weights)
# Custom mu has no effect in the following optimisation.
rm = FLPM(; mu = custom_mu)
# Hierarchical optimisation, no JuMP model.
w7 = optimise!(port, HRP(; rm = rm))
w6.weights == w7.weights # true
# If we set `target = Inf`, the target will be the return vector's expected value computed with the weights.
rm = FLPM(; target = Inf, w = ew)
# Hierarchical optimisation, no JuMP model.
w8 = optimise!(port, HRP(; rm = rm))
# Compute the first lower partial moment.
r8 = calc_risk(port, :HRP; rm = rm)

#=
# Second Lower Partial Moment, [`SLPM`](@ref)
=#

# Recompute asset statistics to get rid of the exponentially weighted expected returns vector.
asset_statistics!(port)
# Vanilla second lower partial moment.
rm = SLPM()
# Optimise portfolio.
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute second lower partial moment.
r1 = calc_risk(port; rm = rm)
# Values are consistent.
isapprox(r1, value(port.model[:slpm_risk]))
#=
Second lower partial moment with a returns threshold equal to `Inf` will use `rm.mu` (which in this case is zero) in optimisations using [`JuMP`](https://github.com/jump-dev/JuMP.jl) models, and compute the mean of the returns vector when used in the functor.
=#
rm = SLPM(; target = Inf)
# Optimise portfolio using the second lower partial moment with a return threshold that includes all returns.
w2 = optimise!(port, Trad(; obj = MinRisk(), rm = rm, str_names = true))
# The risks do not match. This is because when using the functor, `mu` has no effect, and if `isinf(target)`, it will be set to the expected value of the returns vector. Whereas [`PortfolioOptimiser.set_rm`](@ref) took the value to be `target = range(; start = mu, stop = mu, length = N)`, where `N` is the number of assets, and `mu == 0` in this case.
r2_1 = calc_risk(port; rm = rm)
#
r2_2 = value(port.model[:slpm_risk])
# If we set `rm.target = 0`, then `calc_risk` will compute the correct risk.
rm.target = 0
isapprox(r2_2, calc_risk(port; rm = rm))
#=
# Second lower partial moment with a returns threshold equal to `Inf`, will use `port.mu`in optimisations using [`JuMP`](https://github.com/jump-dev/JuMP.jl) models, and compute the mean of the returns vector when used in the functor.
=#
rm = SLPM(; target = Inf, mu = Inf)
# Value are approximately equal.
w3 = optimise!(port, Trad(; obj = MinRisk(), rm = rm, str_names = true))
# Exponential weights.
ew = eweights(1:size(ret, 1), 0.2; scale = true)
#=
Compute asset statistics, use `ew` in the `Trad` optimisation. This makes it consistent with the risk measure.
=#
asset_statistics!(port; mu_type = MuSimple(; w = ew))
#=
Second lower partial moment with exponential weights. `w` has no effect in the following optimisation, so we account for it in the computation of `port.mu` above.
=#
rm = SLPM(; w = ew)
# Optimise using the exponential weight.
w4 = optimise!(port, Trad(; rm = rm, str_names = true))
#=
Since we used the same exponential weights to compute `port.mu` and passed it on to the functor, the risk computed by `calc_risk` will be consistent with the value in the `JuMP` model.
=#
r4 = calc_risk(port; rm = rm)
# Check they are approximately equal.
isapprox(r4, value(port.model[:slpm_risk]))
# Custom mu (added some random noise).
custom_mu = port.mu + [-0.0025545471368230766, -0.0047554044723918795, 0.010574122455999866,
                       0.0021521845052968917, -0.004417767086053032]
rm = SLPM(; mu = custom_mu)
# Optimise portfolio using this custom mu.
w5 = optimise!(port, Trad(; rm = rm, str_names = true))
#=
Values don't match because the mean return is computed from the portfolio weights and returns matrix.
=#
r5_1 = calc_risk(port; rm = rm)
#
r5_2 = value(port.model[:slpm_risk])
# Vanilla second lower partial moment.
rm = SLPM()
# Hierarchical optimisation, no JuMP model.
w6 = optimise!(port, HRP(; rm = rm))
# Compute the second lower partial moment.
r6 = calc_risk(port, :HRP; rm = rm)
# As a functor.
r6 == rm(port.returns * w6.weights)
# Custom mu has no effect in the following optimisation.
rm = SLPM(; mu = custom_mu)
# Hierarchical optimisation, no JuMP model.
w7 = optimise!(port, HRP(; rm = rm))
w6.weights == w7.weights # true
# If we set `target = Inf`, the target will be the return vector's expected value computed with the weights.
rm = SLPM(; target = Inf, w = ew)
# Hierarchical optimisation, no JuMP model.
w8 = optimise!(port, HRP(; rm = rm))
# Compute the second lower partial moment.
r8 = calc_risk(port, :HRP; rm = rm)
