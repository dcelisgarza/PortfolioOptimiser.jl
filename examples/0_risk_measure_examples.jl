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
rm = MAD(; w = ew1, we = ew2)
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
rm = MAD(; w = ew1, we = ew2)
# Hierarchical optimisation, no JuMP model.
w6 = optimise!(port, HRP(; rm = rm))
# Compute the mean absolute deviation.
r6 = calc_risk(port, :HRP; rm = rm)

#=
## Semi Standard Deviation, [`SSD`](@ref)
=#

# Recompute asset statistics.
asset_statistics!(port)
# Vanilla semi standard deviation.
rm = SSD()
# Optimise portfolio.
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute semi standard deviation.
r1 = calc_risk(port; rm = rm)
# Values are consistent.
isapprox(r1, value(port.model[:sdev_risk]))
#=
Semi standard deviation with a returns threshold equal to the maximum return, this should make it equivalent to using the standard deviation.
=#
rm = SSD(; target = maximum(ret))
# Optimise portfolio using the semi standard deviation with a return threshold that includes all returns.
w2 = optimise!(port, Trad(; obj = MinRisk(), rm = rm, str_names = true))
# Optimise portfolio using the standard deviation.
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

# Recompute asset statistics.
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
First lower partial moment with a returns threshold equal to `Inf`, will use `port.mu`in optimisations using [`JuMP`](https://github.com/jump-dev/JuMP.jl) models, and compute the mean of the returns vector when used in the functor.
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

# Recompute asset statistics.
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
Second lower partial moment with a returns threshold equal to `Inf`, will use `port.mu`in optimisations using [`JuMP`](https://github.com/jump-dev/JuMP.jl) models, and compute the mean of the returns vector when used in the functor.
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

#=
# Worst Realisation, [`WR`](@ref)
=#

# Recompute asset statistics.
asset_statistics!(port)
# Worst Realisation.
rm = WR()
# Optimise portfolio.
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute the worst realisation.
r1 = calc_risk(port; rm = rm)
# Values are consistent.
isapprox(r1, value(port.model[:wr_risk]))
# Hierarchical optimisation, no JuMP model.
w2 = optimise!(port, HRP(; rm = rm))
# Compute the worst realisation.
r2 = calc_risk(port, :HRP; rm = rm)
# Use it in conjunction with another, less conservative risk measure.
rm = [WR(; settings = RMSettings(; scale = 0.15)), Variance()]
w3 = optimise!(port, Trad(; rm = rm, str_names = true))
# WR.
r3_1 = calc_risk(port; rm = WR())
# Variance.
r3_2 = calc_risk(port; rm = Variance())
# This portfolio is not optimal in either risk measure, but mixes their characteristics.
w4 = optimise!(port, Trad(; rm = Variance(), str_names = true))
# Minimum variance portfolio.
r4 = calc_risk(port; rm = Variance())
# WR of mixed portfolio is higher than the minimal worst realisation.
r3_1 > r1
# Variance of mixed portfolio is higher than the minimal worst realisation.
r3_2 > r4

#=
# Conditional Value at Risk, [`CVaR`](@ref)
=#

# Recompute asset statistics.
asset_statistics!(port)
# CVaR with default values.
rm = CVaR()
# Optimise portfolio.
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute CVaR for `alpha  = 0.05`.
r1 = calc_risk(port; rm = rm)
# Risk is consistent.
isapprox(r1, value(port.model[:cvar_risk]); rtol = 5e-8)
# CVaR of the worst 50 % of cases.
rm = CVaR(; alpha = 0.5)
# Optimise portfolio.
w2 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute CVaR for `alpha  = 0.5`.
r2 = calc_risk(port; rm = rm)
# Values are consistent.
isapprox(r2, value(port.model[:cvar_risk]))
# CVaR with default values.
rm = CVaR()
# Hierarchical optimisation, no JuMP model.
w3 = optimise!(port, HRP(; rm = rm))
# Compute the CVaR.
r3 = calc_risk(port, :HRP; rm = rm)
# CVaR of the worst 50 % of cases.
rm = CVaR(; alpha = 0.5)
# Hierarchical optimisation, no JuMP model.
w4 = optimise!(port, HRP(; rm = rm))
# Compute the CVaR.
r4 = calc_risk(port, :HRP; rm = rm)

#=
# Entropic Value at Risk, [`EVaR`](@ref)
=#

# Recompute asset statistics.
asset_statistics!(port)
# EVaR with default values.
rm = EVaR()
# Optimise portfolio.
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute EVaR for `alpha  = 0.05`.
r1 = calc_risk(port; rm = rm)
# As a functor, must provide the solvers.
rm.solvers = port.solvers
r1 == rm(port.returns * w1.weights)
# Risk is consistent.
isapprox(r1, value(port.model[:evar_risk]))
# EVaR of the worst 50 % of cases.
rm = EVaR(; alpha = 0.5)
# Optimise portfolio.
w2 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute EVaR for `alpha  = 0.5`.
r2 = calc_risk(port; rm = rm)
# Values are consistent.
isapprox(r2, value(port.model[:evar_risk]))
# EVaR with default values.
rm = EVaR()
# Hierarchical optimisation, no JuMP model but needs solvers.
w3 = optimise!(port, HRP(; rm = rm))
# Compute the EVaR.
r3 = calc_risk(port, :HRP; rm = rm)
# EVaR of the worst 50 % of cases.
rm = EVaR(; alpha = 0.5)
# Hierarchical optimisation, no JuMP model.
w4 = optimise!(port, HRP(; rm = rm))
# Compute the EVaR.
r4 = calc_risk(port, :HRP; rm = rm)

#=
# Relativistic Value at Risk, [`RLVaR`](@ref)
=#

# Recompute asset statistics.
asset_statistics!(port)
# RLVaR with default values.
rm = RLVaR()
# Optimise portfolio.
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute RLVaR for `alpha  = 0.05`.
r1 = calc_risk(port; rm = rm)
# As a functor, must provide the solvers.
rm.solvers = port.solvers
r1 == rm(port.returns * w1.weights)
# Risk is consistent.
isapprox(r1, value(port.model[:rlvar_risk]))
# RLVaR of the worst 50 % of cases.
rm = RLVaR(; alpha = 0.5)
# Optimise portfolio.
w2 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute RLVaR for `alpha  = 0.5`.
r2 = calc_risk(port; rm = rm)
# Values are consistent.
isapprox(r2, value(port.model[:rlvar_risk]))
# Check the limits as `kappa → 0`, and `kappa → Inf`. We use a large value of alpha because there are very few observations, so we need it to differentiate the results of the optimisations.
w3_1 = optimise!(port, Trad(; rm = RLVaR(; alpha = 0.5, kappa = 5e-5), str_names = true))
#
w3_2 = optimise!(port,
                 Trad(; rm = RLVaR(; alpha = 0.5, kappa = 1 - 5e-5), str_names = true))
#
w3_3 = optimise!(port, Trad(; rm = EVaR(; alpha = 0.5), str_names = true))
#
w3_4 = optimise!(port, Trad(; rm = WR(), str_names = true))
# ``\\lim\\limits_{\\kappa \\to 0} \\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\approx \\mathrm{EVaR}(\\bm{X},\\, \\alpha)``
d1 = rmsd(w3_1.weights, w3_3.weights)
# ``\\lim\\limits_{\\kappa \\to 1} \\mathrm{RLVaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\approx \\mathrm{WR}(\\bm{X})``
d2 = rmsd(w3_2.weights, w3_4.weights)
# RLVaR with default values.
rm = RLVaR()
# Hierarchical optimisation, no JuMP model but needs solvers.
w4 = optimise!(port, HRP(; rm = rm))
# Compute the RLVaR.
r4 = calc_risk(port, :HRP; rm = rm)
# RLVaR of the worst 50 % of cases.
rm = RLVaR(; alpha = 0.5)
# Hierarchical optimisation, no JuMP model.
w5 = optimise!(port, HRP(; rm = rm))
# Compute the RLVaR.
r5 = calc_risk(port, :HRP; rm = rm)

#=
# Maximum Drawdown of uncompounded cumulative returns, [`MDD`](@ref)
=#

# Recompute asset statistics.
asset_statistics!(port)
# Maximum drawdown of uncompounded returns.
rm = MDD()
# Optimise portfolio.
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute MDD.
r1 = calc_risk(port; rm = rm)
# Values are consistent.
isapprox(r1, value(port.model[:mdd_risk]))
# Hierarchical optimisation, no JuMP model.
w2 = optimise!(port, HRP(; rm = rm))
# Compute the MDD.
r2 = calc_risk(port, :HRP; rm = rm)
# Use it in conjunction with another, less conservative risk measure.
rm = [MDD(; settings = RMSettings(; scale = 0.15)), Variance()]
w3 = optimise!(port, Trad(; rm = rm, str_names = true))
# MDD.
r3_1 = calc_risk(port; rm = MDD())
# Variance.
r3_2 = calc_risk(port; rm = Variance())
# This portfolio is not optimal in either risk measure, but mixes their characteristics.
w4 = optimise!(port, Trad(; rm = Variance(), str_names = true))
# Minimum variance portfolio.
r4 = calc_risk(port; rm = Variance())
# MDD of mixed portfolio is higher than the minimal MDD.
r3_1 > r1
# Variance of mixed portfolio is higher than the minimal MDD.
r3_2 > r4

#=
# Average Drawdown of uncompounded cumulative returns, [`ADD`](@ref)
=#

# Recompute asset statistics.
asset_statistics!(port)
# Average drawdown of uncompounded returns.
rm = ADD()
# Optimise portfolio.
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute ADD.
r1 = calc_risk(port; rm = rm)
# Values are consistent.
isapprox(r1, value(port.model[:add_risk]))
# Exponentially weighted average drawdown.
ew = eweights(1:size(ret, 1), 0.3; scale = true)
# Average weighted drawdown of uncompounded returns.
rm = ADD(; w = ew)
# Optimise portfolio.
w2 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute ADD.
r2 = calc_risk(port; rm = rm)
# Values are consistent.
isapprox(r2, value(port.model[:add_risk]))
# Average drawdown of uncompounded returns.
rm = ADD()
# Hierarchical optimisation, no JuMP model.
w3 = optimise!(port, HRP(; rm = rm))
# Compute the ADD.
r3 = calc_risk(port, :HRP; rm = rm)
# Average weighted drawdown of uncompounded returns.
rm = ADD(; w = ew)
# Optimise portfolio.
w4 = optimise!(port, HRP(; rm = rm))
# Compute ADD.
r4 = calc_risk(port, :HRP; rm = rm)

#=
# Conditional Drawdown at Risk of uncompounded cumulative returns, [`CDaR`](@ref)
=#

# Recompute asset statistics.
asset_statistics!(port)
# CDaR with default values.
rm = CDaR()
# Optimise portfolio.
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute CDaR for `alpha  = 0.05`.
r1 = calc_risk(port; rm = rm)
# Risk is consistent.
isapprox(r1, value(port.model[:cdar_risk]))
# CDaR of the worst 50 % of cases.
rm = CDaR(; alpha = 0.5)
# Optimise portfolio.
w2 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute CDaR for `alpha  = 0.5`.
r2 = calc_risk(port; rm = rm)
# Values are consistent.
isapprox(r2, value(port.model[:cdar_risk]))
# CDaR with default values.
rm = CDaR()
# Hierarchical optimisation, no JuMP model.
w3 = optimise!(port, HRP(; rm = rm))
# Compute the CDaR.
r3 = calc_risk(port, :HRP; rm = rm)
# CDaR of the worst 50 % of cases.
rm = CDaR(; alpha = 0.5)
# Hierarchical optimisation, no JuMP model.
w4 = optimise!(port, HRP(; rm = rm))
# Compute the CDaR.
r4 = calc_risk(port, :HRP; rm = rm)

#=
# Ulcer Index of uncompounded cumulative returns, [`MDD`](@ref)
=#

# Recompute asset statistics.
asset_statistics!(port)
# Ulcer Index of uncompounded returns.
rm = UCI()
# Optimise portfolio.
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute UCI.
r1 = calc_risk(port; rm = rm)
# Values are consistent.
isapprox(r1, value(port.model[:uci_risk]))
# Hierarchical optimisation, no JuMP model.
w2 = optimise!(port, HRP(; rm = rm))
# Compute the UCI.
r2 = calc_risk(port, :HRP; rm = rm)

#=
# Entropic Drawdown at Risk of uncompounded cumulative returns, [`EDaR`](@ref)
=#

# Recompute asset statistics.
asset_statistics!(port)
# EDaR with default values.
rm = EDaR()
# Optimise portfolio.
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute EDaR for `alpha  = 0.05`.
r1 = calc_risk(port; rm = rm)
# As a functor, must provide the solvers.
rm.solvers = port.solvers
r1 == rm(port.returns * w1.weights)
# Risk is consistent.
isapprox(r1, value(port.model[:edar_risk]))
# EDaR of the worst 50 % of cases.
rm = EDaR(; alpha = 0.5)
# Optimise portfolio.
w2 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute EDaR for `alpha  = 0.5`.
r2 = calc_risk(port; rm = rm)
# Values are consistent.
isapprox(r2, value(port.model[:edar_risk]))
# EDaR with default values.
rm = EDaR()
# Hierarchical optimisation, no JuMP model but needs solvers.
w3 = optimise!(port, HRP(; rm = rm))
# Compute the EDaR.
r3 = calc_risk(port, :HRP; rm = rm)
# EDaR of the worst 50 % of cases.
rm = EDaR(; alpha = 0.5)
# Hierarchical optimisation, no JuMP model.
w4 = optimise!(port, HRP(; rm = rm))
# Compute the EDaR.
r4 = calc_risk(port, :HRP; rm = rm)

#=
# Relativistic Drawdown at Risk of uncompounded cumulative returns, [`RLVaR`](@ref)
=#

# Recompute asset statistics.
asset_statistics!(port)
# RLDaR with default values.
rm = RLDaR()
# Optimise portfolio.
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute RLDaR for `alpha  = 0.05`.
r1 = calc_risk(port; rm = rm)
# As a functor, must provide the solvers.
rm.solvers = port.solvers
r1 == rm(port.returns * w1.weights)
# Risk is consistent.
isapprox(r1, value(port.model[:rldar_risk]))
# RLDaR of the worst 50 % of cases.
rm = RLDaR(; alpha = 0.5)
# Optimise portfolio.
w2 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute RLDaR for `alpha  = 0.5`.
r2 = calc_risk(port; rm = rm)
# Values are consistent.
isapprox(r2, value(port.model[:rldar_risk]))
# Check the limits as `kappa → 0`, and `kappa → Inf`. We use a large value of alpha because there are very few observations, so we need it to differentiate the results of the optimisations.
w3_1 = optimise!(port, Trad(; rm = RLDaR(; alpha = 0.5, kappa = 1e-6), str_names = true))
#
w3_2 = optimise!(port,
                 Trad(; rm = RLDaR(; alpha = 0.5, kappa = 1 - 1e-6), str_names = true))
#
w3_3 = optimise!(port, Trad(; rm = EDaR(; alpha = 0.5), str_names = true))
#
w3_4 = optimise!(port, Trad(; rm = MDD(), str_names = true))
# ``\\lim\\limits_{\\kappa \\to 0} \\mathrm{RLDaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\approx \\mathrm{EVaR}(\\bm{X},\\, \\alpha)``
d1 = rmsd(w3_1.weights, w3_3.weights)
# ``\\lim\\limits_{\\kappa \\to 1} \\mathrm{RLDaR}(\\bm{X},\\, \\alpha,\\, \\kappa) \\approx \\mathrm{WR}(\\bm{X})``
d2 = rmsd(w3_2.weights, w3_4.weights)
# RLDaR with default values.
rm = RLDaR()
# Hierarchical optimisation, no JuMP model but needs solvers.
w4 = optimise!(port, HRP(; rm = rm))
# Compute the RLDaR.
r4 = calc_risk(port, :HRP; rm = rm)
# RLDaR of the worst 50 % of cases.
rm = RLDaR(; alpha = 0.5)
# Hierarchical optimisation, no JuMP model.
w5 = optimise!(port, HRP(; rm = rm))
# Compute the RLDaR.
r5 = calc_risk(port, :HRP; rm = rm)

#=
## Square Root Kurtosis, [`Kurt`](@ref)
=#

# Ensure we use the exact model.
port.max_num_assets_kurt = 0
port.max_num_assets_kurt_scale = 2
# Recompute asset statistics.
asset_statistics!(port)
# Vanilla square root kurtosis.
rm = Kurt()
# Optimise portfolio.
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute semi standard deviation.
r1 = calc_risk(port; rm = rm)
# Values are consistent.
isapprox(r1, value(port.model[:kurt_risk]))
# Exponential weights.
ew = eweights(1:size(ret, 1), 0.2; scale = true)
#=
Compute asset statistics, use `ew` in the `Trad` optimisation. This makes it consistent with the risk measure, because it computes the cokurtosis using this value of `mu`.
=#
asset_statistics!(port; mu_type = MuSimple(; w = ew))
#=
Square root kurtosis with exponential weights. `w` has no effect in the following optimisation, so we account for it in the computation of `port.mu` above.
=#
rm = Kurt(; w = ew)
# Optimise using the exponential weight.
w2 = optimise!(port, Trad(; rm = rm, str_names = true))
#=
Since we used the same exponential weights to compute `port.mu`, and therefore `port.kurt` and passed it on to the functor, the risk computed by `calc_risk` will be consistent with the value in the `JuMP` model.
=#
r2 = calc_risk(port; rm = rm)
# Check they are approximately equal.
isapprox(r2, value(port.model[:kurt_risk]))
# Custom mu (added some random noise).
custom_mu = port.mu + [-0.0025545471368230766, -0.0047554044723918795, 0.010574122455999866,
                       0.0021521845052968917, -0.004417767086053032]
# This won't work even if we use `custom_mu` to compute the cokurtosis, because the expected value is computed inside the functor. We'll also use the `kt` field of the data structure, which takes precedence over `port.kurt` if it is not empty or `nothing`.
rm = Kurt()
rm.kt = cokurt(KurtFull(), port.returns, custom_mu)
# Optimise portfolio using this custom mu.
w3 = optimise!(port, Trad(; rm = rm, str_names = true))
#=
Values don't match because the expected return is computed from the portfolio weights and returns matrix.
=#
r3_1 = calc_risk(port; rm = rm)
#
r3_2 = value(port.model[:kurt_risk])
#=
Both [`Kurt`](@ref) and [`SKurt`](@ref) have approximate formulations for optimisations which use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models, which reduce computational complexity at the cost of accuracy. These are mediated by the `max_num_assets_kurt`, and `max_num_assets_kurt_scale` properties of [`Portfolio`](@ref). Lets make the threshold for using the approximate formulation a single asset (always on), and we will use the largest sets of eigenvalues.
=#
port.max_num_assets_kurt = 1
port.max_num_assets_kurt_scale = 1
# Vanilla square root kurtosis.
rm = Kurt()
# Recompute statistics to reset them.
asset_statistics!(port)
# Run approximate model.
w4 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute the square root kurtosis.
r4 = calc_risk(port, :Trad; rm = rm)
# Because this is an approximate solution, the risk is not minimal.
r4 > r1
# Square root kurtosis.
rm = Kurt(;)
# Hierarchical optimisation, no JuMP model.
w5 = optimise!(port, HRP(; rm = rm))
# Compute the square root kurtosis.
r5 = calc_risk(port, :HRP; rm = rm)
# `w` has an effect in the following optimisation, with no need to recompute `port.mu`.
rm = Kurt(; w = ew)
# Hierarchical optimisation, no JuMP model.
w6 = optimise!(port, HRP(; rm = rm))
# Compute the square root kurtosis.
r6 = calc_risk(port, :HRP; rm = rm)
# The `kt` property of the instance of [`Kurt`](@ref) has no effect in optimisations that don't use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models.
rm.kt = cokurt(KurtFull(), port.returns, custom_mu)
w7 = optimise!(port, HRP(; rm = rm))
w6.weights == w7.weights

#=
## Square Root Semi Kurtosis, [`SKurt`](@ref)
=#

# Ensure we use the exact model.
port.max_num_assets_kurt = 0
port.max_num_assets_kurt_scale = 2
# Recompute asset statistics.
asset_statistics!(port)
# Vanilla square root semi kurtosis.
rm = SKurt(; target = 0.01)
# Optimise portfolio.
w1 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute semi standard deviation.
r1 = calc_risk(port; rm = rm)
# Values are not consistent because [`cokurt`](@ref) tends to produce poorly conditioned matrices even after making them positive definite, more so for [`KurtSemi`](@ref) because it replaces values above the threshold with zero, this is the case in this example where there are few readings and the values fluctuate a lot. This makes the [`JuMP`](https://github.com/jump-dev/JuMP.jl) model difficult to optimise.
isapprox(r1, value(port.model[:skurt_risk]))
# Conditioning number of the semi cokurtosis.
cond(port.skurt)
# Exponential weights.
ew = eweights(1:size(ret, 1), 0.2; scale = true)
#=
Compute asset statistics, use `ew` in the `Trad` optimisation. This makes it consistent with the risk measure, because it computes the cokurtosis using this value of `mu`.
=#
asset_statistics!(port; mu_type = MuSimple(; w = ew))
#=
Square root semi kurtosis with exponential weights. `w` has no effect in the following optimisation, so we account for it in the computation of `port.mu` above.
=#
rm = SKurt(; w = ew)
# Optimise using the exponential weight.
w2 = optimise!(port, Trad(; rm = rm, str_names = true))
#=
Since we used the same exponential weights to compute `port.mu`, and therefore `port.kurt` and passed it on to the functor, the risk computed by `calc_risk` will be consistent with the value in the `JuMP` model.
=#
r2 = calc_risk(port; rm = rm)
# Check they are approximately equal.
isapprox(r2, value(port.model[:kurt_risk]))
# Custom mu (added some random noise).
custom_mu = port.mu + [-0.0025545471368230766, -0.0047554044723918795, 0.010574122455999866,
                       0.0021521845052968917, -0.004417767086053032]
# This won't work even if we use `custom_mu` to compute the cokurtosis, because the expected value is computed inside the functor. We'll also use the `kt` field of the data structure, which takes precedence over `port.kurt` if it is not empty or `nothing`.
rm = SKurt()
rm.kt = cokurt(KurtSemi(), port.returns, custom_mu)
# Optimise portfolio using this custom mu.
w3 = optimise!(port, Trad(; rm = rm, str_names = true))
#=
Values don't match because the expected return is computed from the portfolio weights and returns matrix.
=#
r3_1 = calc_risk(port; rm = rm)
#
r3_2 = value(port.model[:kurt_risk])
#=
Both [`SKurt`](@ref) and [`SKurt`](@ref) have approximate formulations for optimisations which use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models, which reduce computational complexity at the cost of accuracy. These are mediated by the `max_num_assets_kurt`, and `max_num_assets_kurt_scale` properties of [`Portfolio`](@ref). Lets make the threshold for using the approximate formulation a single asset (always on), and we will use the largest sets of eigenvalues.
=#
port.max_num_assets_kurt = 1
port.max_num_assets_kurt_scale = 1
# Vanilla square root semi kurtosis.
rm = SKurt()
# Recompute statistics to reset them.
asset_statistics!(port)
# Run approximate model.
w4 = optimise!(port, Trad(; rm = rm, str_names = true))
# Compute the square root semi kurtosis.
r4 = calc_risk(port, :Trad; rm = rm)
# Because this is an approximate solution, the risk is not minimal.
r4 > r1
# Square root semi kurtosis.
rm = SKurt(;)
# Hierarchical optimisation, no JuMP model.
w5 = optimise!(port, HRP(; rm = rm))
# Compute the square root semi kurtosis.
r5 = calc_risk(port, :HRP; rm = rm)
# `w` has an effect in the following optimisation, with no need to recompute `port.mu`.
rm = SKurt(; w = ew)
# Hierarchical optimisation, no JuMP model.
w6 = optimise!(port, HRP(; rm = rm))
# Compute the square root semi kurtosis.
r6 = calc_risk(port, :HRP; rm = rm)
# The `kt` property of the instance of [`SKurt`](@ref) has no effect in optimisations that don't use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models.
rm.kt = cokurt(KurtSemi(), port.returns, custom_mu)
w7 = optimise!(port, HRP(; rm = rm))
w6.weights == w7.weights
