function MultivariateStats.fit(type::PCATarget, X::AbstractMatrix)
    return MultivariateStats.fit(MultivariateStats.PCA, X; type.kwargs...)
end
function MultivariateStats.fit(type::PPCATarget, X::AbstractMatrix)
    return MultivariateStats.fit(MultivariateStats.PPCA, X; type.kwargs...)
end
function prep_dim_red_reg(type::PCAReg, x::DataFrame)
    N = nrow(x)
    X = transpose(Matrix(x))

    X_std = StatsBase.standardize(StatsBase.ZScoreTransform, X; dims = 2)

    model = fit(type.target, X_std)
    Xp = transpose(predict(model, X_std))
    Vp = projection(model)
    x1 = [ones(N) Xp]

    return X, x1, Vp
end
function regression(type::PCAReg, X::AbstractMatrix, x1::AbstractMatrix, Vp::AbstractMatrix,
                    y::AbstractVector)
    avg = vec(if isnothing(type.mean_w)
                  mean(X; dims = 2)
              else
                  mean(X, type.mean_w; dims = 2)
              end)
    sdev = vec(if isnothing(type.std_w)
                   std(type.ve, X; dims = 2)
               else
                   std(type.ve, X, type.std_w; dims = 2)
               end)

    fit_result = GLM.lm(x1, y)
    beta_pc = coef(fit_result)[2:end]

    beta = Vp * beta_pc ./ sdev
    beta0 = mean(y) - dot(beta, avg)
    pushfirst!(beta, beta0)

    return beta
end
"""
    regression(type::PCAReg, x::DataFrame, y::DataFrame)

# Description
"""
function regression(type::PCAReg, x::DataFrame, y::DataFrame)
    features = names(x)
    rows = ncol(y)
    cols = ncol(x) + 1

    loadings = zeros(rows, cols)

    X, x1, Vp = prep_dim_red_reg(type, x)
    for i ∈ axes(loadings, 1)
        beta = regression(type, X, x1, Vp, y[!, i])
        loadings[i, :] .= beta
    end

    return hcat(DataFrame(; tickers = names(y)), DataFrame(loadings, ["const"; features]))
end
function add_best_asset_after_failure_pval(included, namesx, ovec, x, y)
    if isempty(included)
        excluded = setdiff(namesx, included)
        best_pval = Inf
        new_feature = ""

        for i ∈ excluded
            factors = [included; i]
            x1 = [ovec Matrix(x[!, factors])]
            fit_result = GLM.lm(x1, y)
            new_pvals = coeftable(fit_result).cols[4][2:end]

            idx = findfirst(x -> x == i, factors)
            test_pval = new_pvals[idx]
            if best_pval > test_pval
                best_pval = test_pval
                new_feature = i
            end
        end

        @warn("No asset with p-value lower than threshold. Best we can do is $new_feature, with p-value $best_pval.")

        push!(included, new_feature)
    end

    return nothing
end
function regression(::FReg, criterion::PVal, x::DataFrame, y::AbstractVector)
    ovec = ones(length(y))
    namesx = names(x)

    threshold = criterion.threshold

    included = String[]
    pvals = Float64[]
    val = 0.0
    while val <= threshold
        excluded = setdiff(namesx, included)
        best_pval = Inf
        new_feature = ""

        for i ∈ excluded
            factors = [included; i]
            x1 = [ovec Matrix(x[!, factors])]
            fit_result = GLM.lm(x1, y)
            new_pvals = coeftable(fit_result).cols[4][2:end]

            idx = findfirst(x -> x == i, factors)
            test_pval = new_pvals[idx]
            if best_pval > test_pval && maximum(new_pvals) <= threshold
                best_pval = test_pval
                new_feature = i
                pvals = copy(new_pvals)
            end
        end

        isempty(new_feature) ? break : push!(included, new_feature)
        if !isempty(pvals)
            val = maximum(pvals)
        end
    end

    add_best_asset_after_failure_pval(included, namesx, ovec, x, y)

    return included
end
function regression(::BReg, criterion::PVal, x::DataFrame, y::AbstractVector)
    ovec = ones(length(y))
    fit_result = GLM.lm([ovec Matrix(x)], y)

    included = names(x)
    namesx = names(x)

    threshold = criterion.threshold

    excluded = String[]
    pvals = coeftable(fit_result).cols[4][2:end]
    val = maximum(pvals)

    while val > threshold
        factors = setdiff(namesx, excluded)
        included = factors

        if isempty(factors)
            break
        end

        x1 = [ovec Matrix(x[!, factors])]
        fit_result = GLM.lm(x1, y)
        pvals = coeftable(fit_result).cols[4][2:end]

        val, idx2 = findmax(pvals)
        push!(excluded, factors[idx2])
    end

    add_best_asset_after_failure_pval(included, namesx, ovec, x, y)

    return included
end
function regression_criterion_func(::AIC)
    return GLM.aic
end
function regression_criterion_func(::AICC)
    return GLM.aicc
end
function regression_criterion_func(::BIC)
    return GLM.bic
end
function regression_criterion_func(::RSq)
    return GLM.r2
end
function regression_criterion_func(::AdjRSq)
    return GLM.adjr2
end
function regression_threshold(::AIC)
    return Inf
end
function regression_threshold(::AICC)
    return Inf
end
function regression_threshold(::BIC)
    return Inf
end
function regression_threshold(::RSq)
    return -Inf
end
function regression_threshold(::AdjRSq)
    return -Inf
end
function get_forward_reg_incl_excl!(::MinValStepwiseRegressionCriteria, value, excluded,
                                    included, threshold)
    val, key = findmin(value)
    idx = findfirst(x -> x == key, excluded)
    if val < threshold
        push!(included, popat!(excluded, idx))
        threshold = val
    end
    return threshold
end
function get_forward_reg_incl_excl!(::MaxValStepwiseRegressionCriteria, value, excluded,
                                    included, threshold)
    val, key = findmax(value)
    idx = findfirst(x -> x == key, excluded)
    if val > threshold
        push!(included, popat!(excluded, idx))
        threshold = val
    end
    return threshold
end
function regression(::FReg, criterion::StepwiseRegressionCriteria, x::DataFrame,
                    y::AbstractVector)
    ovec = ones(length(y))
    namesx = names(x)

    criterion_func = regression_criterion_func(criterion)
    threshold = regression_threshold(criterion)

    included = String[]
    excluded = namesx
    for _ ∈ eachindex(y)
        ni = length(excluded)
        value = Dict()

        for i ∈ excluded
            factors = copy(included)
            push!(factors, i)

            x1 = [ovec Matrix(x[!, factors])]
            fit_result = GLM.lm(x1, y)

            value[i] = criterion_func(fit_result)
        end

        if isempty(value)
            break
        end

        threshold = get_forward_reg_incl_excl!(criterion, value, excluded, included,
                                               threshold)

        if ni == length(excluded)
            break
        end
    end

    return included
end
function get_backward_reg_incl!(::MinValStepwiseRegressionCriteria, value, included,
                                threshold)
    val, idx = findmin(value)
    if val < threshold
        i = findfirst(x -> x == idx, included)
        popat!(included, i)
        threshold = val
    end
    return threshold
end
function get_backward_reg_incl!(::MaxValStepwiseRegressionCriteria, value, included,
                                threshold)
    val, idx = findmax(value)
    if val > threshold
        i = findfirst(x -> x == idx, included)
        popat!(included, i)
        threshold = val
    end
    return threshold
end
function regression(::BReg, criterion::StepwiseRegressionCriteria, x::DataFrame,
                    y::AbstractVector)
    ovec = ones(length(y))
    fit_result = GLM.lm([ovec Matrix(x)], y)

    included = names(x)

    criterion_func = regression_criterion_func(criterion)
    threshold = criterion_func(fit_result)

    for _ ∈ eachindex(y)
        ni = length(included)
        value = Dict()
        for (i, factor) ∈ pairs(included)
            factors = copy(included)
            popat!(factors, i)
            if !isempty(factors)
                x1 = [ovec Matrix(x[!, factors])]
            else
                x1 = reshape(ovec, :, 1)
            end
            fit_result = GLM.lm(x1, y)
            value[factor] = criterion_func(fit_result)
        end

        if isempty(value)
            break
        end

        threshold = get_backward_reg_incl!(criterion, value, included, threshold)

        if ni == length(included)
            break
        end
    end

    return included
end
function regression(type::StepwiseRegression, x::DataFrame, y::DataFrame)
    features = names(x)
    rows = ncol(y)
    cols = ncol(x) + 1

    N = nrow(y)
    ovec = ones(N)

    loadings = zeros(rows, cols)

    for i ∈ axes(loadings, 1)
        included = regression(type, type.criterion, x, y[!, i])

        x1 = !isempty(included) ? [ovec Matrix(x[!, included])] : reshape(ovec, :, 1)

        fit_result = GLM.lm(x1, y[!, i])

        params = coef(fit_result)

        loadings[i, 1] = params[1]
        if isempty(included)
            continue
        end
        idx = [findfirst(x -> x == i, features) + 1 for i ∈ included]
        loadings[i, idx] .= params[2:end]
    end

    return hcat(DataFrame(; tickers = names(y)), DataFrame(loadings, ["const"; features]))
end
"""
```
loadings_matrix(x::DataFrame, y::DataFrame, type::RegressionType = FReg())
```
"""
function loadings_matrix(x::DataFrame, y::DataFrame, type::RegressionType = FReg())
    return regression(type, x, y)
end
function set_noposdef(::NoPosdef, ::Any)
    return nothing
end
function set_noposdef(::Any, cov_type)
    old_posdef = cov_type.posdef
    cov_type.posdef = NoPosdef()
    return old_posdef
end
function set_factor_posdef_cov_type(cov_type::PosdefFixCovCor)
    return set_noposdef(cov_type.posdef, cov_type)
end
function set_factor_posdef_cov_type(::Any)
    return nothing
end
function reset_posdef_cov_type(cov_type::PosdefFixCovCor, sigma)
    return posdef_fix!(cov_type.posdef, sigma)
end
function reset_posdef_cov_type(args...)
    return nothing
end
function risk_factors(x::DataFrame, y::DataFrame; factor_type::FactorType = FactorType(),
                      cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                      mu_type::MeanEstimator = MuSimple())
    B = factor_type.B

    if isnothing(B)
        B = regression(factor_type.type, x, y)
    end
    namesB = names(B)
    old_posdef = nothing
    x1 = if "const" ∈ namesB
        old_posdef = set_factor_posdef_cov_type(cov_type)
        [ones(nrow(y)) Matrix(x)]
    else
        Matrix(x)
    end
    B_mtx = Matrix(B[!, setdiff(namesB, ("tickers",))])

    f_cov, f_mu = sigma_mu(x1, cov_type, mu_type)

    if !isnothing(old_posdef)
        cov_type.posdef = old_posdef
        f_cov2 = f_cov[2:end, 2:end]
        posdef_fix!(cov_type.posdef, f_cov2)
        f_cov[2:end, 2:end] .= f_cov2
    end

    returns = x1 * transpose(B_mtx)
    mu = B_mtx * f_mu

    sigma = if factor_type.error
        e = Matrix(y) - returns
        S_e = diagm(vec(if isnothing(factor_type.var_w)
                            var(factor_type.ve, e; dims = 1)
                        else
                            var(factor_type.ve, e, factor_type.var_w; dims = 1)
                        end))
        B_mtx * f_cov * transpose(B_mtx) + S_e
    else
        B_mtx * f_cov * transpose(B_mtx)
    end

    reset_posdef_cov_type(cov_type, sigma)

    return mu, sigma, returns, B
end

function factor_statistics(assets, returns, f_assets, f_returns;
                           factor_type::FactorType = FactorType(),
                           cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                           mu_type::MeanEstimator = MuSimple())
    f_cov, f_mu = sigma_mu(f_returns, cov_type, mu_type)

    fm_mu, fm_cov, fm_returns, loadings = risk_factors(DataFrame(f_returns, f_assets),
                                                       DataFrame(returns,
                                                                 if any(eltype(assets) .<:
                                                                        (AbstractString,
                                                                         Symbol))
                                                                     assets
                                                                 else
                                                                     Symbol.(assets)
                                                                 end);
                                                       factor_type = factor_type,
                                                       cov_type = cov_type,
                                                       mu_type = mu_type)

    return f_cov, f_mu, fm_mu, fm_cov, fm_returns, loadings
end

export prep_dim_red_reg, regression, loadings_matrix, risk_factors, factor_statistics
