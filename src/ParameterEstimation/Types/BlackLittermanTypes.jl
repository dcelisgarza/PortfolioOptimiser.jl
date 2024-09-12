abstract type BlackLitterman end

"""
```
abstract type BlackLittermanFactor <: BlackLitterman end
```

Abstract type for subtyping Black Litterman models.
"""
abstract type BlackLittermanFactor <: BlackLitterman end

"""
```
@kwdef mutable struct BLType{T1 <: Real} <: BlackLitterman
    eq::Bool = true
    delta::Union{<:Real, Nothing} = 1.0
    rf::T1 = 0.0
    posdef::PosdefFix = PosdefNearest()
    denoise::Denoise = NoDenoise()
    logo::AbstractLoGo = NoLoGo()
end
```

Defines the parameters for computing the Black-Litterman model [`black_litterman`](@ref). We define `N` as the number of assets, and `Nv` the number of asset views.

```math
\\begin{align}
\\bm{\\Pi} &= \\begin{cases}
                    \\delta \\mathbf{\\Sigma} \\bm{w} &\\quad \\mathrm{if~ eq = true}\\\\
                      \\bm{\\mu} - r &\\quad \\mathrm{if~ eq = false}
                  \\end{cases}\\\\                            
\\mathbf{\\Omega} &= \\tau \\mathrm{Diagonal}\\left(\\mathbf{P} \\mathbf{\\Sigma} \\mathbf{P}^{\\intercal}\\right)\\\\
\\mathbf{M} &= \\left[ \\left(\\tau  \\mathbf{\\Sigma} \\right)^{-1} + \\mathbf{P}^{\\intercal} \\mathbf{\\Omega}^{-1} \\mathbf{P}\\right]^{-1}\\\\
\\bm{\\Pi}_{\\mathrm{BL}} &= \\mathbf{M} \\left[\\left(\\tau \\mathbf{\\Sigma}\\right)^{-1} \\bm{\\Pi} + \\mathbf{P}^{\\intercal} \\mathbf{\\Omega}^{-1} \\mathbf{Q} \\right]\\\\
\\tau &= \\dfrac{1}{T}\\\\
\\bm{\\mu}_{\\mathrm{BL}} &= \\bm{\\Pi}_{\\mathrm{BL}} + r\\\\
\\mathbf{\\Sigma}_{\\mathrm{BL}} &= \\mathbf{\\Sigma} + \\mathbf{M}
\\end{align}
```

Where:

  - ``\\bm{\\Pi}``:

      + if error is true: is `N×1` the equilibrium excess returns vector.
      + else: is `N×1` the historical excess returns vector.

  - ``\\delta``: is the risk aversion parameter.
  - ``\\mathbf{\\Sigma}``: is the `N×N` asset covariance matrix.
  - ``\\bm{w}``: is the `N×1` vector of benchmark asset weights.
  - ``\\mathbf{P}``: is the `Nv×N` asset views matrix.
  - ``\\bm{Q}``: is the `Nv×1` asset views returns vector.
  - ``\\mathbf{\\Omega}``: is the `Nv×Nv` covariance matrix of the errors of the asset views.
  - ``\\mathbf{M}``: is an `N×N` intermediate covariance matrix, and `M` the number of assets.
  - ``\\bm{\\Pi}_{\\mathbf{BL}}``: is the `N×1` equilibrium excess returns after being adjusted by the views.
  - ``T``: is the number of returns observations.
  - ``\\bm{\\mu}_{\\mathbf{BL}}``: is the `N×1` vector of asset expected returns obtained via the Black-Litterman model.
  - ``\\mathbf{\\Sigma}_{\\mathrm{BL}}``: is the `N×N` asset covariance matrix obtained via the Black-Litterman model.

# Parameters

  - `eq`:

      + if `true`: use the equilibrium excess returns vector.
      + else: use the historical excess returns vector.

  - `delta`: risk aversion factor.
  - `rf`: risk free rate.
  - `posdef`: method for fixing non positive Black-Litterman matrices [`PosdefFix`](@ref).
  - `denoise` method for denoising the Black-Litterman covariance matrix [`Denoise`](@ref).
  - `logo`: method for computing the LoGo Black-Litterman covariance matrix [`AbstractLoGo`](@ref).
"""
mutable struct BLType{T1 <: Real} <: BlackLitterman
    eq::Bool
    delta::Union{<:Real, Nothing}
    rf::T1
    posdef::PosdefFix
    denoise::Denoise
    logo::AbstractLoGo
end
function BLType(; eq::Bool = true, delta::Union{<:Real, Nothing} = 1.0, rf::Real = 0.0,
                posdef::PosdefFix = PosdefNearest(), denoise::Denoise = NoDenoise(),
                logo::AbstractLoGo = NoLoGo())
    return BLType{typeof(rf)}(eq, delta, rf, posdef, denoise, logo)
end

"""
```
@kwdef mutable struct ABLType{T1 <: Real} <: BlackLittermanFactor
    constant::Bool = true
    eq::Bool = true
    delta::Union{<:Real, Nothing} = 1.0
    rf::T1 = 0.0
    posdef::PosdefFix = PosdefNearest()
    denoise::Denoise = NoDenoise()
    logo::AbstractLoGo = NoLoGo()
end
```

Defines the parameters for computing the Augmented Black-Litterman factor model [`black_litterman`](@ref). We define `Na` as the number of assets, `Nva` the number of asset views, `Nf` As the number of factors, and `Nvf` the number of factor views.

```math
\\begin{align*}
\\bm{\\Pi}_{a} &= \\begin{cases}
                    \\delta\\begin{bmatrix}
                      \\mathbf{\\Sigma}\\\\
                      \\mathbf{\\Sigma}_{F} \\mathbf{B}^{\\intercal}
                      \\end{bmatrix} \\bm{w} &\\quad \\mathrm{if~ eq = true}\\\\
                      \\bm{\\mu} - r &\\quad \\mathrm{if~ eq = false}
                  \\end{cases}\\\\
\\mathbf{P}_{a} &=  \\begin{bmatrix}
                      \\mathbf{P} & \\mathbf{0}\\\\
                      \\mathbf{0} & \\mathbf{P}_{F}
                    \\end{bmatrix}\\\\
\\bm{Q}_{a} &=  \\begin{bmatrix}
                \\bm{Q}\\\\
                \\bm{Q}_{F}
                \\end{bmatrix}\\\\
\\mathbf{\\Sigma}_{a} &=  \\begin{bmatrix}
                            \\mathbf{\\Sigma} & \\mathbf{B} \\mathbf{\\Sigma}_{F}\\\\
                            \\mathbf{\\Sigma}_{F} \\mathbf{B}^{\\intercal} & \\mathbf{\\Sigma}_{F}
                          \\end{bmatrix}\\\\
\\mathbf{\\Omega}_{a} &=  \\begin{bmatrix}
                            \\mathbf{\\Omega} & \\mathbf{0}\\\\
                            \\mathbf{0} & \\mathbf{\\Omega}_{F}
                          \\end{bmatrix}\\\\
\\mathbf{\\Omega} &= \\tau \\mathrm{Diagonal}\\left(\\mathbf{P} \\mathbf{\\Sigma} \\mathbf{P}^{\\intercal}\\right)\\\\
\\mathbf{\\Omega}_{F} &= \\tau \\mathrm{Diagonal}\\left(\\mathbf{P}_{F} \\mathbf{\\Sigma}_{F} \\mathbf{P}_{F}^{\\intercal}\\right)\\\\
\\mathbf{M}_{a} &= \\left[ \\left(\\tau  \\mathbf{\\Sigma}_{a} \\right)^{-1} + \\mathbf{P}_{a}^{\\intercal} \\mathbf{\\Omega}_{a}^{-1} \\mathbf{P}_{a}\\right]^{-1}\\\\
\\bm{\\Pi}_{\\mathrm{ABL}} &= \\mathbf{M}_{a} \\left[\\left(\\tau \\mathbf{\\Sigma}_{a}\\right)^{-1} \\bm{\\Pi}_{a} + \\mathbf{P}_{a}^{\\intercal} \\mathbf{\\Omega}_{a}^{-1} \\mathbf{Q}_{a} \\right]\\\\
\\tau &= \\dfrac{1}{T}\\\\
\\bm{\\mu}_{\\mathrm{ABL}} &= \\bm{\\Pi}_{\\mathrm{ABL}} + r\\\\
\\mathbf{\\Sigma}_{\\mathrm{ABL}} &= \\mathbf{\\Sigma}_{a} + \\mathbf{M}_{a}
\\end{align*}
```

Where:

  - ``\\bm{\\Pi}_{a}``:

      + if error is true: is the `Na×1` augmented equilibrium excess returns vector.
      + else: is the `Na×1` historical excess returns vector.

  - ``\\delta``: is the risk aversion parameter.
  - ``\\mathbf{\\Sigma}``: is the `Na×Na` asset covariance matrix.
  - ``\\mathbf{\\Sigma}_{F}``: is the `Nf×Nf` factor covariance matrix.
  - ``\\bm{w}``: is the `Na×1` vector of benchmark asset weights.
  - ``\\mathbf{P}_{a}``: is the `(Nva+Nvf)×(Na+Nf)` augmented views matrix. The zeros pad the matrix so all columns and rows are of equal length.
  - ``\\mathbf{P}``: is the `Nva×Na` asset views matrix.
  - ``\\mathbf{P}_{F}``: is the `Nvf×Nf` factor views matrix.
  - ``\\bm{Q}_{a}``: is the `(Nva+Nvf)×1` augmented views returns vector.
  - ``\\bm{Q}``: is the `Nva×1` asset views returns vector.
  - ``\\bm{Q}_{F}``: is the `Nvf×1` factor views returns vector.
  - ``\\mathbf{\\Sigma}_{a}``: is the `(Na+Nf)×(Na+Nf)` augmented covariance matrix.
  - ``\\mathbf{B}``: is the `Na×Nf` loadings matrix.
  - ``\\mathbf{\\Omega}_{a}``: is the `(Nva+Nvf)×(Nva+Nvf)` covariance matrix of the errors of the augmented views.
  - ``\\mathbf{\\Omega}``: is the `Nva×Nva` covariance matrix of the errors of the asset views.
  - ``\\mathbf{\\Omega}_{F}``: is the `Nvf×Nvf` covariance matrix of the errors of the factor views.
  - ``\\mathbf{M}_{a}``: is an `(Na+Nf)×(Na+Nf)` intermediate covariance matrix.
  - ``\\bm{\\Pi}_{\\mathbf{ABL}}``: is the `Na×1` equilibrium excess returns vector after being adjusted by the augmented views.
  - ``T``: is the number of returns observations.
  - ``\\bm{\\mu}_{\\mathbf{ABL}}``: is the `Na×1` vector of asset expected returns obtained via the Augmented Black-Litterman model.
  - ``\\mathbf{\\Sigma}_{\\mathrm{ABL}}``: is the `Na×Na` asset covariance matrix obtained via the Augmented Black-Litterman model.

# Parameters

  - `eq`:

      + if `true`: use the equilibrium excess returns vector.
      + else: use the historical excess returns vector.

  - `delta`: risk aversion factor.
  - `rf`: risk free rate.
  - `posdef`: method for fixing non positive Augmented Black-Litterman matrices [`PosdefFix`](@ref).
  - `denoise` method for denoising the Augmented Black-Litterman covariance matrix [`Denoise`](@ref).
  - `logo`: method for computing the LoGo Augmented Black-Litterman covariance matrix [`AbstractLoGo`](@ref).
"""
mutable struct ABLType{T1 <: Real} <: BlackLittermanFactor
    constant::Bool
    eq::Bool
    delta::Union{<:Real, Nothing}
    rf::T1
    posdef::PosdefFix
    denoise::Denoise
    logo::AbstractLoGo
end
function ABLType(; constant::Bool = true, eq::Bool = true,
                 delta::Union{<:Real, Nothing} = 1.0, rf::Real = 0.0,
                 posdef::PosdefFix = PosdefNearest(), denoise::Denoise = NoDenoise(),
                 logo::AbstractLoGo = NoLoGo())
    return ABLType{typeof(rf)}(constant, eq, delta, rf, posdef, denoise, logo)
end

"""
```
mutable struct BBLType{T1 <: Real} <: BlackLittermanFactor
    constant::Bool
    error::Bool
    delta::Union{<:Real, Nothing}
    rf::T1
    ve::StatsBase.CovarianceEstimator
    var_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
    denoise::Denoise
    logo::AbstractLoGo
end
```

Defines the parameters for computing the Bayesian Black-Litterman factor model [`black_litterman`](@ref). We define `Na` as the number of assets, `Nva` the number of asset views, `Nf` As the number of factors, and `Nvf` the number of factor views.

```math
\\begin{align*}
\\mathbf{\\Sigma} &= \\mathbf{B} \\mathbf{\\Sigma}_{F} \\mathbf{B}^{\\intercal} + \\mathbf{\\Sigma}_{\\epsilon} \\\\
\\mathbf{\\Sigma}_{\\epsilon} &= \\begin{cases}\\mathrm{Diagonal}\\left(\\mathrm{var}\\left(\\mathbf{X} - \\mathbf{F} \\mathbf{B}^{\\intercal},\\, \\mathrm{dims} = 1\\right)\\right) &\\quad \\mathrm{if~ error = true}\\\\
\\mathbf{0} &\\quad \\mathrm{if~ error = false}
\\end{cases}\\\\
\\overline{\\mathbf{\\Sigma}}_{F} &= \\left(\\mathbf{\\Sigma}_{F}^{-1} + \\mathbf{P}_{F}^{\\intercal} \\mathbf{\\Omega}_{F}^{-1} \\mathbf{P}_{F}\\right)^{-1}\\\\
\\mathbf{\\Omega}_{F} &= \\tau \\mathrm{Diagonal}\\left(\\mathbf{P}_{F} \\mathbf{\\Sigma}_{F} \\mathbf{P}_{F}^{\\intercal}\\right)\\\\
\\tau &= \\dfrac{1}{T}\\\\
\\overline{\\bm{\\Pi}}_{F} &= \\overline{\\mathbf{\\Sigma}}_{F} \\left(\\mathbf{\\Sigma}_{F}^{-1} \\bm{\\Pi}_{F} + \\mathbf{P}_{F}^{\\intercal} \\mathbf{\\Omega}_{F}^{-1} \\bm{Q}_{F}\\right)\\\\
\\bm{\\Pi}_{F} &= \\bm{\\mu}_{F} - r\\\\
\\mathbf{\\Sigma}_{\\mathrm{BF}} &= \\mathbf{\\Sigma}^{-1} \\mathbf{B} \\left( \\overline{\\mathbf{\\Sigma}}_{F} + \\mathbf{B}^{\\intercal} \\mathbf{\\Sigma}^{-1} \\mathbf{B} \\right)^{-1}\\\\
\\mathbf{\\Sigma}_{\\mathrm{BLB}} &= \\left(\\mathbf{\\Sigma}^{-1} - \\mathbf{\\Sigma}_{\\mathrm{BF}} \\mathbf{B}^{\\intercal} \\mathbf{\\Sigma}^{-1}\\right)^{-1}\\\\
\\bm{\\mu}_{\\mathrm{BLB}} &= \\mathbf{\\Sigma}_{\\mathrm{BLB}} \\mathbf{\\Sigma}_{\\mathrm{BF}} \\overline{\\mathbf{\\Sigma}}_{F} \\overline{\\bm{\\Pi}}_{F} + r
\\end{align*}
```

Where:

  - ``\\mathbf{\\Sigma}``: is the `Na×Na` estimated asset covariance matrix computed using the factor model.
  - ``\\mathbf{B}``: is the `Na×Nf` loadings matrix.
  - ``\\mathbf{\\Sigma}_{F}``: is the `Nf×Nf` factor covariance matrix.
  - ``\\bm{w}``: is the `Na×1` vector of benchmark asset weights.
  - ``\\mathbf{\\Sigma}_{\\epsilon}``: is an `Na×Na` diagonal matrix constructed from the variances of the errors between the asset and estimated asset returns using the factor model.
  - ``\\mathbf{X}``: is the `T×Na` matrix of asset returns.
  - ``\\mathbf{F}``: is the `T×Nf` matrix of factor returns.
  - ``\\overline{\\mathbf{\\Sigma}}_{F}``: is the `Nf×Nf` posterior covariance matrix of the factors after adjusting by the factor views.
  - ``\\mathbf{P}_{F}``: is the `Nvf×Nf` factor views matrix.
  - ``\\mathbf{\\Omega}_{F}``: is the `Nvf×Nvf` covariance matrix of the errors of the factor views.
  - ``T``: is the number of returns observations.
  - ``\\overline{\\bm{\\Pi}}_{F}``: is the `Nf×1` posterior equilibrium excess returns vector of the factors after adjusting by the factor views.
  - ``\\bm{\\Pi}_{F}``: is the `Nf×1` equilibrium excess returns vector of the factors.
  - ``\\bm{\\mu}_{F}``: is the `Nf×1` factor expected returns vector.
  - ``r``: is the risk-free rate.
  - ``\\bm{Q}_{F}``: is the `Nvf×1` factor views returns vector.
  - ``\\mathbf{\\Sigma}_{\\mathrm{BF}}``: is an `Na×Nf` intermediate covariance matrix,.
  - ``\\mathbf{\\Sigma}_{\\mathrm{BLB}}``: is the `Na×Na` posterior asset covariance matrix, aka the asset covariance matrix obtained via the Bayesian Black-Litterman model.
  - ``\\bm{\\mu}_{\\mathrm{BLB}}``: is the `Na×1` posterior asset expected returns vector, aka the asset returns vector obtained via the Bayesian Black-Litterman model.

# Parameters

  - `constant`:

      + if `true`: the loadings matrix contains the constant term as its first column.
      + else: the loadings matrix does not contain the constant term.

  - `error`:

      + if `true`: correct the estimated asset covariance matrix by adding the variances of the errors between the actual returns and factor estimated returns.

!!! note

    Only useful when the factor model is based on a regression model.

  - `delta`: risk aversion factor.
  - `rf`: risk free rate.
  - `ve`: [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator) for computing the errors covariance.
  - `var_w`: optional weights for computing the errors covariance.
  - `posdef`: method for fixing non positive Bayesian Black-Litterman matrices [`PosdefFix`](@ref).
  - `denoise` method for denoising the Bayesian Black-Litterman covariance matrix [`Denoise`](@ref).
  - `logo`: method for computing the LoGo Bayesian Black-Litterman covariance matrix [`AbstractLoGo`](@ref).
"""
mutable struct BBLType{T1 <: Real} <: BlackLittermanFactor
    constant::Bool
    error::Bool
    delta::Union{<:Real, Nothing}
    rf::T1
    ve::StatsBase.CovarianceEstimator
    var_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
    denoise::Denoise
    logo::AbstractLoGo
end
function BBLType(; constant::Bool = true, error::Bool = true,
                 delta::Union{<:Real, Nothing} = 1.0, rf::Real = 0.0,
                 ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                 var_w::Union{<:AbstractWeights, Nothing} = nothing,
                 posdef::PosdefFix = PosdefNearest(), denoise::Denoise = NoDenoise(),
                 logo::AbstractLoGo = NoLoGo())
    return BBLType{typeof(rf)}(constant, error, delta, rf, ve, var_w, posdef, denoise, logo)
end

export BLType, BBLType, ABLType
