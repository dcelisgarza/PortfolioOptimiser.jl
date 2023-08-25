using Test, PortfolioOptimiser, CSV, TimeSeries, DataFrames
A = TimeArray(CSV.File("./assets/stock_prices.csv"), timestamp = :date)
Y = percentchange(A)
returns = dropmissing!(DataFrame(Y))

@testset "DBHT tests" begin
    portfolio = HCPortfolio(returns = returns)
    asset_statistics!(portfolio)

    D = portfolio.dist
    S = 1 .- D .^ 2

    T8, Rpm, Adjv, Dpm, Mv, Z, dbht = DBHTs(D, S, branchorder = :optimal)
    m1 = Z[:, 1] .< 0
    m2 = Z[:, 2] .< 0
    Z[.!m1, 1] .+= size(Z, 1) + 1
    Z[.!m2, 2] .+= size(Z, 1) + 1
    Z[m1, 1] .= -Z[m1, 1]
    Z[m2, 2] .= -Z[m2, 2]
    Z[:, 1:2] .-= 1
    sort!(Z[:, 1:2], dims = 2)
    Zt = reshape(
        [
            14.0,
            5.0,
            18.0,
            7.0,
            16.0,
            6.0,
            9.0,
            26.0,
            4.0,
            28.0,
            19.0,
            8.0,
            30.0,
            20.0,
            23.0,
            33.0,
            25.0,
            35.0,
            36.0,
            13.0,
            10.0,
            11.0,
            17.0,
            1.0,
            24.0,
            15.0,
            12.0,
            0.0,
            3.0,
            2.0,
            22.0,
            29.0,
            21.0,
            32.0,
            31.0,
            34.0,
            27.0,
            37.0,
            0.05263157894736842,
            0.05555555555555555,
            0.058823529411764705,
            0.0625,
            0.06666666666666667,
            0.07142857142857142,
            0.07692307692307693,
            0.08333333333333333,
            0.09090909090909091,
            0.1,
            0.1111111111111111,
            0.125,
            0.14285714285714285,
            0.16666666666666666,
            0.2,
            0.25,
            0.3333333333333333,
            0.5,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            3.0,
            2.0,
            3.0,
            2.0,
            3.0,
            2.0,
            3.0,
            5.0,
            4.0,
            7.0,
            7.0,
            10.0,
            10.0,
            20.0,
        ],
        :,
        4,
    )
    Zt[:, 1:2] .= sort(Zt[:, 1:2], dims = 2)
    @test isapprox(Z, Zt)

    T8, Rpm, Adjv, Dpm, Mv, Z, dbht = DBHTs(D, S, branchorder = :r)
    m1 = Z[:, 1] .< 0
    m2 = Z[:, 2] .< 0
    Z[.!m1, 1] .+= size(Z, 1) + 1
    Z[.!m2, 2] .+= size(Z, 1) + 1
    Z[m1, 1] .= -Z[m1, 1]
    Z[m2, 2] .= -Z[m2, 2]
    Z[:, 1:2] .-= 1
    sort!(Z[:, 1:2], dims = 2)
    Zt = reshape(
        [
            14.0,
            5.0,
            18.0,
            7.0,
            16.0,
            6.0,
            9.0,
            26.0,
            4.0,
            28.0,
            19.0,
            8.0,
            30.0,
            20.0,
            23.0,
            33.0,
            25.0,
            35.0,
            36.0,
            13.0,
            10.0,
            11.0,
            17.0,
            1.0,
            24.0,
            15.0,
            12.0,
            0.0,
            3.0,
            2.0,
            22.0,
            29.0,
            21.0,
            32.0,
            31.0,
            34.0,
            27.0,
            37.0,
            0.05263157894736842,
            0.05555555555555555,
            0.058823529411764705,
            0.0625,
            0.06666666666666667,
            0.07142857142857142,
            0.07692307692307693,
            0.08333333333333333,
            0.09090909090909091,
            0.1,
            0.1111111111111111,
            0.125,
            0.14285714285714285,
            0.16666666666666666,
            0.2,
            0.25,
            0.3333333333333333,
            0.5,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            3.0,
            2.0,
            3.0,
            2.0,
            3.0,
            2.0,
            3.0,
            5.0,
            4.0,
            7.0,
            7.0,
            10.0,
            10.0,
            20.0,
        ],
        :,
        4,
    )
    Zt[:, 1:2] .= sort(Zt[:, 1:2], dims = 2)
    @test isapprox(Z, Zt)

    T8, Rpm, Adjv, Dpm, Mv, Z, dbht = DBHTs(D, S, branchorder = :default)
    m1 = Z[:, 1] .< 0
    m2 = Z[:, 2] .< 0
    Z[.!m1, 1] .+= size(Z, 1) + 1
    Z[.!m2, 2] .+= size(Z, 1) + 1
    Z[m1, 1] .= -Z[m1, 1]
    Z[m2, 2] .= -Z[m2, 2]
    Z[:, 1:2] .-= 1
    Zt = reshape(
        [
            13.0,
            5.0,
            11.0,
            7.0,
            1.0,
            6.0,
            9.0,
            12.0,
            0.0,
            3.0,
            2.0,
            8.0,
            29.0,
            20.0,
            23.0,
            31.0,
            25.0,
            27.0,
            36.0,
            14.0,
            10.0,
            18.0,
            17.0,
            16.0,
            24.0,
            15.0,
            26.0,
            4.0,
            28.0,
            19.0,
            22.0,
            30.0,
            21.0,
            32.0,
            33.0,
            34.0,
            35.0,
            37.0,
            0.05263157894736842,
            0.05555555555555555,
            0.058823529411764705,
            0.0625,
            0.06666666666666667,
            0.07142857142857142,
            0.07692307692307693,
            0.08333333333333333,
            0.09090909090909091,
            0.1,
            0.1111111111111111,
            0.125,
            0.14285714285714285,
            0.16666666666666666,
            0.2,
            0.25,
            0.3333333333333333,
            0.5,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            3.0,
            2.0,
            3.0,
            2.0,
            3.0,
            2.0,
            3.0,
            5.0,
            4.0,
            7.0,
            7.0,
            10.0,
            10.0,
            20.0,
        ],
        :,
        4,
    )
    @test isapprox(Z, Zt)
end