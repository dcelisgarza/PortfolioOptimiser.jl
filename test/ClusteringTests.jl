using CSV, TimeSeries, StatsBase, Statistics, LinearAlgebra, Test, PortfolioOptimiser

path = joinpath(@__DIR__, "assets/stock_prices.csv")
path2 = joinpath(@__DIR__, "assets/stock_prices2.csv")
prices = TimeArray(CSV.File(path); timestamp = :date)
prices2 = TimeArray(CSV.File(path2); timestamp = :date)
rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "HC Portfolio Asset Clustering" begin
    portfolio = Portfolio(; prices = prices)
    asset_statistics!(portfolio; set_kurt = false, set_skurt = false, set_cov = false,
                      set_mu = false)

    ca = DBHT()
    ct = ClustOpt()
    cluster_assets!(portfolio; clust_alg = ca, clust_opt = ct)

    idx, clustering, k = cluster_assets(portfolio; clust_alg = ca, clust_opt = ct)

    idxt = [1, 1, 1, 1, 1, 2, 1, 1, 2, 3, 2, 2, 3, 2, 2, 3, 1, 1, 2, 1]
    mergest = [-14 -15; -11 -6; -19 -12; -18 -8; -17 -2; -7 5; -16 -10; -13 7; -1 -5; -4 9;
               -3 -20; -9 3; 10 11; 2 1; 13 4; 12 14; 6 15; 8 16; 17 18]
    heightst = [0.05263157894736842, 0.05555555555555555, 0.058823529411764705, 0.0625,
                0.06666666666666667, 0.07142857142857142, 0.07692307692307693,
                0.08333333333333333, 0.09090909090909091, 0.1, 0.1111111111111111, 0.125,
                0.14285714285714285, 0.16666666666666666, 0.2, 0.25, 0.3333333333333333,
                0.5, 1.0]
    ordert = [7, 17, 2, 4, 1, 5, 3, 20, 18, 8, 13, 16, 10, 9, 19, 12, 11, 6, 14, 15]

    kt = 3

    @test isapprox(idx, idxt)
    @test isapprox(clustering.merges, mergest)
    @test isapprox(clustering.heights, heightst)
    @test isapprox(clustering.order, ordert)
    @test isequal(clustering.order, ordert)
    @test isequal(k, kt)

    @test isapprox(portfolio.clusters.merges, mergest)
    @test isapprox(portfolio.clusters.heights, heightst)
    @test isapprox(portfolio.clusters.order, ordert)
    @test isequal(portfolio.clusters.order, ordert)
    @test isequal(portfolio.k, kt)

    ct = ClustOpt(; k = 6)
    idx, clustering, k = cluster_assets(portfolio; clust_alg = ca, clust_opt = ct)

    idxt = [1, 2, 1, 1, 1, 3, 2, 1, 4, 5, 3, 4, 5, 3, 3, 5, 2, 1, 4, 1]
    mergest = [-14 -15; -11 -6; -19 -12; -18 -8; -17 -2; -7 5; -16 -10; -13 7; -1 -5; -4 9;
               -3 -20; -9 3; 10 11; 2 1; 13 4; 12 14; 6 15; 8 16; 17 18]
    heightst = [0.05263157894736842, 0.05555555555555555, 0.058823529411764705, 0.0625,
                0.06666666666666667, 0.07142857142857142, 0.07692307692307693,
                0.08333333333333333, 0.09090909090909091, 0.1, 0.1111111111111111, 0.125,
                0.14285714285714285, 0.16666666666666666, 0.2, 0.25, 0.3333333333333333,
                0.5, 1.0]
    ordert = [7, 17, 2, 4, 1, 5, 3, 20, 18, 8, 13, 16, 10, 9, 19, 12, 11, 6, 14, 15]

    kt = ceil(Int, sqrt(size(portfolio.returns, 2)))

    @test isapprox(idx, idxt)
    @test isapprox(clustering.merges, mergest)
    @test isapprox(clustering.heights, heightst)
    @test isapprox(clustering.order, ordert)
    @test isequal(clustering.order, ordert)
    @test isequal(k, kt)

    ct = ClustOpt(; k_method = StdSilhouette())
    idx, clustering, k = cluster_assets(portfolio; clust_alg = ca, clust_opt = ct)

    idxt = [1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1]
    mergest = [-14 -15; -11 -6; -19 -12; -18 -8; -17 -2; -7 5; -16 -10; -13 7; -1 -5; -4 9;
               -3 -20; -9 3; 10 11; 2 1; 13 4; 12 14; 6 15; 8 16; 17 18]
    heightst = [0.05263157894736842, 0.05555555555555555, 0.058823529411764705, 0.0625,
                0.06666666666666667, 0.07142857142857142, 0.07692307692307693,
                0.08333333333333333, 0.09090909090909091, 0.1, 0.1111111111111111, 0.125,
                0.14285714285714285, 0.16666666666666666, 0.2, 0.25, 0.3333333333333333,
                0.5, 1.0]
    ordert = [7, 17, 2, 4, 1, 5, 3, 20, 18, 8, 13, 16, 10, 9, 19, 12, 11, 6, 14, 15]

    kt = 2

    @test isapprox(idx, idxt)
    @test isapprox(clustering.merges, mergest)
    @test isapprox(clustering.heights, heightst)
    @test isapprox(clustering.order, ordert)
    @test isequal(clustering.order, ordert)
    @test isequal(k, kt)

    portfolio = Portfolio(; prices = prices)
    asset_statistics!(portfolio; set_kurt = false, set_skurt = false, set_cov = false,
                      set_mu = false)

    ca = HAC()
    ct = ClustOpt()
    idx, clustering, k = cluster_assets(portfolio; clust_alg = ca, clust_opt = ct)

    idxt = [1, 1, 1, 1, 1, 2, 3, 2, 2, 2, 2, 4, 4, 2, 3, 4, 1, 2, 2, 1]
    mergest = [-3 -1; 1 -5; -19 -9; -17 -2; -20 4; 5 2; 6 -4; -14 -6; -11 8; -10 3; 9 -18;
               -13 -16; 12 -12; 11 -8; -7 -15; 10 14; 15 13; 17 16; 7 18]
    heightst = [0.444008719373704, 0.4749395689083809, 0.2342763563575649,
                0.5002626909483501, 0.5399850650791093, 0.5710442442346079,
                0.6059606113417358, 0.5323895977848843, 0.5591137255770308,
                0.5626399107657335, 0.5718719562298092, 0.6180905459915621,
                0.6301944178114925, 0.6308794462554908, 0.6371130550334823,
                0.6666679941820913, 0.6984752242370402, 0.7565920027449732,
                0.8789050122642453]
    ordert = [20, 17, 2, 3, 1, 5, 4, 7, 15, 13, 16, 12, 10, 19, 9, 11, 14, 6, 18, 8]

    kt = 4

    ct = ClustOpt(; k_method = StdSilhouette())
    idx, clustering, k = cluster_assets(portfolio; clust_alg = ca, clust_opt = ct)

    idxt = [1, 1, 1, 1, 1, 2, 3, 2, 2, 2, 2, 4, 4, 2, 3, 4, 1, 2, 2, 1]
    mergest = [-3 -1; 1 -5; -19 -9; -17 -2; -20 4; 5 2; 6 -4; -14 -6; -11 8; -10 3; 9 -18;
               -13 -16; 12 -12; 11 -8; -7 -15; 10 14; 15 13; 17 16; 7 18]
    heightst = [0.444008719373704, 0.4749395689083809, 0.2342763563575649,
                0.5002626909483501, 0.5399850650791093, 0.5710442442346079,
                0.6059606113417358, 0.5323895977848843, 0.5591137255770308,
                0.5626399107657335, 0.5718719562298092, 0.6180905459915621,
                0.6301944178114925, 0.6308794462554908, 0.6371130550334823,
                0.6666679941820913, 0.6984752242370402, 0.7565920027449732,
                0.8789050122642453]
    ordert = [20, 17, 2, 3, 1, 5, 4, 7, 15, 13, 16, 12, 10, 19, 9, 11, 14, 6, 18, 8]

    kt = 4

    @test isapprox(idx, idxt)
    @test isapprox(clustering.merges, mergest)
    @test isapprox(clustering.heights, heightst)
    @test isapprox(clustering.order, ordert)
    @test isequal(clustering.order, ordert)
    @test isequal(k, kt)

    ct.k = 9
    idx, clustering, k = cluster_assets(portfolio; clust_alg = ca, clust_opt = ct)

    idxt = [1, 1, 1, 1, 1, 2, 3, 2, 4, 4, 2, 5, 5, 2, 3, 5, 1, 2, 4, 1]
    mergest = [-3 -1; 1 -5; -19 -9; -17 -2; -20 4; 5 2; 6 -4; -14 -6; -11 8; -10 3; 9 -18;
               -13 -16; 12 -12; 11 -8; -7 -15; 10 14; 15 13; 17 16; 7 18]
    heightst = [0.444008719373704, 0.4749395689083809, 0.2342763563575649,
                0.5002626909483501, 0.5399850650791093, 0.5710442442346079,
                0.6059606113417358, 0.5323895977848843, 0.5591137255770308,
                0.5626399107657335, 0.5718719562298092, 0.6180905459915621,
                0.6301944178114925, 0.6308794462554908, 0.6371130550334823,
                0.6666679941820913, 0.6984752242370402, 0.7565920027449732,
                0.8789050122642453]
    ordert = [20, 17, 2, 3, 1, 5, 4, 7, 15, 13, 16, 12, 10, 19, 9, 11, 14, 6, 18, 8]

    kt = ceil(Int, sqrt(size(portfolio.returns, 2)))

    @test isapprox(idx, idxt)
    @test isapprox(clustering.merges, mergest)
    @test isapprox(clustering.heights, heightst)
    @test isapprox(clustering.order, ordert)
    @test isequal(clustering.order, ordert)
    @test isequal(k, kt)
end

@testset "Portfolio Asset Clustering" begin
    portfolio = Portfolio(; prices = prices)
    asset_statistics!(portfolio; set_kurt = false, set_skurt = false, set_cov = false,
                      set_mu = false)

    ca = HAC()
    ct = ClustOpt()
    idx, clustering, k = cluster_assets(portfolio; clust_alg = ca, clust_opt = ct)

    idx, clustering, k = cluster_assets(portfolio; clust_alg = ca, clust_opt = ct)

    idxt = [1, 1, 1, 1, 1, 2, 3, 2, 2, 2, 2, 3, 3, 2, 3, 3, 1, 2, 2, 1]
    mergest = [-3 -1; 1 -5; -19 -9; -17 -2; -20 4; 5 2; 6 -4; -14 -6; -11 8; -10 3; 9 -18;
               -13 -16; 12 -12; 11 -8; -7 -15; 10 14; 15 13; 17 16; 7 18]
    heightst = [0.444008719373704, 0.4749395689083809, 0.2342763563575649,
                0.5002626909483501, 0.5399850650791093, 0.5710442442346079,
                0.6059606113417358, 0.5323895977848843, 0.5591137255770308,
                0.5626399107657335, 0.5718719562298092, 0.6180905459915621,
                0.6301944178114925, 0.6308794462554908, 0.6371130550334823,
                0.6666679941820913, 0.6984752242370402, 0.7565920027449732,
                0.8789050122642453]
    ordert = [20, 17, 2, 3, 1, 5, 4, 7, 15, 13, 16, 12, 10, 19, 9, 11, 14, 6, 18, 8]

    kt = 3

    @test isapprox(idx, idxt)
    @test isapprox(clustering.merges, mergest)
    @test isapprox(clustering.heights, heightst)
    @test isapprox(clustering.order, ordert)
    @test isequal(clustering.order, ordert)
    @test isequal(k, kt)

    ca = DBHT()
    idx, clustering, k = cluster_assets(portfolio; clust_alg = ca, clust_opt = ct)

    idxt = [1, 1, 1, 1, 1, 2, 1, 1, 2, 3, 2, 2, 3, 2, 2, 3, 1, 1, 2, 1]
    mergest = [-14 -15; -11 -6; -19 -12; -18 -8; -17 -2; -7 5; -16 -10; -13 7; -1 -5; -4 9;
               -3 -20; -9 3; 10 11; 2 1; 13 4; 12 14; 6 15; 8 16; 17 18]
    heightst = [0.05263157894736842, 0.05555555555555555, 0.058823529411764705, 0.0625,
                0.06666666666666667, 0.07142857142857142, 0.07692307692307693,
                0.08333333333333333, 0.09090909090909091, 0.1, 0.1111111111111111, 0.125,
                0.14285714285714285, 0.16666666666666666, 0.2, 0.25, 0.3333333333333333,
                0.5, 1.0]
    ordert = [7, 17, 2, 4, 1, 5, 3, 20, 18, 8, 13, 16, 10, 9, 19, 12, 11, 6, 14, 15]

    kt = 3

    @test isapprox(idx, idxt)
    @test isapprox(clustering.merges, mergest)
    @test isapprox(clustering.heights, heightst)
    @test isapprox(clustering.order, ordert)
    @test isequal(clustering.order, ordert)
    @test isequal(k, kt)
end

@testset "Non-monotonic clustering" begin
    portfolio = Portfolio(; prices = prices2)
    asset_statistics!(portfolio; cov_type = PortCovCor(; ce = CorGerberSB1()),
                      cor_type = PortCovCor(; ce = CorGerberSB1()),
                      dist_type = DistDistMLP(), set_kurt = false, set_skurt = false,
                      set_skew = false, set_sskew = false)

    cluster_assets!(portfolio; clust_alg = DBHT(; similarity = DBHTMaxDist()),
                    clust_opt = ClustOpt(; k_method = StdSilhouette()))
    @test portfolio.k == 3

    cluster_assets!(portfolio; clust_alg = DBHT(; similarity = DBHTMaxDist()),
                    clust_opt = ClustOpt(; k_method = TwoDiff()))
    @test portfolio.k == 3

    cluster_assets!(portfolio; clust_alg = DBHT(; similarity = DBHTMaxDist()),
                    clust_opt = ClustOpt(; k = 18, max_k = 1))
    @test portfolio.k == 1

    cluster_assets!(portfolio; clust_alg = DBHT(; similarity = DBHTMaxDist()),
                    clust_opt = ClustOpt(; k = 1))
    @test portfolio.k == 1

    cluster_assets!(portfolio; clust_alg = DBHT(; similarity = DBHTMaxDist()),
                    clust_opt = ClustOpt(; k = 7))
    @test portfolio.k == 8

    cluster_assets!(portfolio; clust_alg = DBHT(; similarity = DBHTMaxDist()),
                    clust_opt = ClustOpt(; k = 11))
    @test portfolio.k == 10

    cluster_assets!(portfolio; clust_alg = DBHT(; similarity = DBHTMaxDist()),
                    clust_opt = ClustOpt(; k = 16))
    @test portfolio.k == 15
end
