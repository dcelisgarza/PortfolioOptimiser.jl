using CSV, TimeSeries, StatsBase, Statistics, LinearAlgebra, Test, PortfolioOptimiser

prices = TimeArray(CSV.File("./assets/stock_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

@testset "HC Portfolio Asset Clustering" begin
    portfolio = HCPortfolio(; prices = prices)
    asset_statistics!(portfolio; set_kurt = false, set_skurt = false, set_cov = false,
                      set_mu = false)

    ca = DBHT()
    ct = HCType()
    cluster_assets!(portfolio; hclust_alg = ca, hclust_opt = ct)

    idx, clustering, k = cluster_assets(portfolio; hclust_alg = ca, hclust_opt = ct)

    idxt = [1, 1, 1, 1, 1, 2, 1, 1, 2, 3, 2, 2, 3, 2, 2, 3, 1, 1, 2, 1]
    mergest = [-14 -15; -11 -6; -19 -12; -18 -8; -17 -2; -7 5; -16 -10; -13 7; -1 -5; -4 9;
               -3 -20; -9 3; 10 11; 2 1; 13 4; 12 14; 6 15; 8 16; 17 18]
    heightst = [0.05263157894736842, 0.05555555555555555, 0.058823529411764705, 0.0625,
                0.06666666666666667, 0.07142857142857142, 0.07692307692307693,
                0.08333333333333333, 0.09090909090909091, 0.1, 0.1111111111111111, 0.125,
                0.14285714285714285, 0.16666666666666666, 0.2, 0.25, 0.3333333333333333,
                0.5, 1.0]
    ordert = [7, 17, 2, 4, 1, 5, 3, 20, 18, 8, 13, 16, 10, 9, 19, 12, 11, 6, 14, 15]
    linkaget = :DBHT
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

    ct = HCType(; k = 6)
    idx, clustering, k = cluster_assets(portfolio; hclust_alg = ca, hclust_opt = ct)

    idxt = [1, 2, 1, 1, 1, 3, 2, 4, 5, 6, 3, 5, 6, 3, 3, 6, 2, 4, 5, 1]
    mergest = [-14 -15; -11 -6; -19 -12; -18 -8; -17 -2; -7 5; -16 -10; -13 7; -1 -5; -4 9;
               -3 -20; -9 3; 10 11; 2 1; 13 4; 12 14; 6 15; 8 16; 17 18]
    heightst = [0.05263157894736842, 0.05555555555555555, 0.058823529411764705, 0.0625,
                0.06666666666666667, 0.07142857142857142, 0.07692307692307693,
                0.08333333333333333, 0.09090909090909091, 0.1, 0.1111111111111111, 0.125,
                0.14285714285714285, 0.16666666666666666, 0.2, 0.25, 0.3333333333333333,
                0.5, 1.0]
    ordert = [7, 17, 2, 4, 1, 5, 3, 20, 18, 8, 13, 16, 10, 9, 19, 12, 11, 6, 14, 15]
    linkaget = :DBHT
    kt = 6

    @test isapprox(idx, idxt)
    @test isapprox(clustering.merges, mergest)
    @test isapprox(clustering.heights, heightst)
    @test isapprox(clustering.order, ordert)
    @test isequal(clustering.order, ordert)
    @test isequal(k, kt)

    ct = HCType(; k_method = StdSilhouette())
    idx, clustering, k = cluster_assets(portfolio; hclust_alg = ca, hclust_opt = ct)

    idxt = [1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1]
    mergest = [-14 -15; -11 -6; -19 -12; -18 -8; -17 -2; -7 5; -16 -10; -13 7; -1 -5; -4 9;
               -3 -20; -9 3; 10 11; 2 1; 13 4; 12 14; 6 15; 8 16; 17 18]
    heightst = [0.05263157894736842, 0.05555555555555555, 0.058823529411764705, 0.0625,
                0.06666666666666667, 0.07142857142857142, 0.07692307692307693,
                0.08333333333333333, 0.09090909090909091, 0.1, 0.1111111111111111, 0.125,
                0.14285714285714285, 0.16666666666666666, 0.2, 0.25, 0.3333333333333333,
                0.5, 1.0]
    ordert = [7, 17, 2, 4, 1, 5, 3, 20, 18, 8, 13, 16, 10, 9, 19, 12, 11, 6, 14, 15]
    linkaget = :DBHT
    kt = 2

    @test isapprox(idx, idxt)
    @test isapprox(clustering.merges, mergest)
    @test isapprox(clustering.heights, heightst)
    @test isapprox(clustering.order, ordert)
    @test isequal(clustering.order, ordert)
    @test isequal(k, kt)

    portfolio = HCPortfolio(; prices = prices)
    asset_statistics!(portfolio; set_kurt = false, set_skurt = false, set_cov = false,
                      set_mu = false)

    ca = HAC()
    ct = HCType()
    idx, clustering, k = cluster_assets(portfolio; hclust_alg = ca, hclust_opt = ct)

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
    linkaget = :ward
    kt = 4

    ct = HCType(; k_method = StdSilhouette())
    idx, clustering, k = cluster_assets(portfolio; hclust_alg = ca, hclust_opt = ct)

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
    linkaget = :ward
    kt = 4

    @test isapprox(idx, idxt)
    @test isapprox(clustering.merges, mergest)
    @test isapprox(clustering.heights, heightst)
    @test isapprox(clustering.order, ordert)
    @test isequal(clustering.order, ordert)
    @test isequal(k, kt)

    ct.k = 9
    idx, clustering, k = cluster_assets(portfolio; hclust_alg = ca, hclust_opt = ct)

    idxt = [1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 2, 6, 7, 2, 8, 9, 1, 2, 5, 1]
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
    linkaget = :ward
    kt = 9

    @test isapprox(idx, idxt)
    @test isapprox(clustering.merges, mergest)
    @test isapprox(clustering.heights, heightst)
    @test isapprox(clustering.order, ordert)
    @test isequal(clustering.order, ordert)
    @test isequal(k, kt)
end

@testset "HC Portfolio Asset Clustering" begin
    portfolio = Portfolio(; prices = prices)
    asset_statistics!(portfolio; set_kurt = false, set_skurt = false, set_cov = false,
                      set_mu = false)

    ca = HAC()
    ct = HCType()
    idx, clustering, k, S, D = cluster_assets(portfolio; hclust_alg = ca, hclust_opt = ct)

    idx, clustering, k, S, D = cluster_assets(portfolio; hclust_alg = ca, hclust_opt = ct)

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
    linkaget = :ward
    kt = 3

    @test isapprox(idx, idxt)
    @test isapprox(clustering.merges, mergest)
    @test isapprox(clustering.heights, heightst)
    @test isapprox(clustering.order, ordert)
    @test isequal(clustering.order, ordert)
    @test isequal(k, kt)

    ca = DBHT()
    idx, clustering, k, S, D = cluster_assets(portfolio; hclust_alg = ca, hclust_opt = ct)

    idxt = [1, 1, 1, 1, 1, 2, 1, 1, 2, 3, 2, 2, 3, 2, 2, 3, 1, 1, 2, 1]
    mergest = [-14 -15; -11 -6; -19 -12; -18 -8; -17 -2; -7 5; -16 -10; -13 7; -1 -5; -4 9;
               -3 -20; -9 3; 10 11; 2 1; 13 4; 12 14; 6 15; 8 16; 17 18]
    heightst = [0.05263157894736842, 0.05555555555555555, 0.058823529411764705, 0.0625,
                0.06666666666666667, 0.07142857142857142, 0.07692307692307693,
                0.08333333333333333, 0.09090909090909091, 0.1, 0.1111111111111111, 0.125,
                0.14285714285714285, 0.16666666666666666, 0.2, 0.25, 0.3333333333333333,
                0.5, 1.0]
    ordert = [7, 17, 2, 4, 1, 5, 3, 20, 18, 8, 13, 16, 10, 9, 19, 12, 11, 6, 14, 15]
    linkaget = :DBHT
    kt = 3

    @test isapprox(idx, idxt)
    @test isapprox(clustering.merges, mergest)
    @test isapprox(clustering.heights, heightst)
    @test isapprox(clustering.order, ordert)
    @test isequal(clustering.order, ordert)
    @test isequal(k, kt)
end
