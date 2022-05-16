using Test
using PortfolioOptimiser.AssetAllocation

@testset "Asset Allocation tests" begin
    tickers = ["AAPL", "GOOG", "AMD", "MSFT"]
    n = length(tickers)

    weights = ones(n)
    weights /= sum(weights)
    latest_prices = ones(n)
    investment = 1000
    greedyAlloc, remaining =
        Allocation(Greedy(), tickers, weights, latest_prices, investment)
    nweights = greedyAlloc.weights
    shares = greedyAlloc.shares
    @test sum(shares .* latest_prices) == investment - remaining
    @test all(nweights .≈ 0.25)
    @test abs(sum(nweights) - 1) < eps() * n

    weights = ones(n)
    weights /= sum(weights)
    latest_prices = [1, 2, 3, 4]
    investment = 1000
    greedyAlloc, remaining =
        Allocation(Greedy(), tickers, weights, latest_prices, investment)
    nweights = greedyAlloc.weights
    shares = greedyAlloc.shares
    @test sum(shares .* latest_prices) == investment - remaining
    @test nweights ≈ [0.25, 0.25, 0.252, 0.248]
    @test abs(sum(nweights) - 1) < eps() * n

    weights = [1, 2, 3, 4]
    weights /= sum(weights)
    latest_prices = ones(n)
    investment = 1000
    greedyAlloc, remaining =
        Allocation(Greedy(), tickers, weights, latest_prices, investment)
    nweights = greedyAlloc.weights
    shares = greedyAlloc.shares
    @test sum(shares .* latest_prices) == investment - remaining
    @test nweights ≈ sort!(weights, rev = true)
    @test abs(sum(nweights) - 1) < eps() * n

    weights = [1, 2, 3, 4]
    weights /= sum(weights)
    latest_prices = [4, 3, 2, 1]
    investment = 1000
    greedyAlloc, remaining =
        Allocation(Greedy(), tickers, weights, latest_prices, investment)
    nweights = greedyAlloc.weights
    shares = greedyAlloc.shares
    idx = sortperm(weights, rev = true)
    @test sum(shares .* latest_prices[idx]) == investment - remaining
    @test abs(sum(nweights) - 1) < eps() * n

    weights = [1, -1, -1, 1]
    weights /= 2
    latest_prices = ones(n)
    investment = 1000
    greedyAlloc, remaining =
        Allocation(Greedy(), tickers, weights, latest_prices, investment)
    nweights = greedyAlloc.weights
    shares = greedyAlloc.shares
    idx = sortperm(weights, rev = true)
    @test sum(shares .* latest_prices[idx]) == 2 * investment - remaining
    @test abs(sum(nweights)) < eps() * n

    weights = [1, -2, -1, 2]
    weights /= 3
    latest_prices = ones(n)
    investment = 1000
    greedyAlloc, remaining =
        Allocation(Greedy(), tickers, weights, latest_prices, investment)
    nweights = greedyAlloc.weights
    shares = greedyAlloc.shares
    idx = [sortperm(weights, rev = true)[1:2]; sortperm(weights)[1:2]]
    weights[idx] ≈ nweights
    @test sum(shares .* latest_prices[idx]) == 2 * investment - remaining
    @test abs(sum(nweights)) < eps() * n

    weights = [1, -1, -1, 1]
    weights /= 2
    latest_prices = [2, 1, 3, 1]
    investment = 1000
    greedyAlloc, remaining =
        Allocation(Greedy(), tickers, weights, latest_prices, investment)
    nweights = greedyAlloc.weights
    shares = greedyAlloc.shares
    idx = sortperm(weights, rev = true)
    @test sum(shares .* latest_prices[idx]) == 2 * investment - remaining
    @test abs(sum(nweights)) < eps() * n

    weights = Float64[1, -3, -1, 2]
    weights[[1, 4]] ./= 3
    weights[[2, 3]] ./= 4
    latest_prices = [1, 2, 3, 4]
    investment = 1000
    greedyAlloc, remaining =
        Allocation(Greedy(), tickers, weights, latest_prices, investment)
    nweights = greedyAlloc.weights
    shares = greedyAlloc.shares
    idx = [4, 1, 2, 3]
    @test sum(shares .* latest_prices[idx]) == 2 * investment - remaining
    @test abs(sum(nweights)) < eps() * n

    # Linear integer programming
    weights = ones(length(tickers))
    weights /= sum(weights)
    latest_prices = ones(length(tickers))
    investment = 1000
    lpAlloc, remaining = Allocation(LP(), tickers, weights, latest_prices, investment)
    nweights = lpAlloc.weights
    shares = lpAlloc.shares
    @test sum(shares .* latest_prices) == investment - remaining
    @test all(nweights .≈ 0.25)

    weights = ones(length(tickers))
    weights /= sum(weights)
    latest_prices = [1, 2, 3, 4]
    investment = 1000
    lpAlloc, remaining = Allocation(LP(), tickers, weights, latest_prices, investment)
    nweights = lpAlloc.weights
    shares = lpAlloc.shares
    @test sum(shares .* latest_prices) == investment - remaining
    @test nweights ≈ [0.249, 0.25, 0.249, 0.252]

    weights = [1, 2, 3, 4]
    weights /= sum(weights)
    latest_prices = ones(length(tickers))
    investment = 1000
    lpAlloc, remaining = Allocation(LP(), tickers, weights, latest_prices, investment)
    nweights = lpAlloc.weights
    shares = lpAlloc.shares
    @test sum(shares .* latest_prices) == investment - remaining
    @test nweights ≈ sort!(weights, rev = true)

    weights = [1, 2, 3, 4]
    weights /= sum(weights)
    latest_prices = [4, 3, 2, 1]
    investment = 1000
    lpAlloc, remaining = Allocation(LP(), tickers, weights, latest_prices, investment)
    nweights = lpAlloc.weights
    shares = lpAlloc.shares
    idx = sortperm(weights, rev = true)
    @test sum(shares .* latest_prices[idx]) == investment - remaining
    @test abs(sum(nweights) - 1) < eps() * n

    weights = [1, -1, -1, 1]
    weights /= 2
    latest_prices = ones(length(tickers))
    investment = 1000
    lpAlloc, remaining = Allocation(LP(), tickers, weights, latest_prices, investment)
    nweights = lpAlloc.weights
    shares = lpAlloc.shares
    idx = sortperm(weights, rev = true)
    @test sum(shares .* latest_prices[idx]) == 2 * investment - remaining

    weights = [1, -2, -1, 2]
    weights /= 3
    latest_prices = ones(length(tickers))
    investment = 1000
    lpAlloc, remaining = Allocation(LP(), tickers, weights, latest_prices, investment)
    nweights = lpAlloc.weights
    shares = lpAlloc.shares
    idx = [sortperm(weights, rev = true)[1:2]; sortperm(weights)[1:2]]
    weights[idx] ≈ nweights
    @test sum(shares .* latest_prices[idx]) == 2 * investment - remaining

    weights = [1, -1, -1, 1]
    weights /= 2
    latest_prices = [2, 1, 3, 1]
    investment = 1000
    lpAlloc, remaining = Allocation(LP(), tickers, weights, latest_prices, investment)
    nweights = lpAlloc.weights
    shares = lpAlloc.shares
    idx = sortperm(weights, rev = true)
    @test sum(shares .* latest_prices[idx]) == 2 * investment - remaining

    weights = Float64[1, -3, -1, 2]
    weights[[1, 4]] ./= 3
    weights[[2, 3]] ./= 4
    latest_prices = [1, 2, 3, 4]
    investment = 1000
    lpAlloc, remaining = Allocation(LP(), tickers, weights, latest_prices, investment)
    nweights = lpAlloc.weights
    shares = lpAlloc.shares
    idx = [4, 1, 2, 3]
    @test sum(shares .* latest_prices[idx]) == 2 * investment - remaining
end