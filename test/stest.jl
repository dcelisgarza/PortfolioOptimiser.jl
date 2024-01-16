using COSMO, CSV, Clarabel, HiGHS, Pajarito, LinearAlgebra, OrderedCollections,
      PortfolioOptimiser, Statistics, Test, TimeSeries, Logging

prices = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)

solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                        :params => Dict("verbose" => false,
                                                        "max_step_fraction" => 0.75)),
                      :COSMO => Dict(:solver => COSMO.Optimizer,
                                     :params => Dict("verbose" => false)))
alloc_solvers = Dict(:HiGHS => Dict(:solver => HiGHS.Optimizer,
                                    :params => Dict("log_to_console" => false)))
portfolio = Portfolio(; prices = prices, solvers = solvers, alloc_solvers = alloc_solvers)
asset_statistics!(portfolio)
portfolio.alloc_solvers = alloc_solvers
w0 = optimise!(portfolio; obj = :Min_Risk)
w1 = allocate!(portfolio; alloc_type = :LP)
w2 = allocate!(portfolio; alloc_type = :Greedy)
w3 = allocate!(portfolio; alloc_type = :LP, investment = 1e4)
w4 = allocate!(portfolio; alloc_type = :Greedy, investment = 1e2)

@test isapprox(w0.weights, w1.weights, rtol = 0.01)
@test isapprox(w0.weights, w2.weights, rtol = 0.01)
@test isapprox(w0.weights, w3.weights, rtol = 0.1)
@test isapprox(w0.weights, w4.weights, rtol = 0.1)

###################################

prices_assets = TimeArray(CSV.File("./test/assets/stock_prices.csv"); timestamp = :date)
prices_factors = TimeArray(CSV.File("./test/assets/factor_prices.csv"); timestamp = :date)

rf = 1.0329^(1 / 252) - 1
l = 2.0

portfolio = Portfolio(; prices = prices_assets,
                      solvers = OrderedDict(:PajaritoClara => Dict(:solver => Pajarito.Optimizer,
                                                                   :params => Dict("conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                               "verbose" => false,
                                                                                                                               "max_step_fraction" => 0.75),
                                                                                   "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                            "log_to_console" => false)))),
                      max_number_assets = 3)
portfolio = HCPortfolio(; prices = prices_assets,
                        solvers = OrderedDict(:PajaritoClara => Dict(:solver => Pajarito.Optimizer,
                                                                     :params => Dict("conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                                 "verbose" => false,
                                                                                                                                 "max_step_fraction" => 0.75),
                                                                                     "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                              "log_to_console" => false)))))

asset_statistics!(portfolio)

########################################

println("cov_lt1 = reshape(", vec(portfolio.cov_l), ", 20, 20)")
println("cov_ut1 = reshape(", vec(portfolio.cov_u), ", 20, 20)")
println("cov_mut1 = ", sparse(portfolio.cov_mu))
println("cov_sigmat1 = ", sparse(portfolio.cov_sigma))
println("d_mut1 = ", portfolio.d_mu)
println("k_mut1 = ", portfolio.k_mu)
println("k_sigmat1 = ", portfolio.k_sigma)

println("kurtt = reshape(", vec(kurt), ", 16, 16)")
println("skurtt = reshape(", vec(skurt), ", 16, 16)")

println("w1t = ", w1.weights, "\n")
println("w2t = ", w2.weights, "\n")
println("w3t = ", w3.weights, "\n")
println("w4t = ", w4.weights, "\n")
println("w5t = ", w5.weights, "\n")
println("w6t = ", w6.weights, "\n")
println("w7t = ", w7.weights, "\n")
println("w8t = ", w8.weights, "\n")
println("w9t = ", w9.weights, "\n")
println("w10t = ", w10.weights, "\n")
println("w11t = ", w11.weights, "\n")
println("w12t = ", w12.weights, "\n")
println("w13t = ", w13.weights, "\n")
println("w14t = ", w14.weights, "\n")
println("w15t = ", w15.weights, "\n")
println("w16t = ", w16.weights, "\n")
println("w17t = ", w17.weights, "\n")
println("w18t = ", w18.weights, "\n")
println("w19t = ", w19.weights, "\n")
#######################################

for rtol ∈ [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 2.5e-1, 5e-1, 1e0]
    a1, a2 = [0.007911813464310113, 0.03068510213747275, 0.010505366137939913,
              0.02748375285330511, 0.012276170276067139, 0.03340727036207522,
              3.968191684113028e-7, 0.1398469015680947, 6.62690268489184e-7,
              1.412505225860106e-5, 0.2878192785440307, 4.2200065419448415e-7,
              3.296864762413375e-7, 0.12527560333674628, 1.836077875288932e-6,
              0.015083599354027831, 2.2788913817547395e-5, 0.19311714086150886,
              8.333796539106878e-7, 0.11654660648424862],
             [0.007139796365348484, 0.03069434859366502, 0.010644490240537176,
              0.027531545485003184, 0.012843462256510068, 0.033397780724722555, 0.0,
              0.1398616151795111, 0.0, 3.900003587859001e-5, 0.28781651478054443, 0.0, 0.0,
              0.12528185525467142, 0.0, 0.015103844746978328, 0.0, 0.1931230230620357, 0.0,
              0.11652272327459409]
    if isapprox(a1, a2; rtol = rtol)
        println(", rtol = $(rtol)")
        break
    end
end

portfolio = Portfolio(; prices = prices,
                      solvers = OrderedDict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                                              :params => Dict("verbose" => false,
                                                                              "max_step_fraction" => 0.75)),
                                            :COSMO => Dict(:solver => COSMO.Optimizer,
                                                           :params => Dict("verbose" => false))))
asset_statistics!(portfolio)

w1 = optimise!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad, rm = :Kurt,
               obj = :Min_Risk, kelly = :None,)
risk1 = calc_risk(portfolio; type = :Trad, rm = :Kurt, rf = rf)

rmf = :kurt_u
setproperty!(portfolio, rmf, risk1 + 1e-4 * risk1)
w18 = optimise!(portfolio; rf = rf, l = l, class = :Classic, type = :Trad, rm = :Kurt,
                obj = :Sharpe, kelly = :None,)

@test isapprox(w18.weights, w1.weights, rtol = 1e-3)

w1 = optimise!(portfolio; class = :Classic, type = :RP, rm = :Kurt)
rc1 = risk_contribution(portfolio; type = :RP, rm = :Kurt)
lrc1, hrc1 = extrema(rc1)

portfolio.risk_budget = 1:size(portfolio.returns, 2)
w2 = optimise!(portfolio; class = :Classic, type = :RP, rm = :Kurt)
rc2 = risk_contribution(portfolio; type = :RP, rm = :Kurt)
lrc2, hrc2 = extrema(rc2)

w1t = [0.03879158773899491, 0.04946318916187915, 0.03767536457743636, 0.04975768359685481,
       0.03583384747996175, 0.05474667190193154, 0.02469826359420486, 0.10506491736193022,
       0.031245766025529604, 0.04312788495096333, 0.12822307815405873, 0.03170133005454372,
       0.026067725442004967, 0.057123092045424234, 0.03137705105386256, 0.04155724092469867,
       0.044681796838160794, 0.0754338209703899, 0.03624092724713855, 0.057188760880031476]

w2t = [0.004127710286387879, 0.010592152386952021, 0.012536905345418492,
       0.023303462236461917, 0.01936823663730284, 0.03214466953862615, 0.018650835191729918,
       0.08347430641751365, 0.026201862079995652, 0.04168068597107915, 0.1352680942007192,
       0.03614055044122551, 0.030447496750462644, 0.07180951106902754, 0.03968594759203002,
       0.05644735602737195, 0.07166639041345427, 0.11896200641502389, 0.06340744330857792,
       0.10408437769063927]

@test isapprox(w1.weights, w1t, rtol = 1.0e-5)
@test isapprox(w2.weights, w2t, rtol = 1.0e-5)
@test isapprox(hrc1 / lrc1, 1, atol = 1.6)
@test isapprox(hrc2 / lrc2, 20, atol = 3.2e0)
