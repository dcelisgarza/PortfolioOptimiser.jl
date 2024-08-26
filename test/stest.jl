using CSV, TimeSeries, JuMP, Test, Clarabel, StatsBase, PyCall, DataFrames,
      PortfolioOptimiser, Clustering

function find_rtol(a1, a2)
    for rtol ∈
        [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
         5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0]
        if isapprox(a1, a2; rtol = rtol)
            println(", rtol = $(rtol)")
            break
        end
    end
end

function get_rtol(a1, a2)
    for rtol ∈
        [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
         5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0]
        if isapprox(a1, a2; rtol = rtol)
            return rtol
            break
        end
    end
end

function find_atol(a1, a2)
    for atol ∈
        [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
         5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 2e0]
        if isapprox(a1, a2; atol = atol)
            println(", atol = $(atol)")
            break
        end
    end
end

find_rtol(
          #    
          [0.011338159104388727, 0.02750621719073114, 0.0029200109616546338,
           0.0159492349524423, 0.0001380666631561357, 0.02359645331970738,
           3.375390704002271e-7, 0.13589840786724114, 1.295705479493299e-6,
           1.857923177105043e-5, 0.30742598466946003, 5.27490162202497e-7,
           2.642677194741749e-7, 0.12823832729956616, 1.1003513110979598e-6,
           5.475499873909498e-5, 0.008910963808331375, 0.2073338665016158,
           2.285254340245168e-6, 0.1306651628231121],
          [0.01133815655163677, 0.02750621458048178, 0.0029200084390756014,
           0.01594923238329056, 0.00013806415234585061, 0.02359645072335987,
           3.5931355454602553e-7, 0.13589840487159488, 1.2935884507306267e-6,
           1.8576729294314388e-5, 0.3074259810639388, 5.282334179287036e-7,
           2.827308372529218e-7, 0.12823832433115573, 1.1025873365818162e-6,
           5.4752490550460225e-5, 0.008910961264212489, 0.2073338632519772,
           2.2828674158038536e-6, 0.13066515984607294]
          #
          )

str = "println(\""
for i ∈ 1:11
    if i ∈ (7, 10, 11)
        continue
    end
    str *= "w$(i)t = \$(w$(i).weights)\\n"
end
str *= "\")"
println(str)

println("covt15n = reshape($(vec(cov15n)), $(size(cov15n)))")

function f(rpe, warm)
    rpe = rpe * 10

    if warm == 1
        rpew = 0.6 * rpe[1]
        repsw = range(; start = 6, stop = 10)
    elseif warm == 2
        rpew = [0.5; 0.7] * rpe[1]
        repsw = [range(; start = 6, stop = 10), range(; start = 4, stop = 6)]
    elseif warm == 3
        rpew = [0.45; 0.65; 0.85] * rpe[1]
        repsw = [range(; start = 6, stop = 10), range(; start = 4, stop = 6),
                 range(; start = 3, stop = 4)]
    elseif warm == 4
        rpew = [0.3; 0.5; 0.7; 0.9] * rpe[1]
        repsw = [range(; start = 6, stop = 10), range(; start = 4, stop = 6),
                 range(; start = 3, stop = 4), range(; start = 2, stop = 3)]
    end

    return rpew, repsw
end

r = collect(range(; start = 9, stop = 9.5, length = 3))
p, reps = f(r, 1)
display([p reps])
display(r * 10)

# %%

f(w, h, l) = w * h * l

alex = 2 * f(32, 56, 7) + 3 * f(56, 32, 13)
trotten = 3 * f(15.6, 47, 40)

using Optimization, OptimizationOptimJL, AverageShiftedHistograms
const ASH = AverageShiftedHistograms

function errPDF(x, vals; kernel = ASH.Kernels.gaussian, m = 10, n = 1000, q = 1000)
    e_min, e_max = x[1] * (1 - sqrt(1.0 / q))^2, x[1] * (1 + sqrt(1.0 / q))^2
    rg = range(e_min, e_max; length = n)
    pdf1 = q ./ (2 * pi * x[1] * rg) .*
           sqrt.(clamp.((e_max .- rg) .* (rg .- e_min), 0, Inf))

    e_min, e_max = x[1] * (1 - sqrt(1.0 / q))^2, x[1] * (1 + sqrt(1.0 / q))^2
    res = ash(vals; rng = range(e_min, e_max; length = n), kernel = kernel, m = m)
    pdf2 = [ASH.pdf(res, i) for i ∈ pdf1]
    pdf2[.!isfinite.(pdf2)] .= 0.0
    sse = sum((pdf2 - pdf1) .^ 2)

    return sse
end
function find_max_eval(vals, q; kernel = ASH.Kernels.gaussian, m::Integer = 10,
                       n::Integer = 1000, args = (), kwargs = ())
    res = Optim.optimize(x -> errPDF(x, vals; kernel = kernel, m = m, n = n, q = q), 0.0,
                         1.0, args...; kwargs...)

    x = Optim.converged(res) ? Optim.minimizer(res) : 1.0

    e_max = x * (1.0 + sqrt(1.0 / q))^2

    return e_max, x
end

function find_max_eval2(vals, q; kernel = ASH.Kernels.gaussian, m::Integer = 10,
                        n::Integer = 1000, args = (), kwargs = ())
    u0 = [0.0]
    p = [vals, kernel, m, n, q]
    erpdf(u, p) = errPDF(u, p[1]; kernel = p[2], m = p[3], n = p[4], q = p[5])
    prob = OptimizationProblem(erpdf, u0, p; lb = [0.0], ub = [1.0])
    sol = solve(prob, SAMIN())

    # res = Optim.optimize(x -> errPDF(x, vals; kernel = kernel, m = m, n = n, q = q), 0.0,
    #                      1.0, args...; kwargs...)
    x = sol.u[1]

    # x = Optim.converged(res) ? Optim.minimizer(res) : 1.0

    e_max = x * (1.0 + sqrt(1.0 / q))^2

    return e_max, x
end

X = randn(100, 20)
X = cov(X)
vals, vecs = eigen(X)

max_val = find_max_eval2(vals, 100 / 20)[1]

using PortfolioOptimiser, Makie
