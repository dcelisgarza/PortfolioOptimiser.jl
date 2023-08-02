function sharpe_risk(
    w,
    cov,
    returns,
    rm = :mv,
    rf = 0.0,
    alpha = 0.05,
    a_sim = 100,
    beta = Inf,
    b_sim = -1,
    kappa = 0.3,
    solver = nothing,
)
    if rm == :mv
        risk = dot(w, cov, w)
    elseif rm == :msd
        risk = sqrt(dot(w, cov, w))
    elseif rm == :mad
    end
end