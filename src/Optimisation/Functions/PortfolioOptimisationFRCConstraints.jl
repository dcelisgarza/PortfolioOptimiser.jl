function set_frc_w(port, flag, w_ini)
    model = port.model
    f_returns = port.f_returns
    loadings = port.loadings
    regression_type = port.regression_type
    if flag
        b1, b2 = factors_b1_b2_b3(loadings, f_returns, regression_type)[1:2]
        N = size(port.returns, 2)
        Nf = size(b1, 2)
        @variables(model, begin
                       w1[1:Nf]
                       w2[1:(N - Nf)]
                   end)
        @expression(model, w, b1 * w1 + b2 * w2)
    else
        b1 = factors_b1_b2_b3(loadings, f_returns, regression_type)[1]
        Nf = size(b1, 2)
        @variable(model, w1[1:Nf])
        @expression(model, w, b1 * w1)
    end
    set_w_ini(w1, w_ini)
    return b1
end
