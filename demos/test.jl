using JuMP, Ipopt

m = Model(Ipopt.Optimizer)
@variable(m, x >= 0)
@NLexpression(m, risk, 3x^3 - 2x^2 + 5x + 5)
@NLobjective(m, Min, risk)
optimize!(m)
objective_value(m)

# Use this to add terms to nonlinear objectives and nonlinear expressions.
m = Model(Ipopt.Optimizer)
@variable(m, x >= 0)
@NLexpression(m, risk, 3x^3 - 2x^2 + 5x + 5)

# Add to expression that will become the objective before making the objective.
wak = add_nonlinear_expression(m, :($(m[:risk]) + log($(m[:x]) * 3)))
unregister(m, :risk)
@NLexpression(m, risk, wak)
@NLobjective(m, Min, risk)
optimize!(m)
objective_value(m)
print(m)

m = Model(Ipopt.Optimizer)
@variable(m, x >= 0)
@NLexpression(m, risk, 17exp(11x^3) - log(7x^2) + sin(13x) + 5)

@NLobjective(m, Min, risk)
println(fieldnames(typeof(m.nlp_data)))
# Nonlinear expressions are here, probably shouldn't delete them.
m.nlp_data.nlexpr
fieldnames(typeof(m.nlp_data.nlexpr[1]))
fieldnames(typeof(m.nlp_data.nlexpr[1].nd[1]))
m.nlp_data.nlexpr[1].nd
m.nlp_data.nlexpr[1].nd
m.nlp_data.nlexpr[1].const_values
m.nlp_data.nlobj

m.nlp_data.nlobj.nd[1]
m.nlp_data.nlobj.const_values

risk = add_nonlinear_expression(m, :(exp($(x))))

risk
# println(fieldnames(typeof(m)))
# display(m.nlp_data)
