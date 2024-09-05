# Risk measaures

Some risk measures require solvers to compute.

```julia
solvers = Dict(
               # Key-value pair for the solver, solution acceptance 
               # criteria, and solver attributes.
               :Clarabel => Dict(
                                 # Solver we wish to use.
                                 :solver => Clarabel.Optimizer,
                                 # (Optional) Solution acceptance criteria.
                                 :check_sol => (allow_local = true, allow_almost = true),
                                 # (Optional) Solver-specific attributes.
                                 :params => Dict("verbose" => false)))
```

The dictionary contains a key value pair for each solver (plus optional solution acceptance criteria and optional attributes) we want to use.

  - `:solver`: defines the solver to use. One can also use [`JuMP.optimizer_with_attributes`](https://jump.dev/JuMP.jl/stable/api/JuMP/#optimizer_with_attributes) to direcly provide a solver with attributes already attached.
  - `:check_sol`: (optional) defines the keyword arguments passed on to [`JuMP.is_solved_and_feasible`](https://jump.dev/JuMP.jl/stable/api/JuMP/#is_solved_and_feasible) for accepting/rejecting solutions.
  - `:params`: (optional) defines solver-specific parameters.

Users are also able to provide multiple solvers by adding additional key-value pairs to the top-level dictionary/tuple as in the following snippet.

```julia
using JuMP
solvers = Dict(:Clarabel => Dict(:solver => Clarabel.Optimizer,
                                 :check_sol => (allow_local = true, allow_almost = true),
                                 :params => Dict("verbose" => false)),
               # Provide solver with pre-attached attributes and no arguments 
               # for the `JuMP.is_solved_and_feasible` function.
               :COSMO => Dict(:solver => JuMP.optimizer_with_attributes(COSMO.Optimizer,
                                                                        "maxiter" => 5000)))
```

`PortfolioOptimiser` will iterate over the solvers until it finds the first one to successfully solve the problem.

## Public

```@autodocs
Modules = [PortfolioOptimiser]
Public = true
Private = false
Pages = ["RiskMeasures/Types/RiskMeasureTypes.jl", "RiskMeasures/Functions/MiscRiskMeasureFunctions.jl", "RiskMeasures/Functions/RiskValue.jl", "RiskMeasures/Functions/RiskStatistics.jl"]
```

## Private

```@autodocs
Modules = [PortfolioOptimiser]
Public = false
Private = true
Pages = ["RiskMeasures/Types/RiskMeasureTypes.jl", "RiskMeasures/Functions/MiscRiskMeasureFunctions.jl", "RiskMeasures/Functions/RiskValue.jl", "RiskMeasures/Functions/RiskStatistics.jl"]
```
