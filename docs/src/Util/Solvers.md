# Solvers

Some optimisations and risk measures use [`JuMP`](https://github.com/jump-dev/JuMP.jl) models, and therefore require solvers to solve/compute. These can be specified by any container which implements the [`AbstractDict`](https://docs.julialang.org/en/v1/base/collections/#Base.AbstractDict) interfaces. This dictionary must contain key-value pairs for all solvers one wants to use. The key can be of any type, but the value must be an abstract dictionary with the following key-value pairs.

  - `:solver`: defines the solver to be used. One can also use [`JuMP.optimizer_with_attributes`](https://jump.dev/JuMP.jl/stable/api/JuMP/#optimizer_with_attributes) to direcly provide a solver with attributes already attached.
  - `:check_sol`: (optional) defines the keyword arguments passed on to [`JuMP.is_solved_and_feasible`](https://jump.dev/JuMP.jl/stable/api/JuMP/#is_solved_and_feasible) for accepting/rejecting solutions.
  - `:params`: (optional) defines solver-specific parameters/attributes.
  - `:add_bridges`: (optional) value of the `add_bridges` kwarg of [`JuMP.set_optimizer`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_optimizer), if not provided defaults to `true`.

```@example solvers_dict
using JuMP, Clarabel
solvers = Dict(
               # Key-value pair for the solver, solution acceptance 
               # criteria, model bridge argument, and solver attributes.
               :Clarabel => Dict(
                                 # Solver we wish to use.
                                 :solver => Clarabel.Optimizer,
                                 # (Optional) Solution acceptance criteria.
                                 :check_sol => (allow_local = true, allow_almost = true),
                                 # (Optional) Solver-specific attributes.
                                 :params => Dict("verbose" => false),
                                 # (Optional) Flag for adding JuMP bridges to JuMP.Model()
                                 # defaults to true (https://jump.dev/JuMP.jl/stable/api/JuMP/#Model).
                                 :add_bridges => false))
```

Users are able to provide multiple solvers by adding additional key-value pairs to the top-level dictionary as in the following snippet.

```@example solvers_dict
solvers = Dict(:Clarabel_1 => Dict(:solver => Clarabel.Optimizer,
                                   :check_sol => (allow_local = true, allow_almost = true),
                                   :params => Dict("verbose" => false)),
               # Provide solver with pre-attached attributes and no arguments 
               # for the `JuMP.is_solved_and_feasible` function.
               :Clarabel_2 => Dict(:solver => JuMP.optimizer_with_attributes(Clarabel.Optimizer,
                                                                             "max_step_fraction" => 0.75),
                                   # Do not add JuMP bridges to JuMP.Model()
                                   :add_bridges => false))
```

[`PortfolioOptimiser`](https://github.com/dcelisgarza/PortfolioOptimiser.jl) will iterate over the solvers until it finds the first one to successfully solve the problem. If a deterministic ordering is desired (for example when going from relaxed to strict tolerances), use an [`OrderedDict`](https://juliacollections.github.io/OrderedCollections.jl/dev/ordered_containers/#OrderedDicts).
