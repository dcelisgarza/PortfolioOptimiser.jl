---
name: Bug report
about: Create a report to help us improve
title: "[BUG]"
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is. Please place any code in `Julia` codeblocks like so:

````
```julia
# Code
```
````

**System information**
In order to know if the bug is platform- or version-specific we need some information. First, find your Julia version and system information. To get this go to the `Julia REPL` and type:

```julia
julia> versioninfo()
```

and paste the result here. For me this is:

```julia
Julia Version 1.11.3
Commit d63adeda50 (2025-01-21 19:42 UTC) 
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Windows (x86_64-w64-mingw32)       
  CPU: 16 × AMD Ryzen 7 1700 Eight-Core Processor
  WORD_SIZE: 64
  LLVM: libLLVM-16.0.6 (ORCJIT, znver1)
Threads: 1 default, 0 interactive, 1 GC (on 16 virtual cores)
```

Aside from this, we need to know what version of `PortfolioOptimise` you're using. This can also be obtained from the `REPL`, like so:

```julia
julia> ]status PortfolioOptimiser
```

and paste the results here. For me this is:

```julia
Status `C:\Users\Daniel Celis Garza\.julia\environments\v1.11\Project.toml`
  [748726ae] PortfolioOptimiser v0.1.0 `D:\Daniel Celis Garza\dev\PortfolioOptimiser.jl`
```

**Minimal Working Example**
Please provide a Minimal Working Example (MWE) showing the bug, as well as the error message if there is one. If there is any data, please make it available via a [Github Gist](https://gist.github.com/).

**Expected behavior**
Please describe the expected behaviour, or explain what you think the problem is. If there are academic papers or examples from other libraries, please provide the references and/or code here.

**Additional context**
Add any other context about the problem here.
