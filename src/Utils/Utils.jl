# Copywrite (c) 2025
# Author: Daniel Celis Garza <daniel.celis.garza@gmail.com>
# SPDX-License-Identifier: MIT

for (op, name) ∈
    zip((SD, Variance, MAD, SSD, SVariance, FLPM, WR, VaR, CVaR, EVaR, EVaRRG, RLVaR,
         RLVaRRG, DaR, MDD, ADD, CDaR, UCI, EDaR, RLDaR, DaR_r, MDD_r, ADD_r, CDaR_r, UCI_r,
         EDaR_r, RLDaR_r, Kurt, SKurt, GMD, RG, CVaRRG, TG, TGRG, OWA, BDVariance, NQSkew,
         NQSSkew, NSkew, NSSkew, Equal, WCVariance, TCM, TLPM, FTCM, FTLPM, Skewness,
         SSkewness, Kurtosis, SKurtosis, SchurParams, DRCVaR, NCOArgs, TrackingRM,
         TurnoverRM, PortOptSolver, VaRRG),
        ("SD", "Variance", "MAD", "SSD", "SVariance", "FLPM", "WR", "VaR", "CVaR", "EVaR",
         "EVaRRG", "RLVaR", "RLVaRRG", "DaR", "MDD", "ADD", "CDaR", "UCI", "EDaR", "RLDaR",
         "DaR_r", "MDD_r", "ADD_r", "CDaR_r", "UCI_r", "EDaR_r", "RLDaR_r", "Kurt", "SKurt",
         "GMD", "RG", "CVaRRG", "TG", "TGRG", "OWA", "BDVariance", "NQSkew", "NQSSkew",
         "NSkew", "NSSkew", "Equal", "WCVariance", "TCM", "TLPM", "FTCM", "FTLPM",
         "Skewness", "SSkewness", "Kurtosis", "SKurtosis", "SchurParams", "DRCVaR",
         "NCOArgs", "TrackingRM", "TurnoverRM", "PortOptSolver", "VaRRG"))
    eval(quote
             Base.iterate(S::$op, state = 1) = state > 1 ? nothing : (S, state + 1)
             function Base.String(s::$op)
                 return $name
             end
             function Base.Symbol(::$op)
                 return Symbol($name)
             end
             function Base.length(::$op)
                 return 1
             end
             function Base.getindex(S::$op, ::Any)
                 return S
             end
             function Base.view(S::$op, ::Any)
                 return S
             end
         end)
end
