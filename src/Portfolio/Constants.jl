abstract type AbstractPortfolio end
const KellyRet = (:exact, :approx, :none)
const TrackingErrKinds = (:weights, :returns)
const ObjFuncs = (:min_risk, :utility, :sharpe, :max_ret)
const ValidTermination =
    (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED)
const PortClasses = (:classic,)
const OWAMethods = (:crra, :me, :mss, :msd)
const RiskMeasures = (
    :mv,
    :mad,
    :msv,
    :cvar,
    :wr,
    :flpm,
    :slpm,
    :mdd,
    :add,
    :cdar,
    :uci,
    :evar,
    :edar,
    :rdar,
    :rvar,
    :gmd,
    :tg,
    :rg,
    :rcvar,
    :rtg,
    :krt,
    :skrt,
)