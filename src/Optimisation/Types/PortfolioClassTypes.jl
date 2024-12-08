"""
```
abstract type PortClass end
```
"""
abstract type PortClass end

"""
```
abstract type BlackLittermanClass <: PortClass end
```
"""
abstract type BlackLittermanClass <: PortClass end

"""
```
struct Classic <: PortClass end
```
"""
struct Classic <: PortClass end

"""
```
@kwdef mutable struct FC <: PortClass
    flag::Bool = true
end
```
"""
mutable struct FC <: PortClass
    flag::Bool
end
function FC(; flag::Bool = true)
    return FC(flag)
end

"""
```
@kwdef mutable struct FM{T1 <: Integer} <: PortClass
    type::T1 = 1
end
```
"""
mutable struct FM{T1 <: Integer} <: PortClass
    type::T1
end
function FM(; type::Integer = 1)
    @smart_assert(type ∈ (1, 2))
    return FM{typeof(type)}(type)
end
function Base.setproperty!(obj::FM, sym::Symbol, val)
    if sym == :type
        @smart_assert(sym ∈ (1, 2))
    end
    return setfield!(obj, sym, val)
end

"""
```
@kwdef mutable struct BL{T1 <: Integer} <: BlackLittermanClass
    type::T1 = 1
end
```
"""
mutable struct BL{T1 <: Integer} <: BlackLittermanClass
    type::T1
end
function BL(; type::Integer = 1)
    @smart_assert(type ∈ (1, 2))
    return BL{typeof(type)}(type)
end
function Base.setproperty!(obj::BL, sym::Symbol, val)
    if sym == :type
        @smart_assert(sym ∈ (1, 2))
    end
    return setfield!(obj, sym, val)
end

"""
```
@kwdef mutable struct BLFM{T1 <: Integer} <: BlackLittermanClass
    type::T1 = 1
end
```
"""
mutable struct BLFM{T1 <: Integer} <: BlackLittermanClass
    type::T1
end
function BLFM(; type::Integer = 1)
    @smart_assert(type ∈ (1, 2, 3))
    return BLFM{typeof(type)}(type)
end
function Base.setproperty!(obj::BLFM, sym::Symbol, val)
    if sym == :type
        @smart_assert(sym ∈ (1, 2, 3))
    end
    return setfield!(obj, sym, val)
end

export Classic, FC, FM, BL, BLFM
