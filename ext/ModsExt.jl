module ModsExt

import AbstractAnalysis, Mods

AbstractAnalysis.gequal(a::Mods.AbstractMod,b::Mods.AbstractMod) = a == b
AbstractAnalysis.isinvertible(n::Mods.AbstractMod) = Mods.is_invertible(n)

end # module
