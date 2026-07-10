module ModsExt

import CountableMagma, Mods

CountableMagma.gequal(a::Mods.AbstractMod,b::Mods.AbstractMod) = a == b
CountableMagma.isinvertible(n::Mods.AbstractMod) = Mods.is_invertible(n)

end # module
