module ModsExt

using CountableMagma, Mods

CountableMagma.gequal(a::Mods.AbstractMod,b::Mods.AbstractMod) = a == b

end # module
