from __future__ import annotations

from ..param import Param, GLOBAL_BLENDS

PARAMS: list[Param] = [
    # --------------------------------------------------------- global grain
    # Applied last -- see step 13 in engine.render(). Five layers, built from
    # one set of shape controls (Size Min, Size Max, Smoothness, Chroma) on five
    # different seeds, each with its own amount slider and its own mask. Every
    # amount ships at 0 so the section never alters an existing look until asked
    # for, and the blend mode ships on Add, which is what it has always done.
    #
    # The menu goes first because it governs all five sliders under it.
    Param(
        "global_blend", "Blend Mode", "Global Grain",
        0.0, float(len(GLOBAL_BLENDS) - 1), 1.0, 0.0, "",
        "How all five layers in this section are combined with the picture "
        "underneath, the way a layer's blend mode works in an image editor. "
        "The grain is treated as an image that is mid grey where there is no "
        "grain, and each layer's amount and mask together act as its "
        "opacity -- so every mode still fades to nothing as you take the "
        "sliders down.\n"
        "\n"
        "Add is the plain sum this section has always used, and the only one "
        "that is completely even-handed: it lifts and drops every tone by the "
        "same amount, which is why it can lift a black. Overlay and Soft Light "
        "are the two that behave like film -- both leave mid tones grainiest "
        "and taper off toward black and white on their own, Overlay firmly and "
        "Soft Light gently, so grain settles into the picture instead of "
        "sitting on it. Hard Light is Overlay driven from the grain rather "
        "than the image; it is much stronger and clips readily.\n"
        "\n"
        "Multiply and Screen are the odd pair, and worth knowing before you "
        "reach for them: neither has a neutral grey, so they darken or lighten "
        "the whole frame as well as texturing it, and the amount sliders are "
        "the only thing holding that back. Multiply grains the highlights and "
        "leaves shadows alone; Screen does the reverse. Use them low.",
        choices=GLOBAL_BLENDS,
    ),
    Param(
        "global_intensity", "Global Intensity", "Global Grain",
        0.0, 100.0, 0.5, 0.0, "%",
        "A flat grain layer over the finished frame, at one strength "
        "everywhere. Unlike the main grain it ignores the luminance band, the "
        "edge bias and the smooth-area guard, so it reaches skies, skin and "
        "blown highlights that those masks deliberately keep clean. 0 = off. "
        "Because nothing holds it back it bites far harder than the main "
        "Intensity slider at the same number -- 32 here measures 8% luminance "
        "sigma against 3.5% there. 5-20 is the usable range.\n"
        "\n"
        "This is the unmasked layer of the five. The four below it are the "
        "same grain following the picture instead of ignoring it; because this "
        "one goes on first and everywhere, nothing they mask away is ever left "
        "perfectly clean.",
    ),
    Param(
        "global_size", "Global Size Min", "Global Grain",
        0.1, 20.0, 0.05, 1.6, "px",
        "Clump diameter of the global layer, at full resolution -- the "
        "smallest a clump can be, and the only size that exists while Global "
        "Size Max stays at or below it. Set it apart from Clump Size and the "
        "two layers read as separate structures; match them and it just "
        "thickens the main grain. Large sizes are safe at any setting now: "
        "the layer is drawn as scattered grains on a lattice tilted off the "
        "pixel grid, so it no longer breaks into rectangular blocks the way "
        "it did past about 8px.",
        spatial=True,
    ),
    Param(
        "global_size_max", "Global Size Max", "Global Grain",
        0.1, 20.0, 0.05, 1.6, "px",
        "The largest a global-grain clump can be. At or below Global Size Min "
        "every clump renders at exactly Min; raised above it, each clump "
        "independently draws its own diameter somewhere between the two, so "
        "the layer reads as real crystals of differing sizes rather than one "
        "uniform grain. It is a range, not a switch -- widening it changes "
        "how much the sizes vary and nothing else about how the layer is "
        "drawn. A very wide gap leaves visible clear patches between clumps; "
        "real grain has them too, but narrow the gap if it reads as sparse.",
        spatial=True,
    ),
    Param(
        "global_chroma", "Global Chroma Grain", "Global Grain",
        0.0, 1.0, 0.01, 0.0, "",
        "The same job as Chroma Grain under Grain Structure, for this layer: "
        "0 = one monochrome field shared by all three channels, 1 = an "
        "independent field per channel so the layer carries colour speckle "
        "rather than pure luminance noise. Unlike that slider this one holds "
        "the layer's amplitude to within 3% across its whole range, so it "
        "changes colour without changing loudness. Its own slider because the "
        "two "
        "layers model different things -- the main grain is the negative's "
        "emulsion, where the dye layers are genuinely separate, while this one "
        "stands in for print stock and scanner noise and is often wanted "
        "neutral over a chromatic main grain. Ships at 0, which is what this "
        "layer has always been.\n"
        "\n"
        "Governs all five layers in the section, not just Global Intensity's. "
        "Note what it does and does not do: it colours each *grain*, keeping "
        "one grain's edge in the same place in all three channels. The four "
        "sliders below are the other thing -- separate grains in separate "
        "places, picked out by what colour the picture already is.",
    ),
    # The source-masked set -- see step 13 in engine.render(). Four more layers
    # of the same grain on four more seeds, each multiplied by an envelope read
    # off the picture. They stack on the flat layer above rather than replacing
    # it, so all four at 0 is exactly the layer every preset was dialled in
    # against, and with them up no area is ever left grain-free.
    #
    # **The colour names are the mask, never the output channel.** Each of these
    # is a full-colour grain field written into all three channels and taking
    # Chroma Grain like every other layer here; "Red" says only that it shows up
    # where the picture is red. Saying otherwise in a help string would be the
    # easiest thing in this section to get wrong.
    Param(
        "global_src_r", "Source Red", "Global Grain",
        0.0, 100.0, 0.5, 0.0, "%",
        "A grain layer that only shows up in the red parts of the picture -- "
        "stronger the redder and the brighter an area is, and absent from "
        "everything neutral. It is the *mask* that is red, not the grain: this "
        "is the same full-colour field as Global Intensity's, on its own seed, "
        "and it takes Chroma Grain like the rest.\n"
        "\n"
        "Its own seed is the point of having four of these -- a red-masked "
        "grain and a blue-masked grain are separate particles in separate "
        "places, the way three emulsion layers are. That is the other half of "
        "Global Chroma Grain, which colours one shared grain in place. Reach "
        "for these when you want grain to pick out what is already in the "
        "frame.\n"
        "\n"
        "On the same scale as Global Intensity before the mask takes its "
        "share, and the mask takes a lot: hue rarely dominates by more than "
        "0.3-0.5 in a real photograph, so expect to run these well above the "
        "number you would use up there. 0 = off, and off is free -- the field "
        "is never built.",
    ),
    Param(
        "global_src_g", "Source Green", "Global Grain",
        0.0, 100.0, 0.5, 0.0, "%",
        "The green member of the source-masked set: the same grain again, on "
        "its own seed, showing up in the green parts of the picture. Foliage, "
        "in other words, and not much else in most frames -- it is the one of "
        "the three that tends to find a single subject rather than spreading "
        "over the whole image.\n"
        "\n"
        "The three colour masks are mutually exclusive by construction: only "
        "one channel can be the dominant one at any pixel, so no two of them "
        "ever land on the same spot and turning all three up cannot pile them "
        "into a hot patch.",
    ),
    Param(
        "global_src_b", "Source Blue", "Global Grain",
        0.0, 100.0, 0.5, 0.0, "%",
        "The blue member of the source-masked set: the same grain again, on "
        "its own seed, showing up in the blue parts of the picture -- skies "
        "and water, mostly, which is exactly where the flat layer is most "
        "obvious and hardest to place. The blue-sensitive layer is the "
        "grainiest one in most real colour stocks, so running this above the "
        "other two is the closest thing here to that behaviour.",
    ),
    Param(
        "global_src_l", "Source Lightness", "Global Grain",
        0.0, 100.0, 0.5, 0.0, "%",
        "The odd one out, and the one to try first. Instead of a colour it "
        "follows *exposure*, and not as a ramp: it is loudest at mid grey and "
        "fades away toward both ends, so highlights stay clean and shadows "
        "stay clean and everything between them takes the grain. Full strength "
        "at mid grey, about a tenth of it near black and near white, nothing "
        "at all at either extreme.\n"
        "\n"
        "That is where film grain actually lives -- a blown highlight has no "
        "silver left to be grainy and a solid black has none developed -- so "
        "this is the layer that reads as emulsion rather than as noise laid "
        "over a photograph. Unlike the three above it, it does not care what "
        "colour anything is, so it works on a monochrome frame.",
    ),
    Param(
        "global_src_l_pivot", "Source Lightness Pivot", "Global Grain",
        0.05, 0.95, 0.01, 0.5, "",
        "Where Source Lightness peaks. That layer is a bell over exposure, and "
        "this is the tone the bell is centred on: at 0.5, the default, it is "
        "loudest at mid grey and falls away to nothing at both black and "
        "white. Drag it down and the grain moves into the shadows -- the peak "
        "lands on a darker grey and the falloff toward black gets steep while "
        "the run up to white stretches out. Drag it up and it moves into the "
        "highlights the same way.\n"
        "\n"
        "The two halves are stretched independently, so the layer still "
        "reaches exactly zero at both ends wherever you put the peak -- pure "
        "black and pure white are never grainy. Does nothing unless Source "
        "Lightness is above 0; it steers that layer and no other.",
    ),
    Param(
        "global_opacity", "Global Opacity", "Global Grain",
        0.0, 1.0, 0.01, 1.0, "",
        "How much of the global layer is mixed in. It multiplies with Global "
        "Intensity -- intensity is how coarse and strong the layer is in its "
        "own right, opacity is how far it is dialled back over the image. It "
        "governs the whole section, the four source-masked layers included, so "
        "it is the one dial that takes all five down together.",
    ),
    Param(
        "global_smooth", "Global Smoothness", "Global Grain",
        0.0, 1.0, 0.01, 0.0, "",
        "Blurs the Global Grain layer by up to half a clump, rounding its "
        "grains off and softening the boundaries between them. It used to be "
        "the cure for that layer breaking into rectangular blocks at large "
        "sizes; the field is no longer built on a lattice that does that, so "
        "this is now a shape control rather than a repair -- reach for it "
        "when the grain reads as too crisp. Strength is held constant as you "
        "raise it, so it changes the shape of the grain and not how much "
        "there is. Scaled to Global Size, so one setting stays right as you "
        "resize the clumps.",
    ),
    Param(
        "global_mottle", "Global Mottling", "Global Grain",
        0.0, 1.0, 0.01, 0.0, "",
        "How much the layer's grain strength varies from place to place. At 0 "
        "the grain is dead even -- the same loudness in every part of a smooth "
        "area, which is what you want when an unbroken tone is reading as "
        "blotchy rather than as film. Raise it and grain thins out in some "
        "regions and crowds into others, in soft patches several clumps "
        "across, the way real emulsion mottles.\n"
        "\n"
        "**0.6 is what this layer did before this slider existed**, so that is "
        "the number to put back if an older look is what you are after. It is "
        "not the default, because on smooth subjects -- a sky, a wall, a "
        "studio backdrop -- it lands the wrong side of the line: the noise "
        "stops reading as either organic or digital and starts reading as "
        "patches of nothing. On a busy frame the same setting is what keeps "
        "the layer from looking like an even screen when you step back from "
        "it. Which is right depends on the picture, which is why it is a "
        "slider and not a constant.\n"
        "\n"
        "Above about 0.8 it reads as patchiness in the photograph rather than "
        "as grain, at any subject. Strength is held flat as you raise it -- to "
        "within 0.3% across the whole range -- so this changes where the grain "
        "is and never how much of it there is. Governs all five layers in the "
        "section.",
    ),
    Param(
        "global_seed", "Global Seed", "Global Grain",
        0.0, 9999.0, 1.0, 0.0, "",
        "Reshuffles where every grain in this section falls, without touching "
        "the main grain you have already dialled in. All five layers reroll "
        "together and each stays independent of the other four -- what changes "
        "is the whole set at once, not their relationship to each other.\n"
        "\n"
        "It is an *offset* on the Seed slider under Grain Structure rather than "
        "a seed in its own right, which is why it starts at 0 and not at some "
        "arbitrary number. Two things fall out of that and both are wanted: "
        "moving Seed still rerolls the entire frame including this section, the "
        "way its help promises; and 0 is the exact layer every preset was "
        "dialled in against, whatever seed that preset happens to use.\n"
        "\n"
        "Reach for it when the global layer has landed a clump somewhere "
        "unhelpful -- across an eye, along a horizon -- and everything else is "
        "already right.",
    ),
]
