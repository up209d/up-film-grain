from __future__ import annotations

from ..param import Param

PARAMS: list[Param] = [
    # ---------------------------------------------------------- film texture
    # Physical damage to the film, not emulsion behaviour -- applied dead last
    # and masked by nothing. See step 15 in engine.render().
    Param(
        "dust", "Dust Count", "Film Texture",
        0.0, 400.0, 1.0, 0.0, "",
        "How many specks land on the frame -- a count, not a strength, so it "
        "means the same thing whatever the image size. **Exact**: the specks "
        "are drawn one at a time from a list anchored to the frame, so 20 here "
        "is twenty specks, not roughly twenty. Raising it adds specks and "
        "leaves the ones already there where they were. 0 = none.",
    ),
    Param(
        "dust_balance", "Dust Dark / Light", "Film Texture",
        -1.0, 1.0, 0.01, 0.0, "",
        "Which way the population leans. **-1 is every speck dark, +1 is every "
        "speck bright, 0 is an even mix.** Dark specks are opaque motes sitting "
        "on the emulsion; bright ones are pinholes in it and lint on the "
        "scanner glass, and a frame of only dark specks reads as sensor dirt "
        "rather than as film.\n"
        "\n"
        "The split is exact and it converts specks *in place*: moving this "
        "changes which of the specks are bright without moving any of them, so "
        "you can find the ratio you want without the frame reshuffling under "
        "you. Dust Count stays the total either way.",
    ),
    Param(
        "dust_size", "Dust Size", "Film Texture",
        0.5, 120.0, 0.05, 2.0, "px",
        "Mean speck diameter at full resolution. Small is scanner dust; large "
        "is lint and debris on the negative. Individual specks are drawn "
        "around this rather than all cut to it -- real debris comes in a range "
        "of sizes, and a frame of identically-sized specks is the clearest "
        "sign the texture was generated.",
        spatial=True,
    ),
    Param(
        "dust_irregular", "Dust Irregularity", "Film Texture",
        0.0, 1.0, 0.01, 0.0, "",
        "How far a speck's outline strays from a clean ellipse. **0 is a "
        "circle or an oval and nothing else** -- specks still vary in size, in "
        "how elongated they are and in which way they point, but their edges "
        "are smooth. Turning it up dents the outline with three angular "
        "harmonics, so specks become chipped, notched and irregular the way "
        "real debris is; at 1 the deepest notches reach most of the way to the "
        "centre and a speck reads as a fleck of something rather than as a "
        "dot.\n"
        "\n"
        "The amount is drawn per speck around this, so a frame is a population "
        "of shapes rather than one lumpiness stamped out over and over.",
    ),
    Param(
        "dust_opacity", "Dust Opacity", "Film Texture",
        0.0, 1.0, 0.01, 0.85, "",
        "How much of the photograph a speck hides at full strength. Separate "
        "from luminosity: opacity is coverage, luminosity is what colour the "
        "speck is, so a solid grey mote and a faint black veil are different "
        "things rather than the same number twice.",
    ),
    Param(
        "dust_opacity_var", "Dust Opacity Variation", "Film Texture",
        0.0, 1.0, 0.01, 0.6, "",
        "How much opacity differs between specks. At 1 the faintest are "
        "barely there while the strongest are solid -- real dust sits at "
        "different depths and in different thicknesses.",
    ),
    Param(
        "dust_lum_var", "Dust Luminosity Variation", "Film Texture",
        0.0, 1.0, 0.01, 0.5, "",
        "How much the specks differ in tone. Dark motes spread across black "
        "to mid-grey and bright pinholes across off-white to white, so each "
        "population varies within itself without the two swapping places.",
    ),
    Param(
        "dust_soften", "Dust Softness", "Film Texture",
        0.0, 5.0, 0.01, 0.35, "",
        "How far out of focus the specks are. Debris sits at different "
        "depths, so this is a *spread* rather than a uniform blur -- some "
        "specks stay crisp and others go soft at any setting. Soft specks "
        "also come out fainter, which is what out-of-focus debris actually "
        "does. 0 = all crisp.\n"
        "\n"
        "**1 is where this used to stop.** Everything below it is unchanged, so "
        "a value carried over from before means what it always did. Past 1 the "
        "edge grows wider than the speck itself: the solid core goes, and a "
        "speck becomes a soft smudge of tone rather than a mark with an "
        "outline. That is what debris well off the focal plane looks like, and "
        "it is the range to use when the specks are reading as too crisp and "
        "too drawn.",
    ),
    Param(
        "scratches", "Scratch Count", "Film Texture",
        0.0, 60.0, 1.0, 0.0, "",
        "Roughly how many gouges run down the frame. They follow the "
        "direction of travel and print bright, because a scratch through "
        "the emulsion lets light straight through, and they break along "
        "their length rather than ruling the whole frame. 0 = none.",
    ),
    Param(
        "scratch_width", "Scratch Width", "Film Texture",
        0.3, 20.0, 0.05, 1.0, "px",
        "Width of a scratch at full resolution. Hairline values are the "
        "convincing ones; wide reads as damage rather than wear.",
        spatial=True,
    ),
    Param(
        "scratch_soften", "Scratch Softness", "Film Texture",
        0.0, 1.0, 0.01, 0.35, "",
        "Softens the gouges. A perfectly crisp scratch is the clearest sign "
        "the texture was generated -- real ones are cut at different depths "
        "and the scanner only focuses on one plane. Softness varies between "
        "scratches but stays constant along each one, so a scratch never "
        "fades in and out down its own length. 0 = all crisp.",
    ),
    Param(
        "hair", "Hair Count", "Film Texture",
        0.0, 40.0, 1.0, 0.0, "",
        "How many hairs and fibres are lying on the frame, printing as dark "
        "wandering filaments. **Exact**: one hair is one hair. Each is drawn "
        "as its own filament with its own direction, curl and taper, so "
        "raising the count adds hairs and leaves the ones already there "
        "untouched. 0 = none.",
    ),
    Param(
        "hair_length", "Hair Length", "Film Texture",
        20.0, 600.0, 5.0, 160.0, "px",
        "Mean hair length at full resolution -- independent of how many there "
        "are, and drawn around rather than cut to, so a frame carries long "
        "fibres and short ones. It also sets how far a hair curls and wanders "
        "over its own length, because a longer filament bends more.",
        spatial=True,
    ),
    Param(
        "hair_soften", "Hair Softness", "Film Texture",
        0.0, 1.0, 0.01, 0.35, "",
        "Softens the filaments. A hair lying on the glass is sharp; one on "
        "the negative a layer away is not, so the population wants both. 0 "
        "= all crisp.",
    ),
    Param(
        "light_leak", "Leak Count", "Film Texture",
        0.0, 12.0, 1.0, 0.0, "",
        "Roughly how many light leaks reach in from the frame edges. "
        "Counted against the perimeter rather than the area, because that "
        "is where they happen. Added in linear light so it behaves like "
        "light falling on the emulsion, not a gradient painted over the "
        "picture. 0 = none -- and so is anything below 1, because you "
        "cannot render a fraction of a leak. A hand-edited file holding "
        "0.05 here renders nothing at all rather than a faint leak.",
    ),
    Param(
        "leak_strength", "Leak Strength", "Film Texture",
        0.0, 3.0, 0.01, 1.0, "",
        "How much light each leak lets in. The response saturates one dye "
        "layer at a time, so this is not just an opacity: a faint leak is deep "
        "red because only the red-sensitive layer caught enough light, and "
        "pushing it up takes the core through orange and yellow to white while "
        "leaving the colour in the falloff. Past about 1.5 most leaks have a "
        "blown white core, which is the 'sun got in the back' look.",
    ),
    Param(
        "leak_size_min", "Leak Size Min", "Film Texture",
        5.0, 3000.0, 1.0, 250.0, "px",
        "How far the *smallest* leak reaches in from the frame edge, at full "
        "resolution. Each leak picks its own reach somewhere between this and "
        "the maximum, so the two together are what makes a frame of leaks look "
        "accidental rather than stamped -- set them equal and every leak comes "
        "in exactly as far as the next.",
        spatial=True,
    ),
    Param(
        "leak_size_max", "Leak Size Max", "Film Texture",
        5.0, 3000.0, 1.0, 850.0, "px",
        "How far the *largest* leak reaches in. Given below the minimum the "
        "two simply swap, so you can drag either one past the other without "
        "the leaks disappearing. Corners bloom further than edge midpoints "
        "whatever this says -- that is where the cassette mouth and the film "
        "gate actually let light past. Reach is capped at half the frame's "
        "short side, which is the distance at which a leak just dies in the "
        "middle: past that it would leave a floor over the whole frame, and a "
        "leak that fogs the centre reads as a bad exposure rather than a leak.",
        spatial=True,
    ),
    Param(
        "leak_feather", "Leak Feather", "Film Texture",
        1.0, 1500.0, 1.0, 180.0, "px",
        "How far in from the border a leak has faded to *half* strength, at "
        "full resolution -- so it is a distance you can see rather than an "
        "abstract softness. Small against the size gives a tight bright rim "
        "hugging the edge; around half the size gives a straight ramp; most of "
        "the way to the size gives a broad wash that hardly falls off until it "
        "ends. Because it is absolute, the same feather is a wash on a small "
        "leak and a rim on a large one, which is what stops a frame of "
        "differently-sized leaks looking like one shape at several scales.\n"
        "\n"
        "It softens the leak's *other* edge too -- the transition along the "
        "border where one leak stops. A leak has two visible edges and "
        "softening only one still reads as a painted shape.",
        spatial=True,
    ),
    Param(
        "leak_variation", "Leak Variation", "Film Texture",
        0.0, 1.0, 0.01, 0.7, "",
        "How much one leak differs from the next in everything *except* size, "
        "which Leak Size Min and Max now set directly: how hard its edge is, "
        "how broad or tight its halo is, and how strong it arrives. 0 makes "
        "every leak identical in those respects; 1 is a wide spread. Light "
        "gets in through whatever gap it finds, and no two gaps are alike.",
    ),
    Param(
        "leak_hue", "Leak Hue", "Film Texture",
        0.0, 1.0, 0.01, 0.3, "",
        "0 = deep red, the classic 35mm canister leak. 1 = amber, closer to "
        "daylight getting in around a loose back.",
    ),
    Param(
        "texture_seed", "Texture Seed", "Film Texture",
        0.0, 9999.0, 1.0, 77.0, "",
        "Re-rolls where every mark lands. Separate from the grain Seed on "
        "purpose: you will want to reshuffle the damage without disturbing "
        "grain you have already dialled in.",
    ),
]
