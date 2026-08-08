from __future__ import annotations

from ..param import Param

PARAMS: list[Param] = [
    # ----------------------------------------------------- edge destruction
    # (scatter and micro-blur -- formerly their own "Optical" group, merged in
    # here 2026-08-04 on request; the engine's step numbering is unaffected,
    # this is a UI grouping only.)
    #
    # Scatter first, micro-blur last, in the panel and in the pipeline alike --
    # see step 1 in engine.render(). The order is the point: scatter gets the
    # source's own detail to take apart, and the blur then averages what is
    # left rather than handing scatter a frame that is already smooth.
    #
    # Scatter: diffusion resolved as discrete deflections instead of as an
    # average. See _scatter for why that is not a blur.
    Param(
        "scatter", "Scatter", "Edge Destruction",
        0.0, 1.0, 0.01, 0.0, "",
        "Spreads detail into the neighbouring pixels *without* averaging "
        "anything, so the picture loses its digital exactness while keeping "
        "its bite. Every displaced pixel is an exact copy of a real pixel "
        "nearby -- no in-between values are invented, so contrast, grit and "
        "texture come through at full strength where a blur of the same reach "
        "would have flattened them. The number is the fraction of the frame "
        "that moves: 0.3 relocates three pixels in ten and leaves the rest "
        "exactly where they were. It is deliberately a coverage, not a blend "
        "-- blending a moved pixel with its original *is* averaging, and that "
        "is the one thing this stage must never do. Smooth regions are "
        "untouched for free: shuffling pixels that already match their "
        "neighbours changes nothing, so skies and skin stay clean while "
        "detail is the only thing that comes apart. 0 = off.",
    ),
    Param(
        "scatter_radius", "Scatter Reach", "Edge Destruction",
        0.5, 24.0, 0.1, 3.0, "px",
        "How far a displaced pixel travels, at full resolution. Small reads "
        "as an emulsion that will not quite resolve; large tears detail into "
        "streaks and crumbs. It is also what decides *which* structure comes "
        "apart, because moving a pixel only changes anything where the "
        "picture varies over the distance travelled: a short reach disorders "
        "fine texture and leaves shapes standing, a long one starts taking "
        "the shapes with it.",
        spatial=True,
    ),
    Param(
        "scatter_pattern", "Scatter Pattern", "Edge Destruction",
        0.0, 8.0, 1.0, 0.0, "",
        "Where a displaced pixel is allowed to land -- the stencil. Restricting "
        "the choice is what makes the result read as a *structure* rather than "
        "as noise: detail smears the way the shape says and nowhere else.\n"
        "\n"
        "Any is isotropic and reads as plain diffusion. Cross, Diagonal and Box "
        "are the 4-, 45- and 8-neighbour stencils. Diamond keeps every angle "
        "but reaches furthest along the axes and pulls in on the diagonals, so "
        "detail spreads as a rhombus rather than a disc. Donut holds a hole "
        "open in the middle -- nothing lands near where it started, so detail "
        "is thrown outward and hollowed out, and it stays hollow whatever Reach "
        "Spread is set to. Star is eight spokes with every other one running "
        "short, which is the shape a cross filter flares into. Horizontal and "
        "Vertical are the extreme case, a one-axis slip that leaves edges "
        "running along that axis completely untouched.",
        choices=("Any", "Cross", "Diagonal", "Box", "Diamond", "Donut",
                 "Star", "Horizontal", "Vertical"),
    ),
    Param(
        "scatter_spread", "Reach Spread", "Edge Destruction",
        0.0, 1.0, 0.01, 1.0, "",
        "Whether every displaced pixel travels the full reach or a share of "
        "it. 0 is a shell -- everything lands on the edge of the pattern's "
        "shape, which hollows detail out into an outline and is the harshest "
        "setting here. 1 fills the shape inward, with distances spread evenly "
        "from nothing up to the reach, which reads as diffusion rather than as "
        "an outline. Donut is the exception by design: it holds its hole open "
        "at any setting, so this only decides how thick its ring is.",
    ),
    Param(
        "scatter_cell", "Scatter Clump", "Edge Destruction",
        0.1, 5.0, 0.1, 1.0, "px",
        "How big a piece of the picture moves as one. At 1 every pixel "
        "chooses for itself and the image crumbles; larger values move whole "
        "tiles of detail intact, so structure survives the trip and lands "
        "somewhere else. Past about 4px the tiles start reading as tiles -- "
        "which is a look, a shattered plate rather than a soft one, but it is "
        "no longer subtle. Held in full-res pixels like every other length.\n"
        "\n"
        "Below one *working* pixel there is nothing left to resolve -- one "
        "choice per pixel is already the finest this can be -- so the bottom "
        "of the range is only reachable through supersampling, which is what "
        "makes a working pixel smaller than a real one. At supersample 2 that "
        "puts the floor at 0.5; below it every setting renders identically.",
        spatial=True,
    ),
    Param(
        "micro_blur", "Micro-Blur", "Edge Destruction",
        0.0, 3.0, 0.01, 0.45, "px",
        "Light diffusion through the gel layers, as an average: every pixel "
        "is mixed with its neighbours. That is the smooth half of diffusion, "
        "and it costs texture along with the edges -- Scatter above is the "
        "same physics without the averaging. Last in the light path, so it "
        "averages whatever scatter has already pulled apart rather than "
        "handing scatter a frame that is smooth before it starts. Applied to "
        "the base image before grain injection so grain stays sharp against a "
        "soft base.",
        spatial=True,
    ),
]
