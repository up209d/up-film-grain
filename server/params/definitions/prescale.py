from __future__ import annotations

from ..param import Param

PARAMS: list[Param] = [
    # ------------------------------------------------------------- prescale
    # Step -3: above Normalize, and so above everything. Unlike every other
    # section this one is not a stage at all -- it never appears in
    # `engine/stages/render.py` and reserves nothing in `pad_for`, because it
    # changes the *frame the pipeline is handed* rather than the pixels in it.
    # The resample lives in `server/models/upload.py`, above the engine, which
    # is what keeps tile independence and scale invariance untouched by it.
    #
    # It is the other answer to the question `reference_mp` answers. Size
    # Scaling moves the *parameters* to fit the photograph; this moves the
    # *photograph* to fit the parameters, and since every shipped preset is
    # stamped at 24MP, prescaling to 24 collapses that rescale to exactly 1.00x.
    Param(
        # Labelled "Enabled" rather than "Prescale" for the reason Normalize's
        # switch is: the section heading above it already says the rest.
        "prescale", "Enabled", "Prescaling Source",
        0.0, 1.0, 1.0, 1.0, "",
        "Resample the photograph to a fixed size before anything else runs, so "
        "the whole pipeline sees the same resolution whatever came out of the "
        "camera. On by default.\n"
        "\n"
        "Why it matters: every length in here -- clump size, every radius, "
        "jitter, speck and scratch size -- is a number of pixels, so the same "
        "settings resolve finer, denser grain on a 50MP frame than on a 12MP "
        "one. The look you dialled in follows the file's resolution rather than "
        "the photograph. With this on, it does not: a 50MP frame is reduced and "
        "a 12MP frame is enlarged to the target below, and both then behave "
        "exactly as though that were the file you opened.\n"
        "\n"
        "It is the same problem Size Scaling solves from the other end. That "
        "rescales every length to fit the photograph; this rescales the "
        "photograph to fit the lengths. Every preset shipped here records that "
        "it was dialled in at 24MP, so with the target at 24 the two agree and "
        "Size Scaling's factor sits at 1.00x -- which is the point, and is why "
        "this section sits directly above it.\n"
        "\n"
        "The honest cost: reducing a large photograph throws away real detail, "
        "and enlarging a small one adds none -- an upscaled 6MP frame is a 6MP "
        "photograph on a 24MP grid, and the grain drawn on top of it is the "
        "only thing there at 24MP. 0 = off, and the photograph is used at "
        "whatever size it arrived.",
        toggle=True,
    ),
    Param(
        "prescale_mp", "Target", "Prescaling Source",
        1.0, 120.0, 0.1, 24.0, "MP",
        "The size the photograph is resampled to, in megapixels. Only does "
        "anything with Prescaling on.\n"
        "\n"
        "24 is the default and the number every preset here was dialled in at, "
        "so it is the setting at which a preset does what its author intended. "
        "Raise it for finer, denser grain relative to the frame and lower it "
        "for coarser, larger grain -- which is the same axis Size Scaling's "
        "factor moves along, approached from the other side.\n"
        "\n"
        "Only the megapixel count is set here; the aspect ratio is the "
        "photograph's own and never changes. 120 is the largest image the "
        "server accepts at all, so the top of this slider is a real ceiling "
        "rather than a chosen one.\n"
        "\n"
        "Cost scales with it. The frame is resampled once per photograph and "
        "kept, so moving any other slider is as fast as it ever was -- but "
        "every render after that is over this many pixels, and a large target "
        "on a small photograph buys resolution that carries no detail.",
    ),
    Param(
        # A two-entry menu rather than a checkbox because neither answer is
        # "off": both write a file, and the labels are what tell them apart.
        #
        # Index order is load-bearing in every saved preset -- 0 has to stay
        # the prescaled size. Appending is safe, reordering is not.
        "prescale_output", "Export size", "Prescaling Source",
        0.0, 1.0, 1.0, 0.0, "",
        "Which pixel dimensions the exported file is written at. Changes "
        "nothing on screen -- the preview is scaled by the browser either way "
        "-- and only does anything with Prescaling on.\n"
        "\n"
        "**Prescaled size** writes the file at the target above, so a 50MP "
        "photograph exports at 24MP and a 6MP one exports at 24MP too. This is "
        "the honest one: it is the frame that was actually rendered, and every "
        "pixel in it was computed rather than interpolated.\n"
        "\n"
        "**Photo's own size** resamples the finished render back to the "
        "dimensions of the file you opened, for when something downstream "
        "expects them. It is a resample of finished grain rather than a render "
        "at that size, so going up it is soft and going down it averages grain "
        "away -- the texture you judged on screen is not quite the texture in "
        "the file. Exports carry a size tag in the filename either way, so two "
        "files from one photograph are never indistinguishable in a folder.",
        choices=("Prescaled size", "Photo's own size"),
    ),
]
