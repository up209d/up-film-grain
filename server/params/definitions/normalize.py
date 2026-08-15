from __future__ import annotations

from ..param import Param

PARAMS: list[Param] = [
    # ------------------------------------------------------------ normalize
    # Step -2: above Colour Grading, and so above everything. One control, and
    # it is the only one in the app whose *settings* are measured from the
    # photograph rather than dialled in -- see
    # `server/engine/stages/normalize.py` for the metering and
    # `server/models/upload.py` for where it is cached.
    #
    # Ships off, like every other section, so the pipeline is still a
    # pass-through with nothing selected.
    Param(
        # Labelled "Enabled" rather than "Normalize": the section heading above
        # it already says Normalize, so repeating it on the only row in the
        # section reads as a stutter.
        "normalize", "Enabled", "Normalize",
        0.0, 1.0, 1.0, 0.0, "",
        "Auto levels, before anything else runs. Measures the photograph and "
        "corrects its lightness and its white balance, so an under- or "
        "over-exposed frame -- or one shot under the wrong lamp -- reaches the "
        "film pipeline looking like a normally exposed photograph. Everything "
        "below is calibrated around one, so a frame two stops under puts the "
        "grain in the wrong tones and makes the preset you picked read wrong.\n"
        "\n"
        "It corrects three things at once: the exposure, so the mid-tones land "
        "where they should; the colour of the light, so a tungsten or shade "
        "cast comes back to neutral without changing the brightness; and the "
        "dynamic range, so a photograph carrying more range than fits is "
        "compressed inward at both ends instead of being clipped.\n"
        "\n"
        "That last part is what keeps it from costing you anything. Brightening "
        "a dark frame would normally burn the highlights off the top and "
        "darkening a bright one would crush the shadows into black; instead "
        "both ends roll off along a curve that is strictly increasing "
        "everywhere, so two tones that differed before still differ after. "
        "Nothing is thrown away, and nothing ever reaches pure white or pure "
        "black. The compression is deliberately gentle -- enough to keep the "
        "ends, well short of the flat look a log video profile has, because "
        "every preset here expects a normally contrasted picture.\n"
        "\n"
        "A photograph that is already well exposed and neutral comes back "
        "essentially untouched: the correction is sized to what the frame "
        "actually needs, so there is no penalty for leaving this on.\n"
        "\n"
        "What it cannot do is invent detail that was never recorded. If a "
        "highlight was already blown out in the file, it arrives as a flat "
        "patch and stays one -- that is what Highlight Reconstruction below is "
        "for. Note the two interact: reconstruction finds blown areas by "
        "looking for channels pinned at the ceiling, and this stage runs first "
        "and moves them off it, so reach for one or the other rather than "
        "both. 0 = off.",
        toggle=True,
    ),
    Param(
        "highlight_priority", "Highlight Priority", "Normalize",
        0.0, 1.0, 0.01, 0.0, "",
        "How much of the original highlight detail to keep, at the expense of "
        "brightening those highlights. Only does anything with Normalize on.\n"
        "\n"
        "There is a real trade here and this slider is where you settle it. "
        "Lifting a dark frame's mid-tones means the bright end has nowhere to "
        "go -- it is already near white in the file -- so the correction has to "
        "compress it, and compressed highlights lose the fine separation that "
        "reads as texture in a sky, a cloud or a lit face.\n"
        "\n"
        "At 0 the whole frame is corrected together and the highlights take "
        "whatever compression that costs. Raise it and the bright areas are "
        "blended back toward what the original file recorded, in proportion to "
        "how bright they are -- so at 1 the highlights come back at their "
        "original tonal spacing, with every level the source had, while the "
        "mid-tones and shadows keep most of the correction. Measured on a dark "
        "photograph lifted two stops: the bright region carries 250 of its 256 "
        "distinct levels at 0 and all 256 at 1, and the mid-tones still land "
        "well above where they started.\n"
        "\n"
        "What it costs is brightness up there, not detail: highlights end up "
        "nearer their original level rather than the corrected one, so a scene "
        "with a lot of bright area will look less lifted overall as you raise "
        "this. It cannot recover a highlight that was already blown out in the "
        "file -- there is nothing recorded to come back -- so those stay white. "
        "0 = off.",
    ),
]
