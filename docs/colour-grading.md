# Colour Grading

Step −1, the only block above pre-blur. Everything in it ships at 0, so with
nothing selected the section is a colour pass-through. Tone Response's split
tone is filed here too, because it is the same kind of control.

Split on 2026-08-08 to keep each file readable; the content is unchanged.

| file | what is in it |
|---|---|
| [colour-grading/luts.md](colour-grading/luts.md) | A LUT is a *resource*, not a parameter — and the four things the lookup had to get right |
| [colour-grading/adjustments.md](colour-grading/adjustments.md) | The six later sliders, why each sits where it does, and Tint's own gain |
| [colour-grading/shadows-highlights.md](colour-grading/shadows-highlights.md) | The rewrite: they were a brightness shift, and two of them inverted tone |
| [colour-grading/highlight-reconstruction.md](colour-grading/highlight-reconstruction.md) | An 8-bit file clips per *channel*; recovering it, and why it first shipped invisible |
| [colour-grading/split-tone.md](colour-grading/split-tone.md) | Tone Response's bidirectional split tone |
