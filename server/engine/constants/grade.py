from __future__ import annotations


#
# Peak channel gain for the temperature shift, at grade_temp = +/-1. Red and
# blue move by this much in opposite directions and green is left alone; the
# whole vector is then normalised against the luma weights, so the control
# changes the light's colour and not its level. 0.40 puts the extremes at a
# roughly 3200K/8000K feel without any setting driving a channel to a rail.
_GRADE_TEMP_GAIN = 0.40
#
# Peak channel gain for Tint, the other white-balance axis: green against
# magenta. Deliberately *smaller* than Temperature's -- green carries most of
# the luma weight (0.7152 against red's 0.2126 and blue's 0.0722), so the same
# magnitude on this axis costs more level. Measured on an asymmetric plate,
# 0.40 drifts luma up to 2.3%, against Temperature's own 1.8% at its 0.40; 0.30
# brings the worst case to 1.4%, in the same envelope as Temperature.
_GRADE_TINT_GAIN = 0.30
#
# Where the shadow and highlight ramps meet. Each control acts on one side of
# it only -- so every pixel is in exactly one of them, the two are independent
# by construction rather than by bookkeeping, and both curves leave the knee at
# slope exactly 1, which is what makes the join invisible on a gradient without
# a ramp to fade them in. Not exposed: a knee plus a falloff per end would be
# four more sliders for a section that is meant to stay cheap and quick.
_GRADE_TONE_KNEE = 0.5
#
# How far a tone lift can travel at +/-1, as a share of the headroom it has.
#
# Not 1.0, and the difference matters more than it sounds. The lift is a share
# of the distance to the rail, so at 1.0 a setting of +1 takes a black pixel to
# *pure white* -- which means the whole top of the slider is unusable and the
# useful range is squeezed into its first tenth. Measured on a real photograph
# (mean luma 0.21), Shadows at only +0.5 took the frame's mean from 0.19 to
# 0.53; that is not a shadow lift, it is a different exposure. At 0.35 the same
# +1 moves a black pixel to 0.35 -- a thoroughly lifted, faded-print look, which
# is a fair thing to find at the end of the travel -- and +0.3 gives the natural
# one-stop-ish lift you actually reach for. Same lesson as `_JITTER_MAX`, from
# the other direction: a control whose whole range has to be usable.
#
# Applies to the *expanding* half of each control only -- Shadows negative and
# Highlights positive, which push a tone toward the rail it is already nearest.
# The recovering halves (Shadows positive, Highlights negative) do not use it:
# they are asymptotic rolls whose endpoint is fixed by the curve's own shape,
# and a share-of-headroom cap is exactly what made them non-monotonic. See
# `_tone_roll`.
_GRADE_TONE_MAX = 0.35
#
# Where a channel starts counting as clipped, for highlight reconstruction: the
# soft window over which "this is a real measurement" becomes "this hit the
# ceiling and the true value is somewhere above it".
#
# Soft rather than a single threshold because the window doubles as the blend
# weight, and a hard switch would draw a visible contour around every blown
# region. The top is just short of 1.0 rather than at it because an 8-bit
# ceiling is 255/255 and JPEG ringing puts real clipped pixels a code value or
# two either side of it.
_RECON_LO = 0.94
_RECON_HI = 0.999
#
# How much local evidence a channel needs before its reconstruction is trusted,
# as a share of the neighbourhood that still holds a valid measurement of that
# channel. Below this the estimate fades out rather than being divided into
# existence: in the middle of a region blown wide in every channel there is
# genuinely nothing to recover from, and saying so is better than inventing it.
_RECON_MIN_EVIDENCE = 0.02
#
# Ceiling on a reconstructed value, in display-referred units -- two stops above
# white. The estimate divides by the local chromaticity of whichever channels
# survived, so a highlight lit by something the surviving channel barely sees
# (deep blue in tungsten light) has a small denominator and would otherwise run
# away. Two stops is more headroom than any 8-bit source can justify and still
# bounded.
_RECON_CEIL = 4.0
#
# Knee for reconstruction's own roll -- the curve that brings what it recovered
# back inside the visible range, without which the whole stage is invisible.
#
# High, at 0.80, and that is the point: the recovered data lands just above
# white, so only the top fifth of the range has to give way to make room for it.
# A knee at `_GRADE_TONE_KNEE` (0.5) would work too and would be the wrong
# control -- that is a broad highlight roll, i.e. what `grade_highlights` is
# for, and duplicating it here would mean reconstruction quietly graded the
# picture as well as repairing it.
_RECON_ROLL_KNEE = 0.80
#
# How wide to smooth the roll's gate, as a fraction of the reconstruction
# radius. The gate is the reconstruction's own weight field, so the roll engages
# only where something was actually repaired -- but a per-pixel gate would draw
# an edge around every repaired region, so it is dilated and feathered.
#
# 0.25 rather than the 0.5 first tried, and it is better on every axis measured,
# which is worth recording because the larger value looks like the safer one. The
# gate only has to be wide enough not to contour: swept against the second
# derivative of a repaired ramp, 0.05 starts introducing curvature the source
# does not have (3.2e-03 against the source's own 2.2e-03) while 0.10, 0.25 and
# 0.50 all sit at 1.4e-03, so 0.25 clears it with a 5x margin. Going wider then
# only costs -- the recovered span holds flatter across the radius at 0.25
# (0.0736 / 0.0736 / 0.0735 / 0.0657 against 0.5's 0.0736 / 0.0735 / 0.0650 /
# 0.0654 at 8/16/32/64px) and `pad_for` at the top of the radius range is 933px
# against 1233px.
_RECON_ROLL_GATE_FRAC = 0.25
# Gain on the positive side of Clarity. The negative side is pinned at exactly
# 1.0 and cannot be raised: at gain 1 a setting of -1 removes precisely 100% of
# the local-contrast band, and anything past that does not flatten further, it
# *inverts* -- dark halos on the light side of every edge. Positive has no such
# limit, so it gets the headroom to be worth reaching for.
_GRADE_CLARITY_GAIN = 1.6
#
# Gain on Contrast's pivot-about-middle-grey, at +/-1. Floored so the gain can
# never reach 0 or go negative: at -1 it is 0.1, a tenth of the original
# spread rather than a flattened or inverted picture. Deliberately smaller than
# the film characteristic curve's own 1.1 -- this control has no shoulder to
# catch what it steepens, so a gentler reach keeps the top of the slider from
# clipping immediately.
_GRADE_CONTRAST_GAIN = 0.9

# Local mean-absolute-deviation thresholds, in luma units, separating "smooth"
# from "textured" over a medium radius. Skin and clear sky sit near or below
# _TEX_LO; fabric, foliage and hair sit above _TEX_HI. Fixed constants, not
# per-image statistics, so tiles stay independent.
_TEX_LO = 0.002
_TEX_HI = 0.015

# Luma-step thresholds separating a real transition from fine texture, for the
# edge-softening mask. Calibrated by measurement: fine texture measures a step
# an order of magnitude under a hard border, so the gap between these is what
# lets softening take the snap off a border while leaving fabric and hair
# intact. Fixed constants, not per-image statistics, so tiles stay independent.
_STEP_LO = 0.030
_STEP_HI = 0.110

__all__ = [
    '_GRADE_TEMP_GAIN',
    '_GRADE_TINT_GAIN',
    '_GRADE_TONE_KNEE',
    '_GRADE_TONE_MAX',
    '_RECON_LO',
    '_RECON_HI',
    '_RECON_MIN_EVIDENCE',
    '_RECON_CEIL',
    '_RECON_ROLL_KNEE',
    '_RECON_ROLL_GATE_FRAC',
    '_GRADE_CLARITY_GAIN',
    '_GRADE_CONTRAST_GAIN',
    '_TEX_LO',
    '_TEX_HI',
    '_STEP_LO',
    '_STEP_HI',
]
