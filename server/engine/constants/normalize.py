from __future__ import annotations


#
# Where the metering aims the mid-tones, in *linear* light.
#
# This is `_MID_GREY` (0.46 display-referred) through the sRGB transfer, and it
# lands on 0.179 -- the 18% grey every meter in photography is built around.
# That is a check on the number rather than a coincidence: the film curve pivots
# on `_MID_GREY` and Contrast pivots on `_MID_GREY`, so aiming anywhere else
# would hand the pipeline a frame whose mid-point is not the one every stage
# below it assumes.
#
# Metered as a *log* average rather than a mean. The mean of a frame with a
# bright sky in it is the sky; the log average is dominated by the bulk of the
# tones, which is what "how bright is this photograph" means to a person. It is
# also the classic tone-mapping key, for the same reason.
_NORM_TARGET_LIN = 0.179
#
# Ceiling on the exposure correction, in stops, each way.
#
# Two stops is a lot -- 4x the light -- and the cap exists for the frames where
# the metering is *right* and the correction is still not wanted: a night scene,
# a deliberately low-key portrait, a silhouette. Those genuinely have a low log
# average, and a normalizer with no cap would faithfully lift them to mid-grey
# and destroy the photograph. The cap says "correct an exposure error, do not
# relight the scene".
#
# Not clamped tighter because real exposure errors do reach this far: a frame
# shot at 1/250 when it wanted 1/60 is exactly two stops under, and refusing to
# fix it is the failure the whole control exists to prevent.
_NORM_EV_MAX = 2.0
#
# Minkowski exponent for the white-balance estimate.
#
# p = 1 is grey-world (the mean of each channel, assume the scene averages to
# neutral) and p = infinity is white-patch (the brightest pixel is white). Both
# are famously wrong on their own: grey-world neutralises a sunset, white-patch
# keys the entire frame off one specular hit. p = 6 is the standard "shades of
# grey" compromise and measures better than either on real photographs -- it
# weights bright, well-lit surfaces more than a mean does without letting the
# single brightest pixel decide.
_NORM_WB_P = 6.0
#
# Ceiling on any one white-balance channel multiplier.
#
# A cast strong enough to want more than this is either a real coloured light
# the photographer wanted (a neon sign, a sunset, a stage wash) or a scene with
# no neutral in it at all, and in both cases the estimator is confidently wrong.
# 1.6 covers tungsten-shot-as-daylight, which is about the worst *mistake* that
# occurs in practice, and stops well short of turning a red sunset grey.
_NORM_WB_MAX = 1.6
#
# Chroma diversity below which the white-balance correction is damped out.
#
# The failure grey-world has is a scene that is *legitimately* one colour --
# blue hour, a sunset, a close-up of red fabric. Those are indistinguishable
# from a cast by the channel means alone, so the discriminator has to be
# something else: how much the hues in the frame *vary*. A real cast shifts
# every hue together and leaves the spread intact; a genuinely monochrome scene
# has no spread to begin with.
#
# Measured as the mean absolute deviation of the per-pixel chroma axes. Below
# `_NORM_WB_DIV_LO` the correction is switched off entirely, above
# `_NORM_WB_DIV_HI` it is applied in full, and it ramps between -- so the
# control degrades toward doing nothing on exactly the frames where it would
# otherwise do harm.
_NORM_WB_DIV_LO = 0.010
_NORM_WB_DIV_HI = 0.045
#
# Where the toe starts, display-referred. Low on purpose: it catches what would
# otherwise be crushed against black, and is not a place to grade from.
#
# There is deliberately **no matching shoulder knee, and its absence is a bug
# fix.** The first version had one, fixed at 0.82, and it was the direct cause
# of the highlights being destroyed: at +2 EV every source value above 0.29
# already lands above 0.82, so 70% of the tonal range had to be crammed into 18%
# of the output and 77 8-bit levels of highlight came out as **3.2**. A fixed
# knee cannot work for a variable lift, because the lift decides how much of the
# picture arrives above it.
#
# The highlight roll is the extended Reinhard tone map in linear light now,
# which compresses gradually across the whole top end and needs no knee at all.
# Its only parameter is the frame's own gained maximum, measured rather than
# chosen. Same photograph, same lift: **35.3 levels**. See `docs/normalize.md`.
_NORM_TOE_KNEE = 0.06
#
# The band over which Highlight Priority blends back toward the original file,
# keyed on the **source** pixel's own channel maximum.
#
# Both ends were swept against the transfer's slope rather than picked. The
# obvious choice is a narrow band high up -- "only touch the highlights" -- and
# it is the wrong one, because the blend has to fight the correction: the
# corrected value is far above the original there, so a fast ramp pulls the
# curve down as the input rises and flattens it. Measured on a real photograph
# lifted two stops, a 0.50-1.00 ramp drops the minimum slope to **0.029**, which
# is a worse flat spot than the one the whole feature exists to remove, and
# 0.30-1.00 still reaches 0.197 against the tone map's own 0.289.
#
# 0.15 is wide enough that the weight is already moving before the correction
# has opened up much of a gap, so the two never fight: minimum slope **0.321**,
# *better* than the tone map alone, with mean slope across 0.6-0.9 going 0.404
# to 0.634 and 86% of the mid-tone lift still kept. Going wider (0.0-1.0) keeps
# climbing on slope but gives away the correction -- 72% kept -- which is the
# half of the trade the user asked to protect.
_NORM_HP_LO = 0.15
_NORM_HP_HI = 1.0
#
# Ceiling on the toe strength, as a `_tone_roll` amount. The highlight roll has
# no equivalent, and that asymmetry is the "a bit of HDR, not a log profile"
# decision -- now pointing the opposite way from where it started.
#
# It first read the other way round: the shoulder was uncapped and the toe held
# down, on the argument that a lifted black point is the recognisable half of the
# log look. The argument was fine and the conclusion was wrong, because it
# assumed the shoulder could only ever touch the top of the range. A knee makes
# that true only for a *small* lift; at +2 EV most of the picture arrived above
# the knee, so the uncapped shoulder crushed the whole highlight region and the
# user reported exactly that. The Reinhard roll needs no cap for a different
# reason: it is gradual by construction, so there is no strength to run away.
#
# The toe keeps its ceiling. A lifted black point genuinely is visible as
# flatness, and nothing is being rescued from a hard rail at the bottom -- a
# positive gain cannot push a value below 0, so shadows are compressed rather
# than lost. 0.6 recovers most of the separation that darkening costs while
# leaving the blacks reading as black. The user was explicit that highlights
# matter more than shadows here, so this end is the one that stays conservative.
_NORM_TONE_MAX = 0.6
#
# Where a channel stops counting as a trustworthy sample, for the metering only.
#
# Both estimators exclude pixels outside this window. At the top a clipped
# channel is not a measurement of anything -- it is the ceiling, and averaging
# it in drags the white balance toward the clipped hue and the exposure toward
# whatever the ceiling happens to be. At the bottom, near-black pixels are
# mostly sensor noise and their ratios are meaningless, so a frame with large
# black areas would otherwise have its white balance set by its noise floor.
#
# Deliberately a little tighter at the top than `_RECON_LO` (0.94): that window
# is a blend weight and wants to be soft, this one is a validity test and wants
# to be safe.
_NORM_VALID_LO = 0.02
_NORM_VALID_HI = 0.92
#
# The toe has no constant of its own: it is derived from the exposure
# correction directly, as a share of `_NORM_EV_MAX`. Darkening by `ev` stops
# compresses everything below the knee by exactly `2**ev`, so how much shadow
# separation the correction cost is a property of the correction and not
# something to measure a second time.
#
# It was first sized from the frame's own black level, and that was the wrong
# question: a well-exposed photograph with genuine deep shadows measured a toe
# of 0.216 and had its blacks lifted to fix a problem it did not have.
#
# Largest number of samples the metering draws from the frame.
#
# The statistics here are percentiles and log averages over millions of pixels,
# and both converge long before the pixels run out -- a uniform stride to ~2M
# samples measures the same numbers as the full frame to well under an 8-bit
# level, at a fraction of the cost. Strided rather than random so the metering
# is deterministic: the same photograph must produce the same correction every
# time it is opened, or a re-upload would silently regrade it.
_NORM_MAX_SAMPLES = 2_000_000

__all__ = [
    '_NORM_TARGET_LIN',
    '_NORM_EV_MAX',
    '_NORM_HP_HI',
    '_NORM_HP_LO',
    '_NORM_WB_P',
    '_NORM_WB_MAX',
    '_NORM_WB_DIV_LO',
    '_NORM_WB_DIV_HI',
    '_NORM_TOE_KNEE',
    '_NORM_TONE_MAX',
    '_NORM_VALID_LO',
    '_NORM_VALID_HI',
    '_NORM_MAX_SAMPLES',
]
