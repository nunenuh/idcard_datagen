
from .transforms import *


light_effect_fn = ComposeRandomChoice([
    RandomGamma(gamma_range=(0.5, 2.5), p=0.5),
    RandomContrast(level_range=(0, 20), p=0.5),
    RandomBrightness(level_range=(20, 50), p=0.5),
], k=2, debug=False)

darklight_effect_fn = ComposeRandomChoice([
    Darken(),
    Lighten()
], k=1, debug=False)

color_effect_fn = ComposeRandomChoice([
    RandomHueShifting(shift_range=(1, 100), p=0.5),
    RandomChannelShuffle(p=0.5),
], k=1, debug=False)

morph_effect_fn = ComposeRandomChoice([
    RandomMorphDilation(p=0.5),
    RandomMorphOpening(p=0.5),
    RandomMorphClosing(p=0.5)
], k=1, debug=False)


####################### foreground effect ######################

foreground_simple_effect_fn = ComposeRandomChoice([
    darklight_effect_fn,
    color_effect_fn,
    RandomXenoxPhotocopy(p=0.5, noise_p=0.3, thresh_p=0.5),
    RandomLoRes(factor_range=(0.3, 0.5), p=0.5),
    RandomSharpen(p=0.5),
    RandomNoise(amount_range=(0.05, 0.06), p=0.5),
    RandomGaussionBlur(sigma_range=(1.0, 5.0), p=0.5),
], k=1, debug=False)

foreground_medium_effect_fn = ComposeRandomChoice([
    light_effect_fn,
    darklight_effect_fn,
    color_effect_fn,
    RandomXenoxPhotocopy(p=1, noise_p=0.3, thresh_p=0.8),
    RandomLoRes(factor_range=(0.3, 0.5), p=1),
    RandomSharpen(p=1),
    RandomNoise(amount_range=(0.05, 0.06)),
    RandomGaussionBlur(sigma_range=(1.0, 5.0), p=1),
    morph_effect_fn,
], k=3, debug=False)

foreground_complex_effect_fn = ComposeRandomChoice([
    light_effect_fn,
    darklight_effect_fn,
    color_effect_fn,
    RandomXenoxPhotocopy(p=1, noise_p=0.5, thresh_p=0.8),
    RandomLoRes(factor_range=(0.3, 0.5), p=1),
    RandomSharpen(p=1),
    RandomNoise(amount_range=(0.05, 0.06), p=1),
    RandomGaussionBlur(sigma_range=(1.0, 5.0), p=1),
    morph_effect_fn,
], k=5, debug=False)


foreground_effect_dict = {
    'simple': foreground_simple_effect_fn, 
    "medium": foreground_medium_effect_fn,
    "complex": foreground_complex_effect_fn,
}

######################## eof foreground effect #######################


####################### background effect ######################

background_simple_effect_fn = ComposeRandomChoice([
    darklight_effect_fn,
    color_effect_fn,
    RandomLoRes(factor_range=(0.3, 0.5), p=0.5),
    RandomSharpen(p=0.5),
    RandomNoise(amount_range=(0.05, 0.06), p=0.5),
    RandomGaussionBlur(sigma_range=(1.0, 5.0), p=0.5),
], k=1, debug=False)

background_medium_effect_fn = ComposeRandomChoice([
    light_effect_fn,
    darklight_effect_fn,
    color_effect_fn,
    RandomLoRes(factor_range=(0.3, 0.5), p=1),
    RandomSharpen(p=1),
    RandomNoise(amount_range=(0.05, 0.06)),
    RandomGaussionBlur(sigma_range=(1.0, 5.0), p=1),
    morph_effect_fn,
], k=3, debug=False)

background_complex_effect_fn = ComposeRandomChoice([
    light_effect_fn,
    darklight_effect_fn,
    color_effect_fn,
    RandomLoRes(factor_range=(0.3, 0.5), p=1),
    RandomSharpen(p=1),
    RandomNoise(amount_range=(0.05, 0.06), p=1),
    RandomGaussionBlur(sigma_range=(1.0, 5.0), p=1),
    morph_effect_fn,
], k=5, debug=False)


background_effect_dict = {
    'simple': background_simple_effect_fn, 
    "medium": background_medium_effect_fn,
    "complex": background_complex_effect_fn,
}

######################## eof foreground effect #######################

####################### composite_bfx base effect ######################

composite_bfx_simple_effect_fn = ComposeRandomChoice([
    darklight_effect_fn,
    color_effect_fn,
    RandomLoRes(factor_range=(0.3, 0.5), p=0.5),
    RandomSharpen(p=0.5),
    RandomNoise(amount_range=(0.05, 0.06), p=0.5),
    RandomGaussionBlur(sigma_range=(1.0, 5.0), p=0.5),
], k=1, debug=False)

composite_bfx_medium_effect_fn = ComposeRandomChoice([
    light_effect_fn,
    darklight_effect_fn,
    color_effect_fn,
    RandomLoRes(factor_range=(0.3, 0.5), p=1),
    RandomSharpen(p=1),
    RandomNoise(amount_range=(0.05, 0.06)),
    RandomGaussionBlur(sigma_range=(1.0, 5.0), p=1),
    morph_effect_fn,
], k=3, debug=False)

composite_bfx_complex_effect_fn = ComposeRandomChoice([
    light_effect_fn,
    darklight_effect_fn,
    color_effect_fn,
    RandomLoRes(factor_range=(0.3, 0.5), p=1),
    RandomSharpen(p=1),
    RandomNoise(amount_range=(0.05, 0.06), p=1),
    RandomGaussionBlur(sigma_range=(1.0, 5.0), p=1),
    morph_effect_fn,
], k=5, debug=False)


composite_bfx_effect_dict = {
    'simple': composite_bfx_simple_effect_fn, 
    "medium": composite_bfx_medium_effect_fn,
    "complex": composite_bfx_complex_effect_fn,
}

######################## eof composite_bfx base effect #######################


####################### composite_afx effect ######################

composite_afx_simple_effect_fn = ComposeRandomChoice([
    RandomAddSunFlares(p=0.5),
    RandomShadow(p=0.5),
    RandomNoise(amount_range=(0.05, 0.07), p=0.5),
], k=1, debug=False)

composite_afx_medium_effect_fn = ComposeRandomChoice([
    RandomAddSunFlares(p=0.5),
    RandomAddShadow(p=0.5),
    RandomShadow(p=0.5),
    RandomNoise(p=0.5),
    RandomAddSnow(p=0.5),
    RandomAddRain(p=0.5),
    RandomAddSpeed(p=0.5),
    RandomAddFog(p=0.5),
    RandomAddGravel(p=0.5),
], k=3, debug=False)

composite_afx_complex_effect_fn = ComposeRandomChoice([
    RandomAddSunFlares(p=0.5),
    RandomAddShadow(p=0.5),
    RandomShadow(p=0.5),
    RandomNoise(p=0.5),
    RandomAddSnow(p=0.5),
    RandomAddRain(p=0.5),
    RandomAddSpeed(p=0.5),
    RandomAddFog(p=0.5),
    RandomAddGravel(p=0.5),
], k=5, debug=False)


composite_afx_effect_dict = {
    'simple': composite_afx_simple_effect_fn, 
    "medium": composite_afx_medium_effect_fn,
    "complex": composite_afx_complex_effect_fn,
}

######################## eof composite_afx base effect #######################

