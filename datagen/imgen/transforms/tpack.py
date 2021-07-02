
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

simple_basic_effect_fn = ComposeRandomChoice([
    darklight_effect_fn,
    color_effect_fn,
    RandomLoRes(factor_range=(0.3, 0.5), p=0.5),
    RandomSharpen(p=0.5),
    RandomNoise(amount_range=(0.05, 0.06), p=0.5),
    RandomGaussionBlur(sigma_range=(1.0, 5.0), p=0.5),
], k=1, debug=False)


medium_basic_effect_fn = ComposeRandomChoice([
    light_effect_fn,
    darklight_effect_fn,
    color_effect_fn,
    RandomLoRes(factor_range=(0.3, 0.5), p=1),
    RandomSharpen(p=1),
    RandomNoise(amount_range=(0.05, 0.06)),
    RandomGaussionBlur(sigma_range=(1.0, 5.0), p=1),
    morph_effect_fn,
], k=3, debug=False)


complex_basic_effect_fn = ComposeRandomChoice([
    light_effect_fn,
    darklight_effect_fn,
    color_effect_fn,
    RandomLoRes(factor_range=(0.3, 0.5), p=1),
    RandomSharpen(p=1),
    RandomNoise(amount_range=(0.05, 0.06), p=1),
    RandomGaussionBlur(sigma_range=(1.0, 5.0), p=1),
    morph_effect_fn,
], k=5, debug=False)




simple_advance_effect_fn = ComposeRandomChoice([
    RandomAddSunFlares(p=0.5),
    RandomShadow(p=0.5),
    RandomNoise(amount_range=(0.05, 0.07), p=0.5),
], k=1, debug=False)

medium_advance_effect_fn = ComposeRandomChoice([
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

complex_advance_effect_fn = ComposeRandomChoice([
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



basic_effect_dict = {
    'simple': simple_basic_effect_fn, 
    "medium": medium_basic_effect_fn,
    "complex": complex_basic_effect_fn,
}

foreground_effect_dict = {
    'simple': simple_basic_effect_fn, 
    "medium": medium_basic_effect_fn,
    "complex": complex_basic_effect_fn,
}

background_effect_dict = {
    'simple': simple_basic_effect_fn, 
    "medium": medium_basic_effect_fn,
    "complex": complex_basic_effect_fn,
}

composite_base_effect_dict = {
    'simple': simple_basic_effect_fn, 
    "medium": medium_basic_effect_fn,
    "complex": complex_basic_effect_fn,
}

composite_adv_effect_dict = {
    "simple": simple_advance_effect_fn,
    "medium": medium_advance_effect_fn,
    "complex": complex_advance_effect_fn,
}
