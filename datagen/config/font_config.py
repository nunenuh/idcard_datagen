

arial_ttf_path = 'data/fonts/arial.ttf'
ocra_ttf_path = 'data/fonts/ocr_a_ext.ttf'

font = {
    'arial': arial_ttf_path,
    'ocra': ocra_ttf_path
}

def font_path(name):
    return font[name]