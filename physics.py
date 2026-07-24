"""双缝干涉公式及单位换算。"""


def _positive(name, value):
    value = float(value)
    if value <= 0:
        raise ValueError("{}必须大于 0".format(name))
    return value


def theoretical_spacing_mm(wavelength_nm, screen_distance_m, slit_spacing_mm):
    """由波长、屏距和缝距计算理论条纹间距（mm）。"""
    wavelength_nm = _positive("波长", wavelength_nm)
    screen_distance_m = _positive("屏距", screen_distance_m)
    slit_spacing_mm = _positive("双缝间距", slit_spacing_mm)
    return wavelength_nm * 1e-9 * screen_distance_m / (slit_spacing_mm * 1e-3) * 1e3


def wavelength_nm(spacing_mm, screen_distance_m, slit_spacing_mm):
    """由条纹间距、屏距和缝距反算波长（nm）。"""
    spacing_mm = _positive("条纹间距", spacing_mm)
    screen_distance_m = _positive("屏距", screen_distance_m)
    slit_spacing_mm = _positive("双缝间距", slit_spacing_mm)
    return spacing_mm * 1e-3 * slit_spacing_mm * 1e-3 / screen_distance_m * 1e9


def slit_spacing_mm(spacing_mm, wavelength_nm_value, screen_distance_m):
    """由条纹间距、波长和屏距反算双缝间距（mm）。"""
    spacing_mm = _positive("条纹间距", spacing_mm)
    wavelength_nm_value = _positive("波长", wavelength_nm_value)
    screen_distance_m = _positive("屏距", screen_distance_m)
    return wavelength_nm_value * 1e-9 * screen_distance_m / (spacing_mm * 1e-3) * 1e3


def screen_distance_m(spacing_mm, wavelength_nm_value, slit_spacing_mm_value):
    """由条纹间距、波长和缝距反算屏距（m）。"""
    spacing_mm = _positive("条纹间距", spacing_mm)
    wavelength_nm_value = _positive("波长", wavelength_nm_value)
    slit_spacing_mm_value = _positive("双缝间距", slit_spacing_mm_value)
    return spacing_mm * 1e-3 * slit_spacing_mm_value * 1e-3 / (wavelength_nm_value * 1e-9)


def compare_spacing(measured_mm, theoretical_mm):
    """比较实验值和理论值。"""
    measured_mm = _positive("实验条纹间距", measured_mm)
    theoretical_mm = _positive("理论条纹间距", theoretical_mm)
    absolute_error = measured_mm - theoretical_mm
    return {
        'absolute_error_mm': absolute_error,
        'relative_error_percent': abs(absolute_error) / theoretical_mm * 100.0,
    }
