import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import scipy.signal as sig

def Filter_CharakteristischeStruktur(surface):
    surface_filtered = surface.filter(filter_type='bandpass', cutoff=210, cutoff2=840)
    return surface_filtered

def Filter_Nanofocus(surface):
    surface_filtered = surface.filter(filter_type='bandpass', cutoff=4.1, cutoff2=4210)
    return surface_filtered
def Filter_Rauheit_Nanofocus(surface):
    surface_filtered = surface.filter(filter_type='bandpass', cutoff=2.5, cutoff2=150)
    return surface_filtered
def Filter_Rauheit_Keyence(surface):
    surface_filtered = surface.filter(filter_type='bandpass', cutoff=8, cutoff2=350)
    return surface_filtered



def gaussian_profile_filter(z, dx_um, cutoff_um, mode="lowpass"):
    """
    1D-Gaussian-Profilfilter.

    z        : 1D-Profilwerte
    dx_um    : Abtastabstand entlang des Profils in µm
    cutoff_um: gewünschte Filterlänge in µm
    mode     : 'lowpass' oder 'highpass'
    """
    z = np.asarray(z, dtype=float)

    # grobe praxisnahe Zuordnung: sigma in Pixeln
    sigma_px = cutoff_um / dx_um / 2.0

    low = gaussian_filter1d(z, sigma=sigma_px, mode="reflect")

    if mode == "lowpass":
        return low
    elif mode == "highpass":
        return z - low
    else:
        raise ValueError("mode muss 'lowpass' oder 'highpass' sein.")

def butter_profile_filter(z, dx_um, cutoff_um, mode="lowpass", order=4):
    """
    1D-Butterworth-Profilfilter mit zero-phase (filtfilt).

    z         : 1D-Profilwerte
    dx_um     : Sampling entlang des Profils in µm
    cutoff_um : Cutoff-Wellenlänge in µm
    mode      : 'lowpass' oder 'highpass'
    order     : Filterordnung
    """
    z = np.asarray(z, dtype=float)

    fs = 1.0 / dx_um                 # Samples pro µm
    f_c = 1.0 / cutoff_um            # Zyklen pro µm

    b, a = sig.butter(order, f_c, btype="low" if mode=="lowpass" else "high", fs=fs)
    zf = sig.filtfilt(b, a, z)

    return zf

def butter_profile_bandpass(z, dx_um, low_wavelength_um, high_wavelength_um, order=4):
    """
    Bandpass für Profil:
    lässt Wellenlängen zwischen low_wavelength_um und high_wavelength_um ungefähr durch.
    Beispiel: 2.5 bis 150 µm
    """
    z = np.asarray(z, dtype=float)

    fs = 1.0 / dx_um

    # Wellenlänge -> Frequenz
    f_low  = 1.0 / high_wavelength_um   # längere Wellen -> kleinere Frequenz
    f_high = 1.0 / low_wavelength_um    # kürzere Wellen -> größere Frequenz

    b, a = sig.butter(order, [f_low, f_high], btype="bandpass", fs=fs)
    zf = sig.filtfilt(b, a, z)

    return zf