import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from surfalize import Surface


# ============================================================
# Hilfsfunktionen für 2D-FFT-Auswertung von Oberflächen
# Damit können Keyence- und NanoFocus-Messungen im selben
# Wellenlängenband gefiltert, visualisiert und verglichen werden
# ============================================================


def nan_fill(z: np.ndarray) -> np.ndarray:
    """
    Ersetzt NaN-Werte in einem Höhenfeld.

    Was kann man damit machen?
    - Nicht gemessene oder ungültige Punkte robust auffüllen
    - Arrays für FFT, Filterung und Visualisierung vorbereiten
    - Abstürze durch NaN-Werte vermeiden

    Vorgehen:
    - Falls NaNs vorhanden sind, werden sie durch den Median ersetzt
    - Falls selbst der Median nicht sinnvoll bestimmbar ist, wird 0 gesetzt

    Rückgabe:
    - Array ohne NaN-Werte
    """
    z = z.astype(float, copy=False)
    if np.isnan(z).any():
        med = np.nanmedian(z)
        if not np.isfinite(med):
            med = 0.0
        z = np.nan_to_num(z, nan=med)
    return z


def robust_norm_signed(z: np.ndarray, clip_sigma=3.0) -> np.ndarray:
    """
    Normiert ein Signal auf seine Standardabweichung und clippt Extremwerte.

    Was kann man damit machen?
    - Bandpass-Signale vergleichbar darstellen
    - Positive und negative Auslenkungen symmetrisch visualisieren
    - Ausreißer für die Darstellung begrenzen

    Geeignet für:
    - Signale mit positiven und negativen Werten
    - Bandpass-gefilterte Oberflächenanteile

    Rückgabe:
    - Normiertes und geclipptes Array
    """
    z = nan_fill(z)
    s = np.std(z) + 1e-12
    zn = z / s
    return np.clip(zn, -clip_sigma, clip_sigma)


def robust_norm_pos(z: np.ndarray, hi_q=99.5) -> np.ndarray:
    """
    Normiert ein nur positives Signal auf den Bereich 0..1.

    Was kann man damit machen?
    - Hüllkurven oder Amplitudenbilder robust darstellen
    - Sehr große Spitzen über Perzentile begrenzen
    - Kontraste zwischen relevanten Bereichen sichtbar machen

    Geeignet für:
    - Envelope-/Amplitudenbilder
    - Immer positive Größen

    Rückgabe:
    - Array im Bereich 0..1
    """
    z = nan_fill(z)
    hi = np.percentile(z, hi_q) + 1e-12
    return np.clip(z / hi, 0.0, 1.0)


def resample_to_step(surf: Surface, target_step_um: float) -> Surface:
    """
    Resampelt eine Oberfläche bilinear auf eine isotrope Schrittweite.

    Was kann man damit machen?
    - Unterschiedliche x- und y-Auflösungen angleichen
    - Eine Oberfläche auf ein quadratisches Raster bringen
    - Saubere 2D-FFT-Auswertungen vorbereiten

    Parameter:
    - surf:
      Eingangsoberfläche
    - target_step_um:
      Gewünschte Schrittweite in µm für x und y

    Rückgabe:
    - Neue Surface mit identischer Schrittweite in x und y
    """
    zx = surf.step_x / target_step_um
    zy = surf.step_y / target_step_um
    z = nan_fill(surf.data)
    z_rs = ndimage.zoom(z, zoom=(zy, zx), order=1)
    return Surface(z_rs, target_step_um, target_step_um)


def isotropize(surf: Surface) -> Surface:
    """
    Macht eine Oberfläche isotrop, indem auf die kleinere vorhandene Schrittweite resampelt wird.

    Was kann man damit machen?
    - Oberflächen für richtungsunabhängige FFT-Analysen vorbereiten
    - Verzerrungen durch unterschiedliche Pixelgrößen verhindern
    - Ein konsistentes Raster erzeugen

    Rückgabe:
    - Isotrope Surface
    """
    step = float(min(surf.step_x, surf.step_y))
    return resample_to_step(surf, step)


def symmetric_crop_to(surf: Surface, target_w_um: float, target_h_um: float) -> Surface:
    """
    Cropt eine Oberfläche symmetrisch auf eine gewünschte Größe in µm.

    Was kann man damit machen?
    - Zwei Messsysteme auf denselben physikalischen Ausschnitt bringen
    - Zentral einen gemeinsamen Vergleichsbereich definieren
    - Randbereiche entfernen

    Rückgabe:
    - Zentral gecroppte Surface
    """
    if surf.width_um <= target_w_um or surf.height_um <= target_h_um:
        raise ValueError("Surface zu klein für symmetrischen Crop auf Zielgröße")
    dx = (surf.width_um - target_w_um) / 2.0
    dy = (surf.height_um - target_h_um) / 2.0
    return surf.crop((dx, dx + target_w_um, dy, dy + target_h_um), in_units=True)


# ============================================================
# 2D-FFT-Bandpass und Hüllkurve
# Damit können definierte Wellenlängenbereiche aus der
# Oberfläche isoliert und anschließend als Signal bzw.
# Amplitudenbild ausgewertet werden
# ============================================================


def bandpass_2d_fft_isotropic(
    surf: Surface,
    wl_min_um=200.0,
    wl_max_um=600.0,
    tukey_alpha=0.1,
    remove_dc=True,
) -> Surface:
    """
    Führt einen isotropen 2D-Bandpass im Wellenlängenraum aus.

    Was kann man damit machen?
    - Nur Strukturen in einem gewünschten Wellenlängenbereich extrahieren
    - Z. B. Welligkeiten zwischen 200 und 600 µm isolieren
    - Oberflächen verschiedener Messsysteme direkt im selben Frequenzband vergleichen

    Ablauf:
    1. Oberfläche isotrop resampeln
    2. Formkorrektur durchführen
    3. Tukey-Fenster anwenden
    4. 2D-FFT berechnen
    5. Ringförmigen Bandpass im Frequenzraum anwenden
    6. Rücktransformation in den Ortsraum

    Parameter:
    - wl_min_um:
      Untere Wellenlängengrenze
    - wl_max_um:
      Obere Wellenlängengrenze
    - tukey_alpha:
      Fensterparameter für das Tukey-Fenster
    - remove_dc:
      Entfernt den Gleichanteil

    Rückgabe:
    - Gefilterte Surface im Ortsraum
    """
    # Oberfläche auf isotropes Raster bringen
    s = isotropize(surf)

    # Formkorrektur und Trendabzug
    s = s.level().detrend_polynomial(2)
    z = nan_fill(s.data)

    ny, nx = z.shape

    # 2D-Tukey-Fenster gegen Randartefakte in der FFT
    wy = signal.windows.tukey(ny, alpha=tukey_alpha)
    wx = signal.windows.tukey(nx, alpha=tukey_alpha)
    w2 = np.outer(wy, wx)
    zW = z * w2

    # Fouriertransformierte berechnen
    F = np.fft.fftshift(np.fft.fft2(zW))

    # Frequenzachsen aufbauen
    step = float(s.step_x)
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=step))
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=step))
    FX, FY = np.meshgrid(fx, fy)

    # Radialfrequenz und zugehörige Wellenlänge
    FR = np.sqrt(FX**2 + FY**2)
    WL = np.where(FR > 0, 1.0 / FR, np.inf)

    # Ringförmige Maske im gewünschten Wellenlängenband
    mask = (WL >= wl_min_um) & (WL <= wl_max_um)
    if remove_dc:
        mask &= (FR > 0)

    # Gefiltertes Spektrum zurücktransformieren
    Fb = np.fft.ifftshift(F * mask)
    zb = np.fft.ifft2(Fb).real

    return Surface(zb, s.step_x, s.step_y)


def envelope_from_bandpass(band: Surface, smooth_sigma_px=2.0) -> Surface:
    """
    Berechnet eine geglättete Hüllkurve aus einem bandpass-gefilterten Signal.

    Was kann man damit machen?
    - Die lokale Stärke eines Bandpass-Signals sichtbar machen
    - Amplitudenverteilungen von Welligkeiten oder Texturanteilen darstellen
    - Bereiche mit hoher und niedriger Signalenergie vergleichen

    Vorgehen:
    - Quadrat des Signals bilden
    - Gauß-Glättung anwenden
    - Quadratwurzel ziehen
    - Ergebnis robust auf 0..1 normieren

    Rückgabe:
    - Hüllkurven-Surface im Bereich 0..1
    """
    z = nan_fill(band.data)
    amp = np.sqrt(ndimage.gaussian_filter(z * z, sigma=float(smooth_sigma_px)))
    amp01 = robust_norm_pos(amp, hi_q=99.5)
    return Surface(amp01, band.step_x, band.step_y)


def dominant_orientation_deg(surf: Surface, wl_min_um=200.0, wl_max_um=600.0) -> float:
    """
    Bestimmt die dominante Richtung im ausgewählten Wellenlängenband.

    Was kann man damit machen?
    - Die vorherrschende Frequenzrichtung einer Oberfläche bestimmen
    - Rückschlüsse auf Textur- oder Bearbeitungsrichtung ziehen
    - Messsysteme hinsichtlich ihrer erfassten Richtungsstruktur vergleichen

    Hinweis:
    - Die dominierende Texturrichtung liegt grob senkrecht zur dominierenden Frequenzrichtung

    Rückgabe:
    - Winkel in Grad
    """
    s = isotropize(surf.level().detrend_polynomial(2))
    z = nan_fill(s.data)
    ny, nx = z.shape

    # Leistungsspektrum berechnen
    F = np.fft.fftshift(np.fft.fft2(z))
    P = np.abs(F) ** 2

    # Frequenzgitter
    step = float(s.step_x)
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=step))
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=step))
    FX, FY = np.meshgrid(fx, fy)

    FR = np.sqrt(FX**2 + FY**2)
    WL = np.where(FR > 0, 1.0 / FR, np.inf)

    # Nur gewünschtes Band betrachten
    band = (WL >= wl_min_um) & (WL <= wl_max_um) & (FR > 0)
    w = P * band

    # Winkelgewichtete Mittelung im Frequenzraum
    ang = np.arctan2(FY, FX)
    wsum = w.sum() + 1e-12
    c = (w * np.cos(ang)).sum() / wsum
    sng = (w * np.sin(ang)).sum() / wsum

    return float(np.degrees(np.arctan2(sng, c)))


# ============================================================
# Vergleichsplot für Keyence und NanoFocus
# Damit können beide Messsysteme nach identischer Vorverarbeitung
# direkt als Bandpass-Bild und Envelope miteinander verglichen werden
# ============================================================


def plot_keyence_vs_nanofocus(
    key: Surface,
    nano: Surface,
    wl_min=200.0,
    wl_max=600.0,
    clip_sigma=3.0,
    env_sigma_px=2.0,
    title="2D-FFT Bandpass (200–600 µm) + Envelope",
):
    """
    Erzeugt einen 2x2-Vergleichsplot für Keyence und NanoFocus.

    Was kann man damit machen?
    - Beide Messsysteme auf demselben physikalischen Fenster vergleichen
    - Bandpass-gefilterte Strukturen nebeneinander darstellen
    - Zusätzlich die lokale Amplitude über die Hüllkurve vergleichen
    - Die dominante Richtungsinformation ausgeben

    Darstellungen:
    - Oben links: Keyence Bandpass
    - Oben rechts: Keyence Envelope
    - Unten links: NanoFocus Bandpass
    - Unten rechts: NanoFocus Envelope
    """
    # Keyence zuerst auf dieselbe physikalische Fenstergröße wie NanoFocus bringen
    key_sym = symmetric_crop_to(key, nano.width_um, nano.height_um)

    # Bandpass für beide Messsysteme berechnen
    key_band = bandpass_2d_fft_isotropic(key_sym, wl_min_um=wl_min, wl_max_um=wl_max)
    nano_band = bandpass_2d_fft_isotropic(nano, wl_min_um=wl_min, wl_max_um=wl_max)

    # Hüllkurven aus den Bandpass-Signalen berechnen
    key_env = envelope_from_bandpass(key_band, smooth_sigma_px=env_sigma_px)
    nano_env = envelope_from_bandpass(nano_band, smooth_sigma_px=env_sigma_px)

    # Bandbilder für die Anzeige standardisieren
    key_band_img = robust_norm_signed(key_band.data, clip_sigma=clip_sigma)
    nano_band_img = robust_norm_signed(nano_band.data, clip_sigma=clip_sigma)

    # Gemeinsame Darstellungsskala für die Bandbilder
    vmin, vmax = -clip_sigma, clip_sigma

    # Envelope ist bereits auf 0..1 normiert
    key_env_img = key_env.data
    nano_env_img = nano_env.data

    # Dominante Richtungen bestimmen
    ang_key = dominant_orientation_deg(key_sym, wl_min_um=wl_min, wl_max_um=wl_max)
    ang_nano = dominant_orientation_deg(nano, wl_min_um=wl_min, wl_max_um=wl_max)

    print(f"Keyence: dominante Frequenzrichtung im Band = {ang_key:.1f}°  (Textur grob ~ {ang_key+90:.1f}°)")
    print(f"NanoFocus: dominante Frequenzrichtung im Band = {ang_nano:.1f}° (Textur grob ~ {ang_nano+90:.1f}°)")

    # 2x2-Darstellung erzeugen
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    # Keyence Bandpass
    ax[0, 0].imshow(key_band_img, aspect="auto", vmin=vmin, vmax=vmax)
    ax[0, 0].set_title("Keyence: Bandpass (z-norm, ±3σ)")
    ax[0, 0].set_xlabel("x [px]")
    ax[0, 0].set_ylabel("y [px]")

    # Keyence Envelope
    ax[0, 1].imshow(key_env_img, aspect="auto", vmin=0, vmax=1)
    ax[0, 1].set_title("Keyence: Envelope/Amplitude (0..1)")
    ax[0, 1].set_xlabel("x [px]")
    ax[0, 1].set_ylabel("y [px]")

    # NanoFocus Bandpass
    ax[1, 0].imshow(nano_band_img, aspect="auto", vmin=vmin, vmax=vmax)
    ax[1, 0].set_title("NanoFocus: Bandpass (z-norm, ±3σ)")
    ax[1, 0].set_xlabel("x [px]")
    ax[1, 0].set_ylabel("y [px]")

    # NanoFocus Envelope
    ax[1, 1].imshow(nano_env_img, aspect="auto", vmin=0, vmax=1)
    ax[1, 1].set_title("NanoFocus: Envelope/Amplitude (0..1)")
    ax[1, 1].set_xlabel("x [px]")
    ax[1, 1].set_ylabel("y [px]")

    plt.show()


# ============================================================
# Hauptfunktion zum Laden der Dateien und Starten des Vergleichs
# ============================================================


def main():
    """
    Lädt eine Keyence- und eine NanoFocus-Datei und startet den FFT-Vergleich.

    Was kann man damit machen?
    - Einen direkten visuellen Vergleich zweier konkreter Messungen durchführen
    - Den Wellenlängenbereich gezielt wählen
    - Bandpass und Envelope sofort nebeneinander ausgeben
    """
    key_path = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP00/1x7_Reflektion/Ergebnisse/WSP00_L1_S1.sdf"
    nano_path = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben/WSP00_L1_S1.nms"

    # Oberflächen laden
    key = Surface.load(key_path)
    nano = Surface.load(nano_path)

    # Vergleichsplot erzeugen
    plot_keyence_vs_nanofocus(
        key=key,
        nano=nano,
        wl_min=200.0,
        wl_max=600.0,
        clip_sigma=3.0,
        env_sigma_px=2.0,
        title="Keyence vs NanoFocus – 2D-FFT Bandpass (200–600 µm) + Envelope",
    )


if __name__ == "__main__":
    main()