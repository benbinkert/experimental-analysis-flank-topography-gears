import time
import numpy as np
from scipy import ndimage, signal
from surfalize import Surface


# ============================================================
# Matching über bandpassgefilterte Hüllkurven
#
# Wichtiger Hinweis:
# Dieses Verfahren funktioniert für den direkten Vergleich
# NanoFocus ↔ Keyence in der aktuellen Form nicht zuverlässig.
#
# Warum?
# - Das Matching basiert hier auf charakteristischen periodischen
#   bzw. texturartigen Strukturen im 2D-Bild
# - Beim Keyence sind diese charakteristischen Strukturen in vielen
#   Fällen nicht ausreichend klar oder nicht in derselben Form vorhanden
# - Dadurch findet das Verfahren zwar manchmal ein rechnerisches Maximum,
#   aber oft kein zufriedenstellendes, physikalisch plausibles Ergebnis
#
# Was kann man damit machen?
# - Flächen mit ähnlicher Struktur und ähnlichem Texturinhalt matchen
# - Rotationen und Spiegelungen testen
# - Einen passenden Ausschnitt aus einer größeren Fläche bestimmen
#
# Was eher nicht gut funktioniert?
# - NanoFocus gegen Keyence, wenn die prägenden Strukturen im Keyence
#   nicht deutlich genug vorhanden sind
# ============================================================


# ============================================================
# Basics
# ============================================================

def nan_fill(z: np.ndarray) -> np.ndarray:
    """
    Ersetzt NaN-Werte durch den Median.

    Was kann man damit machen?
    - FFT und Korrelation robust auf Daten mit fehlenden Werten anwenden
    - Verhindern, dass NaNs spätere Rechenschritte zerstören
    """
    z = z.astype(float, copy=False)
    if np.isnan(z).any():
        med = np.nanmedian(z)
        if not np.isfinite(med):
            med = 0.0
        z = np.nan_to_num(z, nan=med)
    return z


def zscore(z: np.ndarray) -> np.ndarray:
    """
    Standardisiert ein Array auf Mittelwert 0 und Standardabweichung 1.

    Was kann man damit machen?
    - Bilder vor der Korrelation vergleichbar normieren
    - Amplitudenunterschiede zwischen Datensätzen reduzieren
    """
    z = nan_fill(z)
    m = float(np.mean(z))
    s = float(np.std(z)) + 1e-12
    return (z - m) / s


def resample_to_step_xy(surf: Surface, target_step_x: float, target_step_y: float) -> Surface:
    """
    Resampelt eine Surface auf neue Schrittweiten in x- und y-Richtung.

    Was kann man damit machen?
    - Zwei Datensätze auf eine gemeinsame Rasterauflösung bringen
    - Für spätere FFT- oder Matching-Schritte ein einheitliches Raster erzeugen
    """
    zx = surf.step_x / target_step_x
    zy = surf.step_y / target_step_y

    z = nan_fill(surf.data)

    # zoom erwartet Reihenfolge (y, x)
    z_rs = ndimage.zoom(z, zoom=(zy, zx), order=1)

    return Surface(z_rs, target_step_x, target_step_y)


def isotropize_to_step(surf: Surface, iso_step_um: float) -> Surface:
    """
    Bringt eine Surface auf ein isotropes Raster.

    Was kann man damit machen?
    - Dieselbe Schrittweite in x und y erzwingen
    - 2D-FFT-Auswertung vereinfachen
    """
    return resample_to_step_xy(surf, iso_step_um, iso_step_um)


def maybe_crop_edges_um(surf: Surface, edge_um: float) -> Surface:
    """
    Cropt optional einen Rand gleicher Breite an allen Seiten ab.

    Was kann man damit machen?
    - Randartefakte vor dem Matching entfernen
    - Unsichere oder gestörte Randzonen ausschließen
    """
    edge_um = float(edge_um)
    if edge_um <= 0.0:
        return surf

    W = float(surf.width_um)
    H = float(surf.height_um)

    if 2.0 * edge_um >= W or 2.0 * edge_um >= H:
        raise ValueError(f"edge crop {edge_um} µm ist zu groß für {W:.1f} x {H:.1f} µm")

    return surf.crop((edge_um, W - edge_um, edge_um, H - edge_um), in_units=True)


# ============================================================
# 2D FFT bandpass (200–600 µm) + envelope as MATCH IMAGE
# ============================================================

def bandpass_2d_fft_isotropic(
    surf: Surface,
    wl_min_um: float,
    wl_max_um: float,
    tukey_alpha: float = 0.10,
    remove_dc: bool = True,
) -> Surface:
    """
    Führt einen 2D-Ring-Bandpass im Wellenlängenbereich auf isotropem Raster aus.

    Was kann man damit machen?
    - Nur Strukturen in einem bestimmten Wellenlängenbereich behalten
    - Sehr kurz- und sehr langwellige Anteile unterdrücken
    - Ein bandpassgefiltertes Bild für das Matching erzeugen

    Rückgabe:
    - Surface mit signiertem bandpassgefiltertem Signal
    """
    z = nan_fill(surf.data)
    ny, nx = z.shape

    # Tukey-Fenster zur Randdämpfung vor der FFT
    wy = signal.windows.tukey(ny, alpha=tukey_alpha)
    wx = signal.windows.tukey(nx, alpha=tukey_alpha)
    w2 = np.outer(wy, wx)
    zW = z * w2

    # 2D-FFT in den Frequenzraum
    F = np.fft.fftshift(np.fft.fft2(zW))

    # Frequenzachsen aufbauen
    step = float(surf.step_x)  # isotrop angenommen
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=step))
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=step))
    FX, FY = np.meshgrid(fx, fy)

    # Radiale Frequenz und entsprechende Wellenlänge
    FR = np.sqrt(FX**2 + FY**2)
    WL = np.where(FR > 0, 1.0 / FR, np.inf)

    # Ringfilter im Wellenlängenraum
    mask = (WL >= wl_min_um) & (WL <= wl_max_um)
    if remove_dc:
        mask &= (FR > 0)

    # Zurück in den Ortsraum
    Fb = np.fft.ifftshift(F * mask)
    zb = np.fft.ifft2(Fb).real

    return Surface(zb, surf.step_x, surf.step_y)


def envelope_from_bandpass(band: Surface, smooth_sigma_px: float = 2.0) -> Surface:
    """
    Erzeugt aus dem bandpassgefilterten Signal eine phasenunabhängige Amplitudenkarte.

    Was kann man damit machen?
    - Die Stärke der periodischen Struktur sichtbar machen
    - Ein robusteres Matching-Bild erzeugen, das weniger von der Phase abhängt
    """
    z = nan_fill(band.data)
    amp = np.sqrt(ndimage.gaussian_filter(z * z, sigma=float(smooth_sigma_px)))
    return Surface(amp, band.step_x, band.step_y)


def make_match_map(
    surf_raw: Surface,
    iso_step_um: float,
    wl_min_um: float,
    wl_max_um: float,
    edge_crop_um: float = 0.0,
    detrend_degree: int = 2,
    env_sigma_px: float = 2.0,
) -> tuple[Surface, Surface]:
    """
    Baut aus einer Roh-Surface eine Matching-Karte auf.

    Was kann man damit machen?
    - Fläche vorverarbeiten
    - auf isotropes Raster bringen
    - bandpassfiltern
    - Hüllkurve bilden
    - Hüllkurve standardisieren

    Rückgabe:
    - match_map:
      standardisierte Hüllkurvenkarte für das Matching
    - s_iso:
      vorverarbeitete isotrope Surface für Debugzwecke
    """
    # Optionalen Randcrop anwenden
    s = maybe_crop_edges_um(surf_raw, edge_crop_um)

    # Grundvorverarbeitung
    s = s.level().detrend_polynomial(detrend_degree)

    # Auf isotropes Raster bringen
    s_iso = isotropize_to_step(s, iso_step_um)

    # Bandpass im gewünschten Wellenlängenbereich
    band = bandpass_2d_fft_isotropic(s_iso, wl_min_um=wl_min_um, wl_max_um=wl_max_um)

    # Hüllkurvenkarte erzeugen
    env = envelope_from_bandpass(band, smooth_sigma_px=env_sigma_px)

    # Für Matching standardisieren
    env_z = zscore(env.data)

    return Surface(env_z, env.step_x, env.step_y), s_iso


# ============================================================
# NCC with peak2 and ratio
# ============================================================

def ncc_map_valid(search_img: np.ndarray, template_img: np.ndarray) -> np.ndarray:
    """
    Berechnet eine valid-NCC-Karte zwischen Suchbild und Template.

    Was kann man damit machen?
    - Für jede mögliche Template-Position die Ähnlichkeit bestimmen
    - Das beste Matching per NCC-Maximum finden
    """
    A = zscore(search_img)
    T = zscore(template_img)

    # Kreuzkorrelation
    corr = signal.correlate2d(A, T, mode="valid")

    # Lokale Energie des Suchbildfensters
    ones = np.ones(T.shape, dtype=float)
    local_energy = signal.correlate2d(A * A, ones, mode="valid")

    # Normierung
    denom = np.sqrt(local_energy) * (np.sqrt(np.sum(T * T)) + 1e-12)
    ncc = corr / (denom + 1e-12)

    return np.nan_to_num(ncc, nan=-np.inf)


def best_peak_with_peak2(ncc: np.ndarray, exclude_px: int = 80) -> tuple[int, int, float, float, float]:
    """
    Bestimmt bestes Maximum, zweitbestes Maximum und deren Verhältnis.

    Was kann man damit machen?
    - Neben dem globalen Peak auch die Eindeutigkeit des Ergebnisses bewerten
    - Das Verhältnis peak1 / peak2 als Qualitätsmaß nutzen
    """
    iy, ix = np.unravel_index(np.argmax(ncc), ncc.shape)
    peak1 = float(ncc[iy, ix])

    # Umgebung des Hauptpeaks maskieren, um den zweitbesten Peak zu finden
    ncc2 = ncc.copy()
    y0 = max(0, iy - exclude_px)
    y1 = min(ncc2.shape[0], iy + exclude_px + 1)
    x0 = max(0, ix - exclude_px)
    x1 = min(ncc2.shape[1], ix + exclude_px + 1)
    ncc2[y0:y1, x0:x1] = -np.inf

    peak2 = float(np.max(ncc2))
    ratio = float(peak1 / (peak2 + 1e-12)) if np.isfinite(peak2) else np.inf

    return int(iy), int(ix), peak1, peak2, ratio


# ============================================================
# Rotation + flip matching on match maps
# ============================================================

def rotate_template(img: np.ndarray, ang_deg: float, order: int = 1, reshape: bool = False) -> np.ndarray:
    """
    Rotiert ein Template um einen gegebenen Winkel.

    Was kann man damit machen?
    - Winkelvarianten des Templates testen
    - Das Template an unterschiedliche Ausrichtungen anpassen
    """
    return ndimage.rotate(img, float(ang_deg), reshape=reshape, order=order, mode="nearest")


def apply_flip(img: np.ndarray, flip: str | None) -> np.ndarray:
    """
    Spiegelt ein Bild optional horizontal oder vertikal.

    Was kann man damit machen?
    - Template zusätzlich gespiegelt testen
    - Links/Rechts- oder Oben/Unten-Varianten berücksichtigen
    """
    if flip is None:
        return img
    if flip == "lr":
        return np.fliplr(img)
    if flip == "ud":
        return np.flipud(img)
    raise ValueError(f"Unknown flip {flip}")


def match_rotation_flip_on_maps(
    key_map: Surface,
    nano_map: Surface,
    angle_range: tuple[float, float],
    angle_step: float,
    try_flips: tuple,
    rotate_order: int,
    rotate_reshape: bool,
    peak2_exclusion_px: int,
    verbose: bool = True,
    log_every: int = 1,
) -> dict:
    """
    Führt das eigentliche Matching über Rotation und Spiegelung durch.

    Was kann man damit machen?
    - Für viele Winkel- und Flip-Kombinationen testen
    - Die beste NCC-Position bestimmen
    - peak, peak2 und ratio als Qualitätskriterien erhalten

    Rückgabe:
    - Dictionary mit bestem Match
    """
    key_img = nan_fill(key_map.data)
    nano_img = nan_fill(nano_map.data)

    Hs, Ws = key_img.shape
    angles = np.arange(angle_range[0], angle_range[1] + 1e-9, angle_step)

    best = {"peak": -np.inf}

    t0 = time.time()
    for ai, ang in enumerate(angles, start=1):
        # Template zunächst rotieren
        T0 = rotate_template(nano_img, ang_deg=float(ang), order=rotate_order, reshape=rotate_reshape)

        for flip in try_flips:
            # Dann ggf. spiegeln
            T = apply_flip(T0, flip)

            # Template muss kleiner als Suchbild sein
            if T.shape[0] >= Hs or T.shape[1] >= Ws:
                continue

            # NCC-Karte berechnen
            ncc = ncc_map_valid(key_img, T)

            # Bestes und zweitbestes Maximum finden
            y0, x0, peak1, peak2, ratio = best_peak_with_peak2(ncc, exclude_px=peak2_exclusion_px)

            if peak1 > best["peak"]:
                best = {
                    "peak": float(peak1),
                    "peak2": float(peak2),
                    "ratio": float(ratio),
                    "angle": float(ang),
                    "flip": flip,
                    "y0": int(y0),
                    "x0": int(x0),
                    "template_shape": T.shape,
                }

        if verbose and (ai == 1 or ai % log_every == 0 or ai == len(angles)):
            dt = time.time() - t0
            print(
                f"{ai:>3}/{len(angles)} | best peak={best['peak']:.4f} "
                f"ratio={best.get('ratio', np.nan):.2f} "
                f"@ ang={best.get('angle', np.nan):.2f} flip={best.get('flip', None)} | {dt:.1f}s"
            )

    if best["peak"] == -np.inf:
        raise RuntimeError("No valid match found on match maps")

    return best


# ============================================================
# Crop from RAW keyence using match coordinates
# ============================================================

def crop_raw_by_iso_match(
    key_raw_after_edge_crop: Surface,
    iso_step_um: float,
    x0_px_iso: int,
    y0_px_iso: int,
    template_shape_iso: tuple[int, int],
    nano_target_step_x: float,
    nano_target_step_y: float,
) -> Surface:
    """
    Schneidet aus der rohen Keyence-Fläche den zum Match gehörenden Ausschnitt aus.

    Was kann man damit machen?
    - Das im isotropen Matching gefundene Ergebnis zurück in Rohdatenkoordinaten übertragen
    - Einen physikalisch passenden Ausschnitt aus der Keyence-Rohfläche gewinnen

    Hinweis:
    - nano_target_step_x und nano_target_step_y werden hier nicht direkt verwendet,
      bleiben aber als Interface erhalten
    """
    h_iso, w_iso = template_shape_iso

    # Startkoordinate in µm im isotropen Raster
    x0_um = float(x0_px_iso) * iso_step_um
    y0_um = float(y0_px_iso) * iso_step_um

    # Templategröße in µm
    w_um = float(w_iso) * iso_step_um
    h_um = float(h_iso) * iso_step_um

    return key_raw_after_edge_crop.crop((x0_um, x0_um + w_um, y0_um, y0_um + h_um), in_units=True)


# ============================================================
# Full pipeline
# ============================================================

def run_match_envelope(
    key_path: str,
    nano_path: str,
    wl_min_um: float = 200.0,
    wl_max_um: float = 600.0,
    iso_step_um: float = 0.0,
    key_edge_crop_um: float = 250.0,
    angle_range: tuple[float, float] = (-10.0, 10.0),
    angle_step: float = 0.5,
    try_flips: tuple = (None, "lr", "ud"),
    fine_halfspan: float = 1.0,
    fine_step: float = 0.25,
    roi_margin_um: float = 2000.0,
    peak2_exclusion_px: int = 80,
    min_peak: float = 0.10,
    min_ratio: float = 1.08,
):
    """
    Führt die komplette Matching-Pipeline aus.

    Was kann man damit machen?
    - Keyence- und NanoFocus-Datei laden
    - Matching-Karten erzeugen
    - Grobsuche und Feinsuche über Winkel/Flip durchführen
    - Einen rohen Keyence-Ausschnitt zurückgeben

    Rückgabe:
    - key_crop_raw:
      ausgeschnittener Bereich aus der Keyence-Rohfläche
    - fine:
      Matching-Informationen der Feinsuche
    """
    # Rohdaten laden
    key_raw = Surface.load(key_path)
    nano_raw = Surface.load(nano_path)

    # Falls kein isotroper Schritt vorgegeben ist:
    # kleinste verfügbare Schrittweite aus beiden Datensätzen wählen
    if iso_step_um <= 0.0:
        iso_step_um = float(min(key_raw.step_x, key_raw.step_y, nano_raw.step_x, nano_raw.step_y))

    # Keyence optional an den Rändern croppen
    key_raw_ec = maybe_crop_edges_um(key_raw, key_edge_crop_um)

    # Matching-Karten erzeugen
    key_map, _ = make_match_map(
        surf_raw=key_raw_ec,
        iso_step_um=iso_step_um,
        wl_min_um=wl_min_um,
        wl_max_um=wl_max_um,
        edge_crop_um=0.0,
        detrend_degree=2,
        env_sigma_px=2.0,
    )

    nano_map, _ = make_match_map(
        surf_raw=nano_raw,
        iso_step_um=iso_step_um,
        wl_min_um=wl_min_um,
        wl_max_um=wl_max_um,
        edge_crop_um=0.0,
        detrend_degree=2,
        env_sigma_px=2.0,
    )

    print(f"iso_step_um = {iso_step_um:.4f} µm")
    print(f"key_map shape = {key_map.data.shape}  nano_map shape = {nano_map.data.shape}")

    # ROI im Keyence-Matching-Bild um die Bildmitte definieren
    key_img = key_map.data
    ny, nx = key_img.shape
    cy_um = (ny * iso_step_um) / 2.0
    cx_um = (nx * iso_step_um) / 2.0

    x_min = max(0.0, cx_um - roi_margin_um)
    x_max = min(float(key_map.width_um), cx_um + roi_margin_um)
    y_min = max(0.0, cy_um - roi_margin_um)
    y_max = min(float(key_map.height_um), cy_um + roi_margin_um)

    key_roi = key_map.crop((x_min, x_max, y_min, y_max), in_units=True)
    roi_off_um = (x_min, y_min)

    print(f"ROI auf key_map: {key_roi.data.shape}  offset_um={roi_off_um[0]:.1f},{roi_off_um[1]:.1f}")

    # --------------------------------------------------------
    # COARSE SEARCH
    # Grobsuche mit gröberem Winkelraster und nearest-Rotation
    # --------------------------------------------------------
    print("COARSE")
    coarse = match_rotation_flip_on_maps(
        key_map=key_roi,
        nano_map=nano_map,
        angle_range=angle_range,
        angle_step=angle_step,
        try_flips=try_flips,
        rotate_order=0,
        rotate_reshape=False,
        peak2_exclusion_px=peak2_exclusion_px,
        verbose=True,
        log_every=max(1, int(1 / angle_step)),
    )

    ang0 = float(coarse["angle"])
    flip0 = coarse["flip"]

    # --------------------------------------------------------
    # FINE SEARCH
    # Feinsuche um den besten Grobwinkel herum
    # --------------------------------------------------------
    print("FINE")
    fine = match_rotation_flip_on_maps(
        key_map=key_roi,
        nano_map=nano_map,
        angle_range=(ang0 - fine_halfspan, ang0 + fine_halfspan),
        angle_step=fine_step,
        try_flips=(flip0,),
        rotate_order=1,
        rotate_reshape=False,
        peak2_exclusion_px=peak2_exclusion_px,
        verbose=True,
        log_every=1,
    )

    # Qualitätskriterien prüfen
    if fine["peak"] < float(min_peak):
        raise RuntimeError(f"peak zu klein: {fine['peak']:.4f}")
    if fine["ratio"] < float(min_ratio):
        raise RuntimeError(f"ratio zu klein: {fine['ratio']:.2f}")

    # Position aus ROI-Koordinaten in globale isotrope Koordinaten zurückrechnen
    x0_px_iso = int(round((roi_off_um[0] / iso_step_um) + fine["x0"]))
    y0_px_iso = int(round((roi_off_um[1] / iso_step_um) + fine["y0"]))

    # Aus Rohdaten den gefundenen Bereich ausschneiden
    key_crop_raw = crop_raw_by_iso_match(
        key_raw_after_edge_crop=key_raw_ec,
        iso_step_um=iso_step_um,
        x0_px_iso=x0_px_iso,
        y0_px_iso=y0_px_iso,
        template_shape_iso=fine["template_shape"],
        nano_target_step_x=nano_raw.step_x,
        nano_target_step_y=nano_raw.step_y,
    )

    print("RESULT")
    print(f"peak={fine['peak']:.4f}  peak2={fine['peak2']:.4f}  ratio={fine['ratio']:.2f}")
    print(f"angle={fine['angle']:.2f}  flip={fine['flip']}")
    print(f"top-left iso px: x0={x0_px_iso} y0={y0_px_iso}")
    print(f"crop raw shape: {key_crop_raw.data.shape}  step=({key_crop_raw.step_x:.4f},{key_crop_raw.step_y:.4f}) µm")

    return key_crop_raw, fine


if __name__ == "__main__":
    # Beispielpfade
    key_path = "WSP00_L1_S1.sdf"
    nano_path = "WSP00_L1_S1.nms"

    # Matching ausführen
    crop, info = run_match_envelope(
        key_path=key_path,
        nano_path=nano_path,
        wl_min_um=200.0,
        wl_max_um=600.0,
        key_edge_crop_um=250.0,
        angle_range=(-10.0, 10.0),
        angle_step=1.0,
        fine_halfspan=1.0,
        fine_step=0.25,
        roi_margin_um=3000.0,
        min_peak=0.10,
        min_ratio=1.08,
    )

    # Ergebnis anzeigen
    crop.show()