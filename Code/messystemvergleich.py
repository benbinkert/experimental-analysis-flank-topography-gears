from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from surfalize import Surface


# ============================================================
# Hilfsfunktionen für den Messsystemvergleich
# Damit können NanoFocus- und Keyence-Daten eingelesen,
# vorverarbeitet, zugeschnitten und gemeinsam geplottet werden
# ============================================================

def symmetric_crop_to(surf: Surface, target_w_um: float, target_h_um: float) -> Surface:
    """
    Schneidet eine Oberfläche symmetrisch auf eine Zielgröße zu.

    Was kann man damit machen?
    - Eine größere Oberfläche auf die Größe einer kleineren Referenzfläche bringen
    - Für den Messsystemvergleich identische physikalische Fenster erzeugen
    - Randbereiche links/rechts und oben/unten gleichmäßig abschneiden

    Parameter:
    - surf:
      Eingangsoberfläche
    - target_w_um, target_h_um:
      Zielbreite und Zielhöhe in µm

    Rückgabe:
    - Gecroppte Surface
    """
    if surf.width_um <= target_w_um or surf.height_um <= target_h_um:
        raise ValueError(
            f"Surface zu klein für Crop: surf={surf.width_um:.1f}x{surf.height_um:.1f} µm, "
            f"target={target_w_um:.1f}x{target_h_um:.1f} µm"
        )

    dx = (surf.width_um - target_w_um) / 2.0
    dy = (surf.height_um - target_h_um) / 2.0

    return surf.crop((dx, dx + target_w_um, dy, dy + target_h_um), in_units=True)


def preprocess_surface(surface: Surface, messsystem: str, filtercase: str) -> Surface:
    """
    Führt die Vorverarbeitung einer Oberfläche für den Messsystemvergleich durch.

    Was kann man damit machen?
    - Oberflächen von NanoFocus und Keyence einheitlich vorbereiten
    - Form, Trend und Ausreißer reduzieren
    - Je nach Auswertefall passende Filter anwenden

    Ablauf:
    1. Leveln
    2. Polynomiales Detrending
    3. Thresholding
    4. Auffüllen nicht gemessener Werte
    5. S- und L-Filter je nach Messsystem und Filterfall

    Parameter:
    - messsystem:
      'Nanofocus' oder 'Keyence'
    - filtercase:
      'prozessfrei', 'vergleich' oder 'gesamt'

    Rückgabe:
    - Vorverarbeitete Surface
    """
    # Grundlegende Formkorrektur
    surface = surface.level()
    surface = Surface.detrend_polynomial(surface, degree=2)
    surface = surface.threshold(threshold=(0.25, 0.25))
    surface = surface.fill_nonmeasured_rowwise_linear()

    # S-Filter abhängig vom Messsystem
    if messsystem.lower() == "nanofocus":
        cutoff_s = 1.6
    elif messsystem.lower() == "keyence":
        cutoff_s = 8.0
    else:
        raise ValueError("messsystem muss 'Nanofocus' oder 'Keyence' sein")

    # Filterstrategie abhängig vom Auswertefall
    if filtercase == "prozessfrei":
        cutoff_l = 100.0
        surface = surface.filter(filter_type="bandpass", cutoff=cutoff_s, cutoff2=cutoff_l)

    elif filtercase == "vergleich":
        cutoff_l = 250.0
        cutoff_s = 8.0  # Im Vergleichsfall bewusst fest auf 8 gesetzt
        surface = surface.filter(filter_type="bandpass", cutoff=cutoff_s, cutoff2=cutoff_l)

    elif filtercase == "gesamt":
        surface = surface.filter(filter_type="lowpass", cutoff=cutoff_s)

    else:
        raise ValueError("filtercase muss 'prozessfrei', 'vergleich' oder 'gesamt' sein")

    # Nach der Filterung erneut nicht gemessene Werte auffüllen
    surface = surface.fill_nonmeasured_rowwise_linear()
    return surface


def robust_limits(z: np.ndarray, q=(1, 99)) -> tuple[float, float]:
    """
    Bestimmt robuste Farbgrenzen über Perzentile.

    Was kann man damit machen?
    - Eine gemeinsame Farbschreibweise für mehrere Bilder bestimmen
    - Ausreißer begrenzen, damit der Hauptkontrast besser sichtbar wird
    - Plots verschiedener Oberflächen fair vergleichbar machen

    Parameter:
    - z:
      Datenarray
    - q:
      Unteres und oberes Perzentil

    Rückgabe:
    - (vmin, vmax) für die Farbskala
    """
    z = z[np.isfinite(z)]

    if z.size == 0:
        return -1.0, 1.0

    lo, hi = np.percentile(z, q)

    # Falls beide Grenzen identisch sind, künstlich etwas spreizen
    if lo == hi:
        eps = 1e-6 if lo == 0 else abs(lo) * 0.05
        return float(lo - eps), float(hi + eps)

    return float(lo), float(hi)


def extent_um(s: Surface) -> list[float]:
    """
    Liefert den Plot-Extent einer Surface in µm.

    Was kann man damit machen?
    - imshow mit physikalischen Achsen in µm statt Pixeln darstellen
    - Mehrere Oberflächen mit korrekten physikalischen Koordinaten plotten

    Rückgabe:
    - [xmin, xmax, ymin, ymax]
    """
    return [0.0, float(s.width_um), 0.0, float(s.height_um)]


def crop_x_center_to_width(surf: Surface, target_width_um: float) -> Surface:
    """
    Schneidet eine Oberfläche nur in x-Richtung symmetrisch auf eine Zielbreite zu.

    Was kann man damit machen?
    - Eine Fläche zentral auf eine gewünschte Breite reduzieren
    - Alle Proben auf denselben x-Ausschnitt bringen
    - Den Fokus auf den zentralen Bereich legen, ohne in y zu croppen

    Parameter:
    - target_width_um:
      Zielbreite in µm

    Rückgabe:
    - Gecroppte Surface
    """
    if target_width_um is None or target_width_um <= 0:
        return surf

    w = float(surf.width_um)
    if w <= target_width_um:
        return surf

    x0 = (w - target_width_um) / 2.0
    x1 = x0 + target_width_um

    return surf.crop((x0, x1, 0.0, float(surf.height_um)), in_units=True)


def load_pair(
    nano_path: Path,
    key_path: Path,
    filtercase: str,
    crop_x_um: float | None
) -> tuple[Surface, Surface]:
    """
    Lädt ein NanoFocus-/Keyence-Paar, bringt beide auf dasselbe Fenster
    und führt die Vorverarbeitung durch.

    Was kann man damit machen?
    - Zwei passende Dateien für denselben Bereich direkt vergleichbar machen
    - Keyence auf das NanoFocus-Fenster croppen
    - Optional beide Datensätze noch zentral in x-Richtung beschneiden

    Rückgabe:
    - Tuple (nano_surface, key_surface)
    """
    # Rohdaten laden
    nano_raw = Surface.load(str(nano_path))
    key_raw = Surface.load(str(key_path))

    # Keyence auf die physikalische Fenstergröße von NanoFocus bringen
    key_crop = symmetric_crop_to(key_raw, float(nano_raw.width_um), float(nano_raw.height_um))

    # Beide Oberflächen passend vorverarbeiten
    nano = preprocess_surface(nano_raw, messsystem="Nanofocus", filtercase=filtercase)
    key = preprocess_surface(key_crop, messsystem="Keyence", filtercase=filtercase)

    # Optional zusätzlich in x-Richtung zentral croppen
    if crop_x_um is not None:
        nano = crop_x_center_to_width(nano, crop_x_um)
        key = crop_x_center_to_width(key, crop_x_um)

    return nano, key


# ============================================================
# Plotfunktion für den Messsystemvergleich
# Damit kann eine 2x4-Ansicht für WSP00/WSP03 und S1/S2
# erzeugt werden, mit enger Anordnung und gemeinsamer Farbskala
# ============================================================

def plot_luecke_grid_tighter(
    nano_wsp00_dir: str,
    key_wsp00_dir: str,
    nano_wsp03_dir: str,
    key_wsp03_dir: str,
    luecke: int = 13,
    filtercase: str = "vergleich",
    q_color=(1, 99),
    crop_x_um: float | None = 3000.0,

    # Abstände und Ränder der Gesamtfigur
    wspace: float = 0.02,
    hspace: float = 0.00,
    top: float = 0.93,
    bottom: float = 0.08,
    left: float = 0.08,
    right: float = 0.90,
):
    """
    Erzeugt einen engen 2x4-Vergleichsplot für NanoFocus und Keyence.

    Was kann man damit machen?
    - Für eine bestimmte Lücke dieselben Proben aus zwei Messsystemen vergleichen
    - WSP00 und WSP03 jeweils für S1 und S2 nebeneinander darstellen
    - Eine gemeinsame Farbskala über alle acht Bilder verwenden
    - Eine kompakte, publikationsnahe Darstellung erzeugen

    Anordnung:
    - obere Zeile: NanoFocus
    - untere Zeile: Keyence
    - Spalten: WSP00 S1, WSP00 S2, WSP03 S1, WSP03 S2
    """
    nano_wsp00_dir = Path(nano_wsp00_dir)
    key_wsp00_dir = Path(key_wsp00_dir)
    nano_wsp03_dir = Path(nano_wsp03_dir)
    key_wsp03_dir = Path(key_wsp03_dir)

    # Kombinationen für die vier Spalten definieren
    combos = [
        ("WSP00", "S1", nano_wsp00_dir / f"WSP00_L{luecke}_S1.nms", key_wsp00_dir / f"WSP00_L{luecke}_S1.sdf"),
        ("WSP00", "S2", nano_wsp00_dir / f"WSP00_L{luecke}_S2.nms", key_wsp00_dir / f"WSP00_L{luecke}_S2.sdf"),
        ("WSP03", "S1", nano_wsp03_dir / f"WSP03_L{luecke}_S1.nms", key_wsp03_dir / f"WSP03_L{luecke}_S1.sdf"),
        ("WSP03", "S2", nano_wsp03_dir / f"WSP03_L{luecke}_S2.nms", key_wsp03_dir / f"WSP03_L{luecke}_S2.sdf"),
    ]

    # Alle vier Paare laden und vorbereiten
    loaded = []
    for wsp, S, nano_p, key_p in combos:
        if not nano_p.exists():
            raise FileNotFoundError(f"NanoFocus fehlt: {nano_p}")
        if not key_p.exists():
            raise FileNotFoundError(f"Keyence fehlt: {key_p}")

        nano, key = load_pair(nano_p, key_p, filtercase=filtercase, crop_x_um=crop_x_um)
        loaded.append((wsp, S, nano, key))

    # Gemeinsame robuste Farbbereiche über alle Oberflächen berechnen
    z_all = np.concatenate([surf.data.ravel() for _, _, nano, key in loaded for surf in (nano, key)])
    vmin, vmax = robust_limits(z_all, q=q_color)

    # 2x4-Plot anlegen
    fig, axs = plt.subplots(
        2, 4,
        figsize=(18, 6.2),
        dpi=170,
        sharex="col",
        sharey="row",
    )

    last_im = None

    # Jede Spalte befüllen
    for j, (wsp, S, nano, key) in enumerate(loaded):
        ax_top = axs[0, j]
        ax_bot = axs[1, j]

        # Oben: NanoFocus
        last_im = ax_top.imshow(
            nano.data,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            extent=extent_um(nano),
            aspect="equal",
        )
        ax_top.set_title(f"{wsp} L{luecke} {S}\nNanoFocus", fontsize=10, pad=2)
        ax_top.set_xlabel("")
        ax_top.set_ylabel("y [µm]" if j == 0 else "")
        ax_top.tick_params(labelsize=8, pad=1)

        # Unten: Keyence
        last_im = ax_bot.imshow(
            key.data,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            extent=extent_um(key),
            aspect="equal",
        )
        ax_bot.set_title("Keyence", fontsize=10, pad=2)
        ax_bot.set_xlabel("x [µm]")
        ax_bot.set_ylabel("y [µm]" if j == 0 else "")
        ax_bot.tick_params(labelsize=8, pad=1)

    # Gesamtabstände der Subplots setzen
    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=wspace,
        hspace=hspace
    )

    # Schmale, separate Colorbar rechts
    cax = fig.add_axes([0.915, bottom + 0.03, 0.012, (top - bottom) - 0.06])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("z (nach Preprocess)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Gesamttitel
    fig.suptitle(f"Messsystemvergleich L{luecke}", fontsize=14, y=0.985)

    plt.show()


if __name__ == "__main__":
    plot_luecke_grid_tighter(
        nano_wsp00_dir="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben",
        key_wsp00_dir="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP00/1x7/ErgebnisseMatInSDF",
        nano_wsp03_dir="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP03/ErodierteProben",
        key_wsp03_dir="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP03/Speicher",
        luecke=13,
        filtercase="vergleich",
        q_color=(1, 99),
        crop_x_um=3000.0,
    )