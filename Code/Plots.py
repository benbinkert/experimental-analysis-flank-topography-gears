# ============================================================
# Imports
#
# Was kann man damit machen?
# - Eigene Hilfsfunktionen und Filter einbinden
# - Peak-Analyse, Profile und Surface-Plots durchführen
# - Vergleich von Simulation, Abdruck und Messung visualisieren
# ============================================================

from Code.Unterprogramme import *                 # Eigene Hilfsfunktionen (Surface/Profile/Plot-Werkzeuge)
from Code.Filter import butter_profile_filter     # Butterworth-Filter für 1D-Profile
from matplotlib.collections import LineCollection # Farbige Liniensegmente für Verlaufscodierung
from matplotlib.colors import Normalize           # Normierung für Colormaps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter


# ============================================================
# 1) Referenzwelle + Peak-Abstände aus einer Oberfläche
# ============================================================

def charakteristischeOberfrlächemitReferenzwelle(surface):
    """
    Zweck:
    - Eine Oberfläche grob formkorrigieren
    - Ein Horizontalprofil extrahieren
    - Extremwerte aus dem Profil entfernen
    - Eine Referenzwelle überlagern
    - Peak-Abstände zwischen Maxima und Minima auswerten

    Was kann man damit machen?
    - Charakteristische Strukturabstände einer Oberfläche bestimmen
    - Prüfen, wie gut ein Profil zu einer Referenzwelle passt
    - Mittlere Maxima- und Minima-Abstände ausgeben
    """

    # 1) Leveln:
    # entfernt globalen Offset bzw. eine Grundneigung
    surface = surface.level()

    # 2) Formkorrektur:
    # entfernt großskalige Krümmung oder Trend mit Polynom 2. Grades
    surface_detrended = surface.detrend_polynomial(degree=2)

    # 3) Ausreißer entfernen:
    # medianbasierte Ausreißerreduktion
    # Achtung: hier wird auf "surface" gearbeitet, nicht auf "surface_detrended"
    surface = surface.remove_outliers(n=2, method='median')

    # 4) Horizontalprofil ziehen:
    # Profil wird bei y = 400 µm genommen
    profile = surface_detrended.get_horizontal_profile(y=400)

    # 5) Thresholding:
    # extreme hohe und tiefe Werte maskieren
    profile_thr = threshold_profile(profile, threshold=0.5, mode="nan")

    # 6) Referenzwelle auf das Profil legen
    overlay_reference_wave_on_profile(profile_thr, lambda_mm=0.4123)

    # 7) Peak-Abstände bestimmen
    result = mean_peak_distances(profile_thr, prominence=0.1)

    print("Mittlerer Max->Max-Abstand [µm]:", result["mean_max_distance"])
    print("Mittlerer Min->Min-Abstand [µm]:", result["mean_min_distance"])


# ============================================================
# 2) Schräge Messlinie auf Surface + Oblique-Profil
# ============================================================

def schragemesslinie_oberfläche_profil():
    """
    Zweck:
    - Eine NanoFocus-Oberfläche laden
    - Formkorrektur anwenden
    - Eine schräge Messlinie einzeichnen
    - Das zugehörige schräge Profil extrahieren

    Was kann man damit machen?
    - Visuell kontrollieren, wo ein Oblique-Profil entnommen wird
    - Schräge Strukturen gezielt entlang einer Linie auswerten
    """
    filepath = '/Data/Nanofocus/WSP00/ErodierteProben/WSP00_L1_S1.nms'

    surface = Surface.load(filepath)

    # Level + Detrend
    surface = Surface.level(surface)
    surface = Surface.detrend_polynomial(surface, degree=2)

    # Zoomplot mit eingezeichneter Linie
    plot_zoomed_line(surface, 1036, 0, 1592, 781, margin_um=500, step_um=100)
    plt.show(block=False)

    # Schräges Profil entlang der festen Linie extrahieren
    profile = surface.get_oblique_profile_fixed(1036, 0, 1592, 781)
    profile.show()


# ============================================================
# 3) Flankenlinie filtern & farbig darstellen + Surface-Ausschnitt
# ============================================================

def Filter_Flankenlinie(surface, y):
    """
    Zweck:
    - Ein Horizontalprofil aus einer Fläche ziehen
    - Das Profil hochpassfiltern
    - Original und gefiltertes Profil darstellen
    - Einen Surface-Ausschnitt zur Orientierung zeigen

    Was kann man damit machen?
    - Langwellige Formanteile aus dem Profil entfernen
    - Die verbleibende Struktur farbig kodiert darstellen
    - Sehen, an welcher y-Position das Profil entnommen wurde
    """

    # 1) Horizontalprofil an Position y aus der Fläche holen
    profile = surface.get_horizontal_profile(y)

    z = np.asarray(profile.data, dtype=float)  # Profilwerte
    dx_um = float(profile.step)                # Sampling-Schritt in µm

    # 2) Highpass-Filter:
    # entfernt langwellige Anteile oberhalb der Cutoff-Skala
    z_f = butter_profile_filter(z, dx_um, cutoff_um=350, mode="highpass", order=4)

    # x-Achse des Profils in µm
    x_um = np.arange(len(z)) * dx_um

    # 3) Figure mit zwei Teilplots erzeugen
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(10, 8),
        gridspec_kw={"height_ratios": [2, 1]},
        dpi=150
    )

    # ---------------- Oben: Profilplot ----------------
    ax1.plot(x_um, z, label="Original", alpha=0.4)
    ax1.axhline(0, color="black", linewidth=1, linestyle="--", label="z = 0")

    # Colormap für gefilterte Werte
    cmap = plt.get_cmap("turbo")
    norm = Normalize(vmin=np.nanmin(z_f), vmax=np.nanmax(z_f))

    # Gefilterte Linie in einzelne Segmente zerlegen,
    # damit jedes Segment farblich codiert werden kann
    points = np.array([x_um, z_f]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2)
    lc.set_array(z_f[:-1])
    ax1.add_collection(lc)

    # Zusätzlich Fläche zwischen 0 und Profil farbig füllen
    for i in range(len(x_um) - 1):
        xs = [x_um[i], x_um[i + 1], x_um[i + 1], x_um[i]]
        ys = [0, 0, z_f[i + 1], z_f[i]]
        color = cmap(norm(0.5 * (z_f[i] + z_f[i + 1])))
        ax1.fill(xs, ys, color=color, alpha=0.6, linewidth=0)

    ax1.set_xlabel("Profilweg [µm]")
    ax1.set_ylabel("z [µm]")
    ax1.set_xlim(0, 2000)

    # y-Grenzen robust auf das gefilterte Signal setzen
    ypad = 0.05 * (np.nanmax(z_f) - np.nanmin(z_f) + 1e-12)
    ax1.set_ylim(min(np.nanmin(z_f), 0) - ypad, max(np.nanmax(z_f), 0) + ypad)
    ax1.legend()

    # Colorbar für gefilterte Profilwerte
    cbar1 = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1)
    cbar1.set_label("z [µm]")

    # ---------------- Unten: Surface-Ausschnitt ----------------
    im = ax2.imshow(
        surface.data,
        cmap="jet",
        extent=(0, surface.width_um, 0, surface.height_um),
        aspect="auto"
    )
    ax2.set_xlim(0, 2000)
    ax2.set_ylim(350, 370)

    # Hier könnte zusätzlich die Profillinie eingezeichnet werden:
    # ax2.plot([0, 2000], [y, y], color="black", linewidth=2)

    ax2.set_xlabel("x [µm]")
    ax2.set_ylabel("y [µm]")

    cbar2 = fig.colorbar(im, ax=ax2)
    cbar2.set_label("z [µm]")

    plt.tight_layout()
    plt.show()

    def plot_oberflaechen_ausschnitt(surface, y_min, y_max, x_min=0, x_max=2000, line_y=None):
        """
        Lokale Hilfsfunktion innerhalb von Filter_Flankenlinie.

        Was kann man damit machen?
        - Einen kleinen Surface-Ausschnitt separat darstellen
        - Optional eine horizontale Linie markieren

        Hinweis:
        - Diese Funktion ist nur innerhalb von Filter_Flankenlinie sichtbar
        """
        fig, ax = plt.subplots(figsize=(10, 3), dpi=150)

        im = ax.imshow(
            surface.data,
            cmap="jet",
            extent=(0, surface.width_um, 0, surface.height_um),
            aspect="auto"
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        if line_y is not None:
            ax.plot([x_min, x_max], [line_y, line_y], color="black", linewidth=2)

        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("z [µm]")

        plt.tight_layout()
        plt.show()

        return fig, ax


# ============================================================
# 4) Globale Version: Surface-Ausschnitt plotten
# ============================================================

def plot_oberflaechen_ausschnitt(
    surface, name, y_min, y_max,
    x_min=0, x_max=2000,
    line_y=None, line_x=None,
    aspect_user="equal"
):
    """
    Zeigt einen Ausschnitt einer Oberfläche mit physikalischen Achsen.

    Was kann man damit machen?
    - Beliebige Bereiche einer Surface gezielt darstellen
    - Horizontale oder vertikale Referenzlinien markieren
    - Für Berichte oder Vergleiche einen sauberen Ausschnitt plotten
    """
    fig, ax = plt.subplots(figsize=(10, 3), dpi=150)

    im = ax.imshow(
        surface.data,
        cmap="jet",
        extent=(0, surface.width_um, 0, surface.height_um),
        aspect=aspect_user
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if line_y is not None:
        ax.plot([x_min, x_max], [line_y, line_y], color="red", linewidth=2)

    if line_x is not None:
        ax.plot([line_x, line_x], [y_min, y_max], color="red", linewidth=2)

    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_title(name)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("z [µm]")

    plt.tight_layout()
    plt.show()

    return fig, ax


# ============================================================
# 5) Simulation-CSV Peaks + Abstände
# ============================================================

def plot_simulation_420mue(liniennummer=1, min_distance_pts=3, prominence=None):
    """
    Liest eine CSV mit simulierten Flankenlinien und wertet Peaks aus.

    Was kann man damit machen?
    - Eine bestimmte Simulationslinie auswählen
    - Maxima und Minima finden
    - Mittlere Peak-Abstände bestimmen
    - Den Verlauf mit markierten Peaks plotten
    """
    df = pd.read_csv("/Data/Simulation/WSP00/ECT/WSP00_L1_flankenlinien_rechts.csv")

    grp = df[df["linie"] == liniennummer].copy()
    grp = grp.sort_values("y")

    y = grp["y"].to_numpy(dtype=float)
    z = grp["z"].to_numpy(dtype=float)

    peaks_max, _ = find_peaks(z, distance=min_distance_pts, prominence=prominence)
    peaks_min, _ = find_peaks(-z, distance=min_distance_pts, prominence=prominence)

    y_max = y[peaks_max]
    y_min = y[peaks_min]

    dist_max = np.diff(y_max) if len(y_max) > 1 else np.array([])
    dist_min = np.diff(y_min) if len(y_min) > 1 else np.array([])

    mean_dist_max = float(np.mean(dist_max)) if dist_max.size else np.nan
    mean_dist_min = float(np.mean(dist_min)) if dist_min.size else np.nan

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.plot(y, z, linewidth=1, label=f"Flankenlinie {liniennummer}")

    ax.plot(y_max, z[peaks_max], linestyle="None", marker="^", markersize=8, color="blue",
            label=f"Maxima ({len(y_max)})")
    ax.plot(y_min, z[peaks_min], linestyle="None", marker="v", markersize=8, color="red",
            label=f"Minima ({len(y_min)})")

    ax.set_xlabel("x [mm]", fontsize=14)
    ax.set_ylabel("Abweichung [mm]", fontsize=14)
    ax.set_title("Flankenlinie Simulation (WSP00,L1,Rechts)", fontsize=16)
    ax.set_xlim(0, float(np.max(y)))
    ax.tick_params(axis="both", labelsize=12)

    txt = (
        f"Anzahl Maxima: {len(y_max)}\n"
        f"Anzahl Minima: {len(y_min)}\n"
        f"Ø Abstand Maxima: {mean_dist_max * 1000:.3f} µm\n"
        f"Ø Abstand Minima: {mean_dist_min * 1000:.3f} µm"
    )
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", boxstyle="round,pad=0.4")
    )

    plt.tight_layout()
    plt.show()

    print(f"Anzahl Maxima: {len(y_max)}")
    print(f"Anzahl Minima: {len(y_min)}")
    if dist_max.size:
        print(f"Mittlerer Abstand benachbarter Maxima: {mean_dist_max:.6f} mm")
    else:
        print("Nicht genug Maxima für Abstandsauswertung.")
    if dist_min.size:
        print(f"Mittlerer Abstand benachbarter Minima: {mean_dist_min:.6f} mm")
    else:
        print("Nicht genug Minima für Abstandsauswertung.")

    return {
        "y": y,
        "z": z,
        "maxima_indices": peaks_max,
        "minima_indices": peaks_min,
        "maxima_y": y_max,
        "minima_y": y_min,
        "dist_max": dist_max,
        "dist_min": dist_min,
        "mean_dist_max": mean_dist_max,
        "mean_dist_min": mean_dist_min,
    }


# ============================================================
# 6) Simulation: schräge Linie mit Profil
# ============================================================

def simulation_schraegelinie(titel):
    """
    Lädt eine Simulationsfläche, croppt sie, levelt sie und extrahiert
    ein schräges Profil entlang einer festen Messlinie.

    Was kann man damit machen?
    - Eine Simulationsfläche entlang einer definierten Linie auswerten
    - Topographie und schräges Profil gemeinsam untersuchen
    """
    surface = Surface.load('/Data/Simulation/WSP00/WST_TOPO0_L1_Rechts.sdf')
    surface_cut = surface.crop((0, surface.width_um, 150, 3200))
    surface_cut = surface_cut.level()

    plot_surface_with_oblique_line(surface_cut, 9569, 0, 12092, 3067, step_label_um=100, Titel=titel)

    profile = surface_cut.get_oblique_profile_fixed(9569, 0, 12092, 3067)
    profile.show_real()


# ============================================================
# 7) Simulation: Surface + Profil in EINER Figure
# ============================================================

def plot_simulation_270mue_onefig(titel, min_distance_pts=3, prominence=None, roi_margin_um=1000):
    """
    Zeigt Simulationstopographie und zugehöriges Oblique-Profil in einer gemeinsamen Figure.

    Was kann man damit machen?
    - Die Lage einer Messlinie direkt auf der Oberfläche kontrollieren
    - Peaks im dazugehörigen Profil markieren
    - Mittlere Peak-Abstände für Maxima und Minima bestimmen
    """
    step_label_um = 1000
    tick_half_um = 25
    text_offset_um = 40

    surface = Surface.load('/Data/Simulation/WSP00/WST_TOPO0_L1_Rechts.sdf')
    surface_cut = surface.crop((0, surface_cut.width_um if 'surface_cut' in locals() else surface.width_um, 150, 3200))
    surface_cut = surface_cut.level()

    x0, y0, x1, y1 = 9569, 0, 12092, 3067

    profile = surface_cut.get_oblique_profile_fixed(x0, y0, x1, y1)

    # Profilweg aufbauen
    s = np.asarray(profile.axis_data, dtype=float) if getattr(profile, "axis_data", None) is not None \
        else np.linspace(0, profile.length_um, len(profile.data))
    z = np.asarray(profile.data, dtype=float)

    peaks_max, _ = find_peaks(z, distance=min_distance_pts, prominence=prominence)
    peaks_min, _ = find_peaks(-z, distance=min_distance_pts, prominence=prominence)

    s_max = s[peaks_max]
    s_min = s[peaks_min]

    dist_max = np.diff(s_max) if len(s_max) > 1 else np.array([])
    dist_min = np.diff(s_min) if len(s_min) > 1 else np.array([])

    mean_dist_max = float(np.mean(dist_max)) if dist_max.size else np.nan
    mean_dist_min = float(np.mean(dist_min)) if dist_min.size else np.nan

    # Layout erzeugen
    fig = plt.figure(figsize=(10, 8), dpi=150)
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.15, 1.0],
        width_ratios=[0.08, 1.0, 0.08],
        hspace=0.50,
        wspace=0.05
    )

    ax_spacer = fig.add_subplot(gs[0, 0])
    ax_surf   = fig.add_subplot(gs[0, 1])
    cax       = fig.add_subplot(gs[0, 2])
    ax_prof   = fig.add_subplot(gs[1, :])

    ax_spacer.axis("off")
    fig.suptitle(titel, fontsize=16, fontweight="bold")

    # --- Surface darstellen ---
    im = ax_surf.imshow(
        surface_cut.data,
        cmap="jet",
        extent=(0, surface_cut.width_um, 0, surface_cut.height_um),
        origin="upper"
    )
    fig.colorbar(im, cax=cax, label="z [µm]")

    ax_surf.plot([x0, x1], [y0, y1], color="red", linewidth=2)
    ax_surf.scatter([x0, x1], [y0, y1], s=30, color="white", edgecolor="black", zorder=3)

    # Liniengeometrie bestimmen
    dx, dy = x1 - x0, y1 - y0
    line_length = float(np.hypot(dx, dy))
    ux, uy = dx / line_length, dy / line_length
    nx, ny = -uy, ux

    # Distanzmarken entlang der Linie setzen
    if step_label_um and step_label_um > 0:
        distances = np.arange(0, line_length + 1e-9, step_label_um)
        for d in distances:
            px = x0 + d * ux
            py = y0 + d * uy

            ax_surf.plot(
                [px - tick_half_um * nx, px + tick_half_um * nx],
                [py - tick_half_um * ny, py + tick_half_um * ny],
                color="white", linewidth=1.2, zorder=4
            )
            ax_surf.text(
                px + text_offset_um * nx,
                py + text_offset_um * ny,
                f"{int(round(d))}",
                fontsize=9, color="white",
                ha="center", va="center",
                zorder=5,
                bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1.5)
            )

    # ROI um die Messlinie
    x_min = max(0.0, min(x0, x1) - roi_margin_um)
    x_max = min(float(surface_cut.width_um), max(x0, x1) + roi_margin_um)
    y_min = max(0.0, min(y0, y1) - roi_margin_um)
    y_max = min(float(surface_cut.height_um), max(y0, y1) + roi_margin_um)

    ax_surf.set_xlim(x_min, x_max)
    ax_surf.set_ylim(y_min, y_max)

    ax_surf.set_title("Topographie + Messlinie", fontsize=12)
    ax_surf.set_xlabel("x [µm]")
    ax_surf.set_ylabel("y [µm]")
    ax_surf.set_aspect("equal", adjustable="box")
    ax_surf.tick_params(axis="x", labelsize=9, pad=2)

    # --- Profil darstellen ---
    ax_prof.plot(s, z, linewidth=1)
    ax_prof.plot(s_max, z[peaks_max], linestyle="None", marker="^", markersize=7, color="blue", label="Maxima")
    ax_prof.plot(s_min, z[peaks_min], linestyle="None", marker="v", markersize=7, color="red", label="Minima")

    ax_prof.set_title("Profil entlang Messlinie – Peaks", fontsize=12, pad=14)
    ax_prof.set_xlabel("Profilweg [µm]")
    ax_prof.set_ylabel("Abweichung [µm]")
    ax_prof.set_xlim(0, float(np.max(s)))
    ax_prof.legend(loc="upper right")

    txt = (
        f"Anzahl Maxima: {len(s_max)}\n"
        f"Anzahl Minima: {len(s_min)}\n"
        f"Mittlerer Abstand Maxima: {mean_dist_max:.3f} µm\n"
        f"Mittlerer Abstand Minima: {mean_dist_min:.3f} µm"
    )
    ax_prof.text(
        0.02, 0.98, txt,
        transform=ax_prof.transAxes,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", boxstyle="round,pad=0.4")
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    result = {
        "s": s, "z": z, "profile": profile,
        "maxima_indices": peaks_max, "minima_indices": peaks_min,
        "maxima_s": s_max, "minima_s": s_min,
        "dist_max": dist_max, "dist_min": dist_min,
        "mean_dist_max": mean_dist_max, "mean_dist_min": mean_dist_min,
        "roi": (x_min, x_max, y_min, y_max),
    }
    return result


# ============================================================
# 8) Plot-Funktion: Surface in vorhandene Achse zeichnen
# ============================================================

def plot_surface_excerpt_on_ax(surface, ax, title=None, y_min=0, y_max=None, x_min=0, x_max=None):
    """
    Zeichnet einen Surface-Ausschnitt in ein vorhandenes Axes-Objekt.

    Was kann man damit machen?
    - Mehrere Oberflächen in einem gemeinsamen Grid-Layout darstellen
    - Einen Plot nicht als neue Figure, sondern in ein bestehendes Subplot schreiben
    """
    if y_max is None:
        y_max = surface.height_um
    if x_max is None:
        x_max = surface.width_um

    im = ax.imshow(
        surface.data,
        cmap="jet",
        extent=(0, surface.width_um, 0, surface.height_um),
        aspect="auto"
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if title is not None:
        ax.set_title(title, fontsize=11)

    return im


# ============================================================
# 9) Vergleich: erodiert vs Abdruck vs Simulation
# ============================================================

def plot_vergleich_alle(
    surface_Erodiert, surface_Abdruck, surface_simulation,
    surface_Erodiert_links, surface_Abdruck_links, surface_simulation_links,
    surface_Erodiert_L13, surface_Abdruck_L13, surface_simulation_L13,
    surface_Erodiert_links_L13, surface_Abdruck_links_L13, surface_simulation_links_L13
):
    """
    Zeigt einen 3x4-Vergleich von erodierter Probe, Abdruck und Simulation.

    Was kann man damit machen?
    - Vier Flankenpositionen gleichzeitig vergleichen
    - Zwischen realer Probe, Abdruck und Simulation visuell unterscheiden
    - Einheitliche Ausschnitte nebeneinander darstellen
    """
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), dpi=150)

    col_titles = [
        "Lücke 1 – rechte Flanke",
        "Lücke 1 – linke Flanke",
        "Lücke 13 – rechte Flanke",
        "Lücke 13 – linke Flanke",
    ]

    row_labels = ["Erodierte Probe", "Abdruck", "Simulation"]

    surfaces = [
        [surface_Erodiert, surface_Erodiert_links, surface_Erodiert_L13, surface_Erodiert_links_L13],
        [surface_Abdruck, surface_Abdruck_links, surface_Abdruck_L13, surface_Abdruck_links_L13],
        [surface_simulation, surface_simulation_links, surface_simulation_L13, surface_simulation_links_L13],
    ]

    y_ranges = [
        (330, 390),
        (330, 490),
        (330, 390),
    ]

    last_im = None

    for r in range(3):
        for c in range(4):
            ax = axes[r, c]
            s = surfaces[r][c]
            y_min, y_max = y_ranges[r]

            title = col_titles[c] if r == 0 else None

            last_im = plot_surface_excerpt_on_ax(
                s,
                ax,
                title=title,
                y_min=y_min,
                y_max=y_max,
                x_min=0,
                x_max=min(2000, s.width_um)
            )

            # Nur in der unteren Zeile x-Label
            if r == 2:
                ax.set_xlabel("x [µm]")
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            # Nur in der ersten Spalte y-Label
            if c == 0:
                ax.set_ylabel("y [µm]")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            # Zeilenbeschriftung links
            if c == 0:
                ax.text(
                    -0.42, 0.5, row_labels[r],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=12,
                    fontweight="bold"
                )

    fig.suptitle(
        "Vergleich von Simulation, Abdruck und erodierter Probe",
        fontsize=16,
        fontweight="bold"
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.84,
        top=0.90,
        bottom=0.09,
        wspace=0.12,
        hspace=0.18
    )

    # Gemeinsame Colorbar rechts
    cax = fig.add_axes([0.88, 0.12, 0.02, 0.72])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("z [µm]")

    plt.show(block=True)


# ============================================================
# 10) Helper: _prep
# ============================================================

def _prep(surface, x0, y0, x1, y1, min_distance_pts, prominence, roi_margin_um, step_label_um):
    """
    Bündelt wiederkehrende Rechenschritte für:
    - eine Messlinie auf einer Oberfläche
    - das zugehörige schräge Profil
    - Peak-Analyse
    - ROI-Bestimmung
    - Geometrie für Distanzmarken entlang der Linie

    Was kann man damit machen?
    - Dieselbe Logik mehrfach für Simulation und Messdaten nutzen
    - Oberflächen- und Profilplots konsistent vorbereiten
    """
    profile = surface.get_oblique_profile_fixed(x0, y0, x1, y1)

    # Profilweg aufbauen
    s = np.asarray(profile.axis_data, dtype=float) if getattr(profile, "axis_data", None) is not None \
        else np.linspace(0, profile.length_um, len(profile.data))
    z = np.asarray(profile.data, dtype=float)

    # Peaks bestimmen
    peaks_max, _ = find_peaks(z, distance=min_distance_pts, prominence=prominence)
    peaks_min, _ = find_peaks(-z, distance=min_distance_pts, prominence=prominence)

    s_max = s[peaks_max]
    s_min = s[peaks_min]

    dist_max = np.diff(s_max) if len(s_max) > 1 else np.array([])
    dist_min = np.diff(s_min) if len(s_min) > 1 else np.array([])

    mean_dist_max = float(np.mean(dist_max)) if dist_max.size else np.nan
    mean_dist_min = float(np.mean(dist_min)) if dist_min.size else np.nan

    # ROI um die Messlinie bestimmen
    x_min = max(0.0, min(x0, x1) - roi_margin_um)
    x_max = min(float(surface.width_um), max(x0, x1) + roi_margin_um)
    y_min = max(0.0, min(y0, y1) - roi_margin_um)
    y_max = min(float(surface.height_um), max(y0, y1) + roi_margin_um)

    # Richtungsvektor + Normale der Messlinie
    dx, dy = x1 - x0, y1 - y0
    L = float(np.hypot(dx, dy))
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux

    # Distanzmarken
    distances = np.arange(0, L + 1e-9, step_label_um) if step_label_um and step_label_um > 0 else None

    return dict(
        surface=surface, x0=x0, y0=y0, x1=x1, y1=y1,
        roi=(x_min, x_max, y_min, y_max),
        s=s, z=z,
        peaks_max=peaks_max, peaks_min=peaks_min,
        s_max=s_max, s_min=s_min,
        mean_dist_max=mean_dist_max, mean_dist_min=mean_dist_min,
        ux=ux, uy=uy, nx=nx, ny=ny, distances=distances
    )


# ============================================================
# 11) Simulation vs Erodiert in EINER Figure
# ============================================================

def plot_simulation_and_erodiert_onefig(
    titel,
    min_distance_pts=5,
    prominence=0.2,
    roi_margin_um=1000,
    step_label_um=1000,
    tick_half_um=25,
    text_offset_um=40,
):
    """
    Vergleicht Simulation und erodierte Messung in einer gemeinsamen Figure.

    Was kann man damit machen?
    - Für beide Datensätze dieselbe Auswertelogik anwenden
    - Oben Simulation, unten Messung direkt gegenüberstellen
    - Oberfläche, Messlinie und Profil in einer Darstellung kombinieren
    """

    # ---- Simulation laden + preprocess ----
    sim_surf = Surface.load('/Data/Simulation/WSP00/WST_TOPO0_L1_Rechts.sdf')
    sim_surf = sim_surf.crop((0, sim_surf.width_um, 150, 3200)).level()
    sim = _prep(sim_surf, 9569, 0, 12092, 3067, min_distance_pts, prominence, roi_margin_um, step_label_um)

    # ---- Erodiert laden + preprocess ----
    er_surf = Surface.load('/Data/Nanofocus/WSP00/ErodierteProben/WSP00_L1_S2.nms')
    er_surf = er_surf.level()
    er_surf = Surface.detrend_polynomial(er_surf, degree=2)
    er_surf = er_surf.filter(filter_type='lowpass', cutoff=100)
    er = _prep(er_surf, 3971, 0, 4649, 778, min_distance_pts, prominence, roi_margin_um, step_label_um)

    fig = plt.figure(figsize=(14, 10), dpi=150)
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.0, 1.0, 0.05],
        hspace=0.35,
        wspace=0.25
    )

    ax_sim_surf = fig.add_subplot(gs[0, 0])
    ax_sim_prof = fig.add_subplot(gs[0, 1])
    cax_sim     = fig.add_subplot(gs[0, 2])

    ax_er_surf  = fig.add_subplot(gs[1, 0])
    ax_er_prof  = fig.add_subplot(gs[1, 1])
    cax_er      = fig.add_subplot(gs[1, 2])

    fig.suptitle(titel, fontsize=16, fontweight="bold")

    def _plot_surface(ax, cax, dct, subtitle):
        """
        Lokaler Helper für Surface + Messlinie + Distanzmarken.
        """
        surf = dct["surface"]
        x0, y0, x1, y1 = dct["x0"], dct["y0"], dct["x1"], dct["y1"]
        x_min, x_max, y_min, y_max = dct["roi"]
        ux, uy, nx, ny = dct["ux"], dct["uy"], dct["nx"], dct["ny"]
        distances = dct["distances"]

        im = ax.imshow(
            surf.data,
            cmap="jet",
            extent=(0, surf.width_um, 0, surf.height_um),
            origin="upper"
        )
        fig.colorbar(im, cax=cax, label="z [µm]")

        ax.plot([x0, x1], [y0, y1], color="red", linewidth=2)
        ax.scatter([x0, x1], [y0, y1], s=25, color="white", edgecolor="black", zorder=3)

        if distances is not None:
            for d in distances:
                px = x0 + d * ux
                py = y0 + d * uy
                ax.plot([px - tick_half_um * nx, px + tick_half_um * nx],
                        [py - tick_half_um * ny, py + tick_half_um * ny],
                        color="white", linewidth=1.2, zorder=4)
                ax.text(px + text_offset_um * nx, py + text_offset_um * ny,
                        f"{int(round(d))}",
                        fontsize=8, color="white",
                        ha="center", va="center", zorder=5,
                        bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1.2))

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(subtitle, fontsize=12)
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
        ax.set_aspect("equal", adjustable="box")

    def _plot_profile(ax, dct, subtitle):
        """
        Lokaler Helper für Profil + Peaks + Textbox.
        """
        s, z = dct["s"], dct["z"]
        pmax, pmin = dct["peaks_max"], dct["peaks_min"]
        s_max, s_min = dct["s_max"], dct["s_min"]

        ax.plot(s, z, linewidth=1)
        ax.plot(s_max, z[pmax], linestyle="None", marker="^", markersize=7, color="blue", label="Maxima")
        ax.plot(s_min, z[pmin], linestyle="None", marker="v", markersize=7, color="red", label="Minima")

        ax.set_title(subtitle, fontsize=12, pad=14)
        ax.set_xlabel("Profilweg [µm]")
        ax.set_ylabel("Abweichung [µm]")
        ax.set_xlim(0, float(np.max(s)))
        ax.legend(loc="upper right")

        txt = (
            f"Maxima: {len(s_max)} | Minima: {len(s_min)}\n"
            f"Ø Abstand Max: {dct['mean_dist_max']:.3f} µm\n"
            f"Ø Abstand Min: {dct['mean_dist_min']:.3f} µm"
        )
        ax.text(
            0.02, 0.98, txt,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", boxstyle="round,pad=0.4")
        )

    _plot_surface(ax_sim_surf, cax_sim, sim, "Simulation: Topographie + Messlinie")
    _plot_profile(ax_sim_prof, sim, "Simulation: Profil + Peaks")

    _plot_surface(ax_er_surf, cax_er, er, "Messung (erodiert): Topographie + Messlinie")
    _plot_profile(ax_er_prof, er, "Messung (erodiert): Profil + Peaks")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return {"simulation": sim, "erodiert": er}


# ============================================================
# 12) WSP00 vs WSP03 ROI + Mikrostruktur
# ============================================================

def plot_WSP00_WSP03_15mu(
    titel,
    min_distance_pts=10,
    prominence=0.05,
    roi_margin_um=50,
    step_label_um=100,
    tick_half_um=8,
    text_offset_um=12,
):
    """
    Vergleicht zwei NanoFocus-Flächen (WSP00 und WSP03) in einer ROI
    inklusive schräger Messlinie und Peak-Auswertung.

    Was kann man damit machen?
    - Mikrostrukturen in kleinen ROIs direkt vergleichen
    - Für WSP00 und WSP03 jeweils Surface + Profil gegenüberstellen
    """

    # ---------- WSP00 ----------
    filepath = '/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben/WSP00_L1_S1.nms'
    s00 = Surface.load(filepath)
    s00 = s00.level()
    s00 = Surface.detrend_polynomial(s00, 2)
    s00 = s00.threshold(threshold=(0.25, 0.25))
    s00 = s00.fill_nonmeasured_rowwise_linear()
    s00 = s00.filter(filter_type='lowpass', cutoff=1.6)
    s00 = s00.fill_nonmeasured_rowwise_linear()

    x_min00, x_max00, y_min00, y_max00 = (0, 100, 310, 410)
    s00_roi = s00.crop((x_min00, x_max00, y_min00, y_max00))

    x0_00, y0_00, x1_00, y1_00 = 58, 100, 70, 0
    d00 = _prep(s00_roi, x0_00, y0_00, x1_00, y1_00,
                min_distance_pts, prominence, roi_margin_um, step_label_um)

    # ---------- WSP03 ----------
    filepath = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP03/ErodierteProben/WSP03_L1_S1.nms"
    s03 = Surface.load(filepath)
    s03 = s03.level()
    s03 = Surface.detrend_polynomial(s03, 2)
    s03 = s03.threshold(threshold=(0.25, 0.25))
    s03 = s03.fill_nonmeasured_rowwise_linear()
    s03 = s03.filter(filter_type='lowpass', cutoff=1.6)
    s03 = s03.fill_nonmeasured_rowwise_linear()

    x_min03, x_max03, y_min03, y_max03 = (360, 460, 250, 350)
    s03_roi = s03.crop((x_min03, x_max03, y_min03, y_max03))

    x0_03, y0_03, x1_03, y1_03 = 43, 100, 47, 0
    d03 = _prep(s03_roi, x0_03, y0_03, x1_03, y1_03,
                min_distance_pts, prominence, roi_margin_um, step_label_um)

    # ---------- Plotlayout ----------
    fig = plt.figure(figsize=(14, 10), dpi=150)
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.0, 1.0, 0.05],
        hspace=0.55,
        wspace=0.25
    )

    ax00_surf = fig.add_subplot(gs[0, 0])
    ax00_prof = fig.add_subplot(gs[0, 1])
    cax00     = fig.add_subplot(gs[0, 2])

    ax03_surf = fig.add_subplot(gs[1, 0])
    ax03_prof = fig.add_subplot(gs[1, 1])
    cax03     = fig.add_subplot(gs[1, 2])

    fig.suptitle(titel, fontsize=16, fontweight="bold")

    def _plot_surface(ax, cax, dct, subtitle):
        """
        Lokaler Helper für ROI-Surface + Messlinie.
        """
        surf = dct["surface"]
        x0, y0, x1, y1 = dct["x0"], dct["y0"], dct["x1"], dct["y1"]
        x_min, x_max, y_min, y_max = dct["roi"]
        ux, uy, nx, ny = dct["ux"], dct["uy"], dct["nx"], dct["ny"]
        distances = dct["distances"]

        im = ax.imshow(
            surf.data, cmap="jet",
            extent=(0, surf.width_um, 0, surf.height_um),
            origin="upper"
        )
        fig.colorbar(im, cax=cax, label="z [µm]")

        ax.plot([x0, x1], [y0, y1], color="red", linewidth=2)
        ax.scatter([x0, x1], [y0, y1], s=25, color="white", edgecolor="black", zorder=3)

        if distances is not None:
            for d in distances:
                px = x0 + d * ux
                py = y0 + d * uy
                ax.plot([px - tick_half_um * nx, px + tick_half_um * nx],
                        [py - tick_half_um * ny, py + tick_half_um * ny],
                        color="white", linewidth=1.2, zorder=4)
                ax.text(px + text_offset_um * nx, py + text_offset_um * ny,
                        f"{int(round(d))}", fontsize=8, color="white",
                        ha="center", va="center", zorder=5,
                        bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1.2))

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(subtitle, fontsize=12)
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
        ax.set_aspect("equal", adjustable="box")

    def _plot_profile(ax, dct, subtitle):
        """
        Lokaler Helper für Profil + Peaks der ROI.
        """
        s, z = dct["s"], dct["z"]
        pmax, pmin = dct["peaks_max"], dct["peaks_min"]
        s_max, s_min = dct["s_max"], dct["s_min"]

        ax.plot(s, z, linewidth=1)
        ax.plot(s_max, z[pmax], linestyle="None", marker="^", markersize=7, color="blue", label="Maxima")
        ax.plot(s_min, z[pmin], linestyle="None", marker="v", markersize=7, color="red", label="Minima")

        ax.set_title(subtitle, fontsize=12, pad=14)
        ax.set_xlabel("Profilweg [µm]")
        ax.set_ylabel("z [µm]")
        ax.set_xlim(0, float(np.max(s)))
        ax.legend(loc="upper right", fontsize=8)

        txt = (
            f"Maxima: {len(s_max)} | Minima: {len(s_min)}\n"
            f"Ø Abstand Max: {dct['mean_dist_max']:.2f} µm\n"
            f"Ø Abstand Min: {dct['mean_dist_min']:.2f} µm"
        )
        ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                ha="left", va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", boxstyle="round,pad=0.4"))

    _plot_surface(ax00_surf, cax00, d00, "WSP00: ROI + Messlinie")
    _plot_profile(ax00_prof, d00, "WSP00: Profil + Peaks")

    _plot_surface(ax03_surf, cax03, d03, "WSP03: ROI + Messlinie")
    _plot_profile(ax03_prof, d03, "WSP03: Profil + Peaks")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return {"WSP00": d00, "WSP03": d03}


# ============================================================
# 13) Horizontalprofil + Peaks + Textbox
# ============================================================

def plot_horizontal_profile_with_peaks(surface, y_um, titel=None, min_distance_pts=5, prominence=0.2):
    """
    Zieht ein Horizontalprofil aus einer Fläche und markiert Peaks.

    Was kann man damit machen?
    - Für eine feste y-Höhe Peak-Abstände bestimmen
    - Maxima und Minima im Horizontalprofil vergleichen
    """
    profile = surface.get_horizontal_profile(y=y_um)

    if getattr(profile, "axis_data", None) is not None:
        s = np.asarray(profile.axis_data, dtype=float)
    else:
        s = np.linspace(0, surface.width_um, len(profile.data))

    z = np.asarray(profile.data, dtype=float)

    peaks_max, _ = find_peaks(z, distance=min_distance_pts, prominence=prominence)
    peaks_min, _ = find_peaks(-z, distance=min_distance_pts, prominence=prominence)

    s_max = s[peaks_max]
    s_min = s[peaks_min]

    dist_max = np.diff(s_max) if len(s_max) > 1 else np.array([])
    dist_min = np.diff(s_min) if len(s_min) > 1 else np.array([])

    mean_dist_max = float(np.mean(dist_max)) if dist_max.size else np.nan
    mean_dist_min = float(np.mean(dist_min)) if dist_min.size else np.nan

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(s, z, linewidth=1, label=f"Horizontalprofil y={y_um:.1f} µm")

    ax.plot(s_max, z[peaks_max], linestyle="None", marker="^", markersize=7, color="blue",
            label=f"Maxima ({len(s_max)})")
    ax.plot(s_min, z[peaks_min], linestyle="None", marker="v", markersize=7, color="red",
            label=f"Minima ({len(s_min)})")

    ax.set_xlabel("x [µm]", fontsize=14)
    ax.set_ylabel("Abweichung [µm]", fontsize=14)
    ax.set_xlim(float(np.min(s)), float(np.max(s)))

    if titel is None:
        titel = f"Horizontalprofil mit Peaks (y={y_um:.1f} µm)"
    ax.set_title(titel, pad=12, fontsize=16)

    txt = (
        f"Anzahl Maxima: {len(s_max)}\n"
        f"Anzahl Minima: {len(s_min)}\n"
        f"Ø Abstand Maxima: {mean_dist_max:.3f} µm\n"
        f"Ø Abstand Minima: {mean_dist_min:.3f} µm"
    )
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", boxstyle="round,pad=0.4")
    )

    plt.tight_layout()
    plt.show()

    return {
        "profile": profile,
        "s": s,
        "z": z,
        "peaks_max": peaks_max,
        "peaks_min": peaks_min,
        "s_max": s_max,
        "s_min": s_min,
        "mean_dist_max": mean_dist_max,
        "mean_dist_min": mean_dist_min,
    }


# ============================================================
# 14) Peaks + Abstände direkt aus einem Profile
# ============================================================

def peaks_and_distances_profile(profile, title="Profil + Peaks",
                                min_distance_um=50, prominence=None,
                                show=True):
    """
    Wertet Peaks direkt aus einem Profilobjekt aus.

    Was kann man damit machen?
    - Maxima- und Minima-Abstände direkt aus einem surfalize.Profile berechnen
    - Optional sofort einen Plot mit markierten Peaks anzeigen
    """
    z = np.asarray(profile.data, dtype=float)
    dx = float(profile.step)
    s = np.arange(z.size) * dx

    min_dist_pts = max(1, int(round(min_distance_um / dx)))

    peaks_max, _ = find_peaks(z, distance=min_dist_pts, prominence=prominence)
    peaks_min, _ = find_peaks(-z, distance=min_dist_pts, prominence=prominence)

    s_max = s[peaks_max]
    s_min = s[peaks_min]

    dist_max = np.diff(s_max) if s_max.size > 1 else np.array([])
    dist_min = np.diff(s_min) if s_min.size > 1 else np.array([])

    mean_max = float(np.mean(dist_max)) if dist_max.size else np.nan
    mean_min = float(np.mean(dist_min)) if dist_min.size else np.nan

    if show:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        ax.plot(s, z, linewidth=1)

        ax.plot(s_max, z[peaks_max], linestyle="None", marker="^",
                markersize=7, color="blue", label=f"Maxima ({len(s_max)})")
        ax.plot(s_min, z[peaks_min], linestyle="None", marker="v",
                markersize=7, color="red", label=f"Minima ({len(s_min)})")

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Profilweg [µm]", fontsize=12)
        ax.set_ylabel("z [µm]", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_xlim(0, s[-1])
        ax.legend(fontsize=10)

        txt = f"Ø Δ Max: {mean_max:.2f} µm\nØ Δ Min: {mean_min:.2f} µm"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                ha="left", va="top", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", boxstyle="round,pad=0.4"))

        plt.tight_layout()
        plt.show()

    print(f"Anzahl Maxima: {len(s_max)} | Ø Abstand Maxima: {mean_max:.3f} µm")
    print(f"Anzahl Minima: {len(s_min)} | Ø Abstand Minima: {mean_min:.3f} µm")

    return {
        "s": s,
        "z": z,
        "peaks_max": peaks_max,
        "peaks_min": peaks_min,
        "s_max": s_max,
        "s_min": s_min,
        "dist_max": dist_max,
        "dist_min": dist_min,
        "mean_dist_max": mean_max,
        "mean_dist_min": mean_min,
        "dx": dx,
    }


# ============================================================
# 15) "Mehr Peaks": detrend + glätten + 2-pass peak filling
# ============================================================

def peaks_more(
    s_um, z,
    detrend_window_um=600,
    smooth_window_um=40,
    distance_um=80,
    prominence_frac=0.04,
    fill_pass=True,
    fill_prominence_frac=0.02,
    title="Peaks (more)"
):
    """
    Robustere Peak-Detektion mit Detrending, Glättung und optionalem zweiten Pass.

    Was kann man damit machen?
    - Mehr relevante Peaks finden als mit einfachem find_peaks
    - Langsame Trends entfernen
    - Das Signal glätten und adaptive Prominence verwenden
    - In einem zweiten Pass Lücken in der Peakliste auffüllen
    """
    s_um = np.asarray(s_um, float)
    z = np.asarray(z, float)
    ds = float(np.median(np.diff(s_um)))

    def _odd(win_um):
        """
        Hilfsfunktion:
        wandelt eine Fenstergröße in µm in eine ungerade Fensterlänge in Samples um
        """
        n = int(np.round(win_um / ds))
        n = max(n, 5)
        return n + 1 if n % 2 == 0 else n

    # 1) Trend entfernen
    w_det = min(_odd(detrend_window_um), z.size - (1 - z.size % 2))
    trend = savgol_filter(z, w_det, polyorder=2, mode="interp")
    z_dt = z - trend

    # 2) Signal glätten
    w_sm = min(_odd(smooth_window_um), z_dt.size - (1 - z_dt.size % 2))
    z_sm = savgol_filter(z_dt, w_sm, polyorder=2, mode="interp")

    # 3) adaptive prominence aus Signalamplitude bestimmen
    amp = np.nanmax(z_sm) - np.nanmin(z_sm)
    prom = max(1e-12, prominence_frac * amp)
    prom_fill = max(1e-12, fill_prominence_frac * amp)

    dist_pts = max(1, int(distance_um / ds))

    # Hauptpass
    pmax, _ = find_peaks(z_sm, distance=dist_pts, prominence=prom)
    pmin, _ = find_peaks(-z_sm, distance=dist_pts, prominence=prom)

    if fill_pass:
        # Zweiter Pass mit lockereren Bedingungen
        pmax2, _ = find_peaks(z_sm, distance=max(1, dist_pts // 2), prominence=prom_fill)
        pmin2, _ = find_peaks(-z_sm, distance=max(1, dist_pts // 2), prominence=prom_fill)

        def merge(base, extra, min_sep_pts):
            """
            Fügt zusätzliche Peaks nur hinzu, wenn sie weit genug
            von vorhandenen Peaks entfernt liegen.
            """
            base = list(base)
            for p in extra:
                if all(abs(p - b) >= min_sep_pts for b in base):
                    base.append(p)
            return np.array(sorted(base), dtype=int)

        pmax = merge(pmax, pmax2, min_sep_pts=max(1, dist_pts // 2))
        pmin = merge(pmin, pmin2, min_sep_pts=max(1, dist_pts // 2))

    s_max, s_min = s_um[pmax], s_um[pmin]
    dmax, dmin = np.diff(s_max), np.diff(s_min)

    mean_dmax = float(np.mean(dmax)) if dmax.size else np.nan
    mean_dmin = float(np.mean(dmin)) if dmin.size else np.nan

    # Plot der Ergebnisse
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(s_um, z, alpha=0.25, lw=1, label="Original")
    ax.plot(s_um, z_sm, lw=1.3, label="Detrended+Smoothed")
    ax.plot(s_max, z_sm[pmax], "^", ms=8, color="blue", label=f"Maxima ({len(s_max)})")
    ax.plot(s_min, z_sm[pmin], "v", ms=8, color="red", label=f"Minima ({len(s_min)})")

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Profilweg [µm]", fontsize=13)
    ax.set_ylabel("z (detrended) [µm]", fontsize=13)
    ax.set_xlim(float(np.min(s_um)), float(np.max(s_um)))
    ax.legend(fontsize=10)

    txt = f"prom={prom:.4g} | dist={distance_um}µm\nØΔMax={mean_dmax:.2f}µm | ØΔMin={mean_dmin:.2f}µm"
    ax.text(0.015, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"), fontsize=10)

    plt.tight_layout()
    plt.show()

    return {
        "pmax": pmax, "pmin": pmin,
        "s_max": s_max, "s_min": s_min,
        "mean_dmax": mean_dmax, "mean_dmin": mean_dmin,
        "z_used": z_sm
    }


# ============================================================
# 16) Preprocess-Funktionen
# ============================================================

def load_preprocess_Gesamtbild(path, cutoff_s=1.6):
    """
    Lädt eine Fläche und bereitet sie für das Gesamtbild auf.

    Was kann man damit machen?
    - Eine vollständige Fläche für großskalige Visualisierung vorbereiten
    - Formkorrektur, Thresholding und S-Filterung in einem Schritt anwenden
    """
    return (Surface.load(path)
            .level()
            .detrend_polynomial(2)
            .threshold(threshold=(0.25, 0.25))
            .fill_nonmeasured(method="nearest")
            .filter(filter_type="lowpass", cutoff=cutoff_s)
            .fill_nonmeasured(method="nearest"))


def load_preprocess_Prozessfrei(path, cutoff_s=1.6, cutoff_l=100.0):
    """
    Lädt eine Fläche und bereitet sie für die prozessfreie Betrachtung auf.

    Was kann man damit machen?
    - Einen bandpassgefilterten Anteil zwischen lambda_s und lambda_c erzeugen
    - Prozessrelevante Anteile ohne sehr langwellige Form betrachten
    """
    return (Surface.load(path)
            .level()
            .detrend_polynomial(2)
            .threshold(threshold=(0.25, 0.25))
            .fill_nonmeasured(method="nearest")
            .filter(filter_type="bandpass", cutoff=cutoff_s, cutoff2=cutoff_l)
            .fill_nonmeasured(method="nearest"))


def load_preprocess_Vergleich(path, cutoff_s=16.0, cutoff_l=400.0):
    """
    Lädt eine Fläche und bereitet sie für den Vergleichsfall auf.

    Was kann man damit machen?
    - Vergleichbare bandpassgefilterte Flächen erzeugen
    - Messsysteme oder Proben unter derselben Filterung gegenüberstellen

    Hinweis:
    - Die gesetzten Defaults sind hier cutoff_s=16 und cutoff_l=400
    """
    return (Surface.load(path)
            .level()
            .detrend_polynomial(2)
            .threshold(threshold=(0.25, 0.25))
            .fill_nonmeasured(method="nearest")
            .filter(filter_type="bandpass", cutoff=cutoff_s, cutoff2=cutoff_l)
            .fill_nonmeasured(method="nearest"))


# ============================================================
# 17) Plot: WSP00 vs WSP03 (4x1)
# ============================================================

def plot_wsp00_wsp03_onecol_S1S2(
    luecke_label: str,
    paths: dict,
    suptitle_extra="Setup",
    preprocess_fn=load_preprocess_Gesamtbild,
    preprocess_kwargs=None,
    robust_percentiles=(1, 99)
):
    """
    Vergleicht WSP00 und WSP03 für rechte und linke Flanke in einer 4x1-Anordnung.

    Was kann man damit machen?
    - Zwei Zustände oder Strategien direkt nebeneinander vergleichen
    - Dieselbe Farbschreibweise über alle vier Flächen verwenden
    """
    if preprocess_kwargs is None:
        preprocess_kwargs = {}

    s00_s1 = preprocess_fn(paths["WSP00_S1"], **preprocess_kwargs)
    s03_s1 = preprocess_fn(paths["WSP03_S1"], **preprocess_kwargs)
    s00_s2 = preprocess_fn(paths["WSP00_S2"], **preprocess_kwargs)
    s03_s2 = preprocess_fn(paths["WSP03_S2"], **preprocess_kwargs)

    surfs = [s00_s1, s03_s1, s00_s2, s03_s2]

    z_all = np.concatenate([s.data.ravel() for s in surfs])
    z_all = z_all[np.isfinite(z_all)]
    vmin, vmax = np.percentile(z_all, robust_percentiles)

    fig, axes = plt.subplots(4, 1, figsize=(8.6, 12), dpi=150, sharex=True)

    fig.suptitle(
        f"Vergleich NanoFocus – {luecke_label} (Linke Flanke / Rechte Flanke) – {suptitle_extra}",
        fontsize=14, fontweight="bold", y=0.99
    )

    titles = [
        f"{luecke_label} – WSP00 Rechte Flanke",
        f"{luecke_label} – WSP03 Rechte Flanke",
        f"{luecke_label} – WSP00 Linke Flanke",
        f"{luecke_label} – WSP03 Linke Flanke",
    ]

    im_last = None
    for ax, surf, ttl in zip(axes, surfs, titles):
        im_last = ax.imshow(
            surf.data,
            cmap="jet",
            extent=(0, surf.width_um, 0, surf.height_um),
            vmin=vmin, vmax=vmax,
            aspect="equal"
        )
        ax.set_title(ttl, fontsize=11)
        ax.set_ylabel("y [µm]")

    axes[-1].set_xlabel("x [µm]")

    fig.subplots_adjust(left=0.12, right=0.84, top=0.95, bottom=0.06, hspace=0.28)

    cax = fig.add_axes([0.87, 0.12, 0.03, 0.78])
    cbar = fig.colorbar(im_last, cax=cax)
    cbar.set_label("z [µm]")

    plt.show()
    return fig, axes


# ============================================================
# 18) Messsystemvergleich: NanoFocus vs Keyence (4x1)
# ============================================================

def plot_messsystemvergleich_onecol_S1S2(
    luecke_label: str,
    paths: dict,
    suptitle_extra="Setup",
    preprocess_fn=load_preprocess_Vergleich,
    preprocess_kwargs=None,
    robust_percentiles=(1, 99),
    scale_mode="per_surface"
):
    """
    Vergleicht NanoFocus und Keyence für rechte und linke Flanke.

    Was kann man damit machen?
    - Für beide Messsysteme dieselben Flanken direkt gegenüberstellen
    - Verschiedene Skalierungsmodi testen:
      * global
      * per_system
      * per_surface
    - Pro Plot eine eigene Colorbar oder gemeinsame Skalenlogik verwenden
    """
    if preprocess_kwargs is None:
        preprocess_kwargs = {}

    nano_s1 = preprocess_fn(paths["Nanofocus_S1"], **preprocess_kwargs)
    key_s1  = preprocess_fn(paths["Keyence_S1"], **preprocess_kwargs)
    nano_s2 = preprocess_fn(paths["Nanofocus_S2"], **preprocess_kwargs)
    key_s2  = preprocess_fn(paths["Keyence_S2"], **preprocess_kwargs)

    surfs = [nano_s1, key_s1, nano_s2, key_s2]

    titles = [
        f"{luecke_label} – Nanofocus Rechte Flanke",
        f"{luecke_label} – Keyence Rechte Flanke",
        f"{luecke_label} – Nanofocus Linke Flanke",
        f"{luecke_label} – Keyence Linke Flanke",
    ]

    fig, axes = plt.subplots(4, 1, figsize=(9.5, 12), dpi=150, sharex=True)

    fig.suptitle(
        f"Messsystemvergleich – {luecke_label} – {suptitle_extra}",
        fontsize=14, fontweight="bold", y=0.99
    )

    # Globale Skala vorbereiten
    if scale_mode == "global":
        z_all = np.concatenate([s.data.ravel() for s in surfs])
        z_all = z_all[np.isfinite(z_all)]
        vmin_global, vmax_global = np.percentile(z_all, robust_percentiles)

    # Skala pro Messsystem vorbereiten
    elif scale_mode == "per_system":
        z_nano = np.concatenate([nano_s1.data.ravel(), nano_s2.data.ravel()])
        z_key  = np.concatenate([key_s1.data.ravel(), key_s2.data.ravel()])
        z_nano = z_nano[np.isfinite(z_nano)]
        z_key  = z_key[np.isfinite(z_key)]

        vmin_nano, vmax_nano = np.percentile(z_nano, robust_percentiles)
        vmin_key,  vmax_key  = np.percentile(z_key, robust_percentiles)

    for ax, surf, ttl in zip(axes, surfs, titles):

        if scale_mode == "global":
            vmin, vmax = vmin_global, vmax_global

        elif scale_mode == "per_system":
            if "Nanofocus" in ttl:
                vmin, vmax = vmin_nano, vmax_nano
            else:
                vmin, vmax = vmin_key, vmax_key

        elif scale_mode == "per_surface":
            z = surf.data[np.isfinite(surf.data)]
            vmin, vmax = np.percentile(z, robust_percentiles)

        else:
            raise ValueError("scale_mode muss 'global', 'per_system' oder 'per_surface' sein.")

        im = ax.imshow(
            surf.data,
            cmap="jet",
            extent=(0, surf.width_um, 0, surf.height_um),
            vmin=vmin, vmax=vmax,
            aspect="equal"
        )

        ax.set_title(ttl, fontsize=11)
        ax.set_ylabel("y [µm]")

        # Eigene Colorbar pro Plot
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("z [µm]", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        if vmin < 0 < vmax:
            ticks = [vmin, 0.0, vmax]
        elif vmax <= 0:
            ticks = [vmin, 0.5 * (vmin + vmax), vmax]
        else:
            ticks = [vmin, 0.5 * (vmin + vmax), vmax]

        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    axes[-1].set_xlabel("x [µm]")
    fig.subplots_adjust(left=0.12, right=0.92, top=0.95, bottom=0.06, hspace=0.38)

    plt.show()
    return fig, axes