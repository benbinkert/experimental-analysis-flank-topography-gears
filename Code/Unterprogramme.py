import numpy as np
from surfalize import Surface
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from lmfit.models import SineModel
import trimesh
from scipy.io import loadmat
from scipy import ndimage
from pathlib import Path
import math
import xml.etree.ElementTree as ET
from scipy import signal


# ============================================================
# Sammlung von Hilfsfunktionen für Profile, Surfaces und Matching
#
# Was kann man damit machen?
# - 1D-Profile analysieren und Sinusfits durchführen
# - Peak-Abstände in Profilen bestimmen
# - Profile per Threshold bereinigen
# - Oberflächen mit Messlinien visualisieren
# - MATLAB-, Perthometer- und Surface-Daten laden
# - Surfaces zuschneiden, detrenden und statistisch prüfen
# - Keyence-/NanoFocus-Daten für Matching vorbereiten
# ============================================================


# ============================================================
# 1) Sinus-Fit auf Profil
# ============================================================

def fit_and_overlay_sine_multi(profile, start_lambdas_mm=(0.35, 0.40, 0.42, 0.45, 0.50)):
    """
    Fit eines Sinusmodells an ein 1D-Profil.

    Was kann man damit machen?
    - Mehrere Startwerte für die Periodenlänge testen
    - Das am besten passende Sinusmodell auswählen
    - Profil und bestes Fitmodell gemeinsam plotten

    Erwartet:
    - profile.data:
      Profilwerte
    - profile.step:
      Abtastschritt entlang des Profils in µm
    """
    z = np.asarray(profile.data, dtype=float)

    # Profilachse in µm aufbauen
    x = np.arange(len(z), dtype=float) * profile.step

    # Nur finite Werte verwenden
    valid = np.isfinite(z)
    x_fit = x[valid]
    z_fit = z[valid]

    # Mittelwert abziehen, damit der Fit numerisch stabiler wird
    offset0 = np.nanmean(z_fit)
    y_fit = z_fit - offset0

    # Startamplitude robust aus 10-/90-Perzentil schätzen
    amp0 = 0.5 * (np.nanpercentile(y_fit, 90) - np.nanpercentile(y_fit, 10))

    model = SineModel()
    best_result = None
    best_lambda_mm = None

    # Mehrere Startwerte für λ testen
    for lambda_mm in start_lambdas_mm:
        lambda_um = lambda_mm * 1000.0

        # lmfit erwartet die Kreisfrequenz in rad/µm
        freq0 = 2 * np.pi / lambda_um

        params = model.make_params(amplitude=amp0, frequency=freq0, shift=0.0)

        # Frequenzbereich einschränken:
        # entspricht hier ungefähr λ zwischen 300 und 550 µm
        lambda_min_um = 300.0
        lambda_max_um = 550.0
        params["frequency"].min = 2 * np.pi / lambda_max_um
        params["frequency"].max = 2 * np.pi / lambda_min_um

        try:
            result = model.fit(y_fit, params, x=x_fit)
        except Exception:
            # Falls ein Startwert numerisch scheitert, einfach überspringen
            continue

        # Aus der gefitteten Frequenz wieder λ bestimmen
        freq_fit = result.params["frequency"].value
        lambda_fit_um = 2 * np.pi / abs(freq_fit)
        lambda_fit_mm = lambda_fit_um / 1000.0

        # Bestes Modell über chi² auswählen
        if best_result is None or result.chisqr < best_result.chisqr:
            best_result = result
            best_lambda_mm = lambda_fit_mm

    if best_result is None:
        print("Kein Fit erfolgreich.")
        return None

    # Modell auf voller x-Achse auswerten und Offset wieder addieren
    z_model = best_result.eval(x=x) + offset0

    # Plot: Originalprofil + bestes Sinusmodell
    plt.figure()
    plt.plot(x, z, label="Profil")
    plt.plot(x, z_model, label=f"bester Sinusfit (λ = {best_lambda_mm:.4f} mm)")
    plt.xlabel("x [µm]")
    plt.ylabel("z")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Bestes gefittetes lambda: {best_lambda_mm:.6f} mm")
    print(best_result.fit_report())

    return best_result, best_lambda_mm


# ============================================================
# 2) Peak-Abstände (Maxima/Minima)
# ============================================================

def mean_peak_distances(profile, expected_period_um=400, prominence=None):
    """
    Findet Maxima und Minima in einem Profil und berechnet
    die Abstände zwischen benachbarten Peaks.

    Was kann man damit machen?
    - Mittlere Peak-zu-Peak-Abstände bestimmen
    - Maxima- und Minima-Serien getrennt auswerten

    Parameter:
    - expected_period_um:
      Erwartete Periode in µm, daraus wird ein Mindestabstand in Punkten abgeleitet
    - prominence:
      Falls None, wird ein robuster Wert aus den Daten geschätzt
    """
    z = np.asarray(profile.data, dtype=float)
    x = np.arange(len(z), dtype=float) * profile.step

    # Nur finite Werte verwenden
    valid = np.isfinite(z)
    z = z[valid]
    x = x[valid]

    if len(z) < 3:
        raise ValueError("Zu wenige gültige Punkte im Profil.")

    # Mindestabstand in Punkten:
    # etwa 70 % der erwarteten Periode
    distance_pts = max(1, int(0.7 * expected_period_um / profile.step))

    # Prominenz schätzen, falls nicht vorgegeben
    if prominence is None:
        z_low = np.nanpercentile(z, 20)
        z_high = np.nanpercentile(z, 80)
        prominence = 0.2 * (z_high - z_low)

    # Maxima und Minima getrennt finden
    max_idx, _ = find_peaks(z, prominence=prominence, distance=distance_pts)
    min_idx, _ = find_peaks(-z, prominence=prominence, distance=distance_pts)

    x_max = x[max_idx]
    x_min = x[min_idx]

    # Abstände zwischen benachbarten Peaks/Tälern
    max_distances = np.diff(x_max) if len(x_max) >= 2 else np.array([])
    min_distances = np.diff(x_min) if len(x_min) >= 2 else np.array([])

    return {
        "x_max": x_max,
        "x_min": x_min,
        "max_distances": max_distances,
        "min_distances": min_distances,
        "mean_max_distance": np.mean(max_distances) if len(max_distances) else np.nan,
        "mean_min_distance": np.mean(min_distances) if len(min_distances) else np.nan,
        "distance_pts": distance_pts,
        "prominence": prominence,
    }


# ============================================================
# 3) Thresholding für Profile
# ============================================================

def threshold_profile(profile, threshold=0.5, mode="nan"):
    """
    Entfernt oder clippt Extremwerte anhand von Perzentilen.

    Was kann man damit machen?
    - Ausreißer am oberen und unteren Rand entfernen
    - Profile für weitere Auswertung robuster machen

    Parameter:
    - threshold:
      Skalar oder Tupel:
      * 0.5 bedeutet 0.5 % unten und 0.5 % oben
      * (lower_pct, upper_pct) erlaubt asymmetrische Grenzen
    - mode:
      * "nan"  -> Werte außerhalb werden NaN
      * "clip" -> Werte werden auf die Grenzen begrenzt
    """
    z = np.asarray(profile.data, dtype=float).copy()

    if isinstance(threshold, (tuple, list)):
        lower_pct, upper_pct = threshold
    else:
        lower_pct = upper_pct = threshold

    valid = np.isfinite(z)
    if valid.sum() == 0:
        return type(profile)(z, profile.step, profile.length_um)

    lower_limit = np.nanpercentile(z, lower_pct)
    upper_limit = np.nanpercentile(z, 100 - upper_pct)

    if mode == "nan":
        z[z < lower_limit] = np.nan
        z[z > upper_limit] = np.nan
    elif mode == "clip":
        z = np.clip(z, lower_limit, upper_limit)
    else:
        raise ValueError("mode muss 'nan' oder 'clip' sein")

    return type(profile)(z, profile.step, profile.length_um)


# ============================================================
# 4) Surface mit Mouse-Over-Koordinaten anzeigen
# ============================================================

def show_surface_with_coords(surface):
    """
    Zeigt eine Surface und blendet beim Mouse-Over die aktuellen Koordinaten ein.

    Was kann man damit machen?
    - Interaktiv x- und y-Positionen ablesen
    - Geeignete Start-/Endpunkte für Linien oder Profile finden
    """
    fig, ax = plt.subplots()
    surface.plot_2d(ax=ax)

    txt = fig.text(
        0.02, 0.98, "",
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.7)
    )

    def on_move(event):
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            txt.set_text(f"x = {event.xdata:.3f}\ny = {event.ydata:.3f}")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    plt.show()


def create_surface_plot_with_coords(surface, title=None):
    """
    Wie show_surface_with_coords(), gibt aber fig und ax zurück.

    Was kann man damit machen?
    - Nachträglich noch Linien, Marker oder Annotationen ergänzen
    - Den Plot weiterverwenden, statt ihn direkt nur anzuzeigen
    """
    fig, ax = plt.subplots()
    surface.plot_2d(ax=ax)

    if title:
        ax.set_title(title)

    txt = fig.text(
        0.02, 0.98, "",
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.7)
    )

    def on_move(event):
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            txt.set_text(f"x = {event.xdata:.3f}\ny = {event.ydata:.3f}")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    return fig, ax


# ============================================================
# 5) Referenzwelle aufs Profil legen
# ============================================================

def overlay_reference_wave_on_profile(
    profile,
    amplitude=None,
    lambda_mm=0.42065,
    x0=0.0,
    x_offset=0.0,
    z_offset=None
):
    """
    Legt eine Cosinus-Referenzwelle über ein Profil.

    Was kann man damit machen?
    - Visuell prüfen, ob eine angenommene Periodenlänge plausibel ist
    - Phase und Startlage der Referenzwelle variieren
    """
    lambda_um = lambda_mm * 1000.0

    z = np.asarray(profile.data, dtype=float)
    x = np.arange(len(z), dtype=float) * profile.step

    valid = np.isfinite(z)
    if valid.sum() < 2:
        print("Zu wenige gültige Profilpunkte.")
        return

    if amplitude is None:
        z_low = np.nanpercentile(z, 10)
        z_high = np.nanpercentile(z, 90)
        amplitude = 0.5 * (z_high - z_low)

    if z_offset is None:
        z_offset = np.nanmean(z)

    # Effektiver Startpunkt der Welle
    x_start = x0 + x_offset

    # Cosinuswelle
    z_ref = z_offset + amplitude * np.cos(2 * np.pi * (x - x_start) / lambda_um)

    plt.figure()
    plt.plot(x, z, label="Profil z(x)")
    plt.plot(x, z_ref, label=f"Referenzwelle λ = {lambda_mm} mm, x_start = {x_start:.2f} µm")
    plt.xlabel("x [µm]")
    plt.ylabel("z")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# 6) Abstand zwischen zwei Linien aus SVG
# ============================================================

def line_distance_from_svg(svg_path, step=1.0):
    """
    Liest zwei SVG-Linien (<line>) und berechnet deren senkrechten Abstand.

    Was kann man damit machen?
    - Geometrische Abstände aus einer SVG-Konstruktion bestimmen
    - SVG-Einheiten mit einem Skalierungsfaktor in physikalische Einheiten umrechnen
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    lines = []
    for elem in root.iter():
        tag = elem.tag.split('}')[-1]
        if tag == 'line':
            x1 = float(elem.attrib['x1'])
            y1 = float(elem.attrib['y1'])
            x2 = float(elem.attrib['x2'])
            y2 = float(elem.attrib['y2'])
            lines.append(((x1, y1), (x2, y2)))

    if len(lines) < 2:
        raise ValueError("Weniger als zwei SVG-Linien gefunden.")

    (x1, y1), (x2, y2) = lines[0]
    (x3, y3), (x4, y4) = lines[1]

    dx = x2 - x1
    dy = y2 - y1
    n = math.hypot(dx, dy)
    if n == 0:
        raise ValueError("Erste Linie ist degeneriert.")

    # Einheitsnormale der ersten Linie
    nx = -dy / n
    ny = dx / n

    # Abstand eines Punktes der zweiten Linie zur ersten Linie
    d_svg = abs((x3 - x1) * nx + (y3 - y1) * ny)
    d_phys = d_svg * step

    return d_svg, d_phys


# ============================================================
# 7) Surface in 3D-Mesh umwandeln
# ============================================================

def surface_to_mesh(surface, fill_nan=True, z_scale=1.0):
    """
    Wandelt eine Surface in ein Dreiecksmesh um.

    Was kann man damit machen?
    - 3D-Geometrien für Visualisierung oder Export erzeugen
    - Aus einer Surface eine triangulierte Fläche aufbauen
    """
    z = surface.data.astype(float).copy()

    if fill_nan and np.isnan(z).any():
        z = np.nan_to_num(z, nan=np.nanmedian(z))

    ny, nx = z.shape

    xs = np.arange(nx) * surface.step_x
    ys = np.arange(ny) * surface.step_y
    X, Y = np.meshgrid(xs, ys)

    vertices = np.column_stack([
        X.ravel(),
        Y.ravel(),
        (z * z_scale).ravel()
    ])

    # Jedes Gitterquadrat in zwei Dreiecke zerlegen
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v0 = j * nx + i
            v1 = v0 + 1
            v2 = v0 + nx
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    faces = np.asarray(faces, dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


# ============================================================
# 8) Surface + schräge Profillinie plotten
# ============================================================

def plot_surface_with_oblique_line(
    surface,
    x0, y0, x1, y1,
    cmap="jet",
    show_profile=False,
    step_label_um=None,
    tick_half_um=25,
    text_offset_um=40,
    Titel=None
):
    """
    Zeichnet eine schräge Linie auf eine Surface.

    Was kann man damit machen?
    - Visuell prüfen, wo ein obliques Profil entnommen wird
    - Tickmarks und Distanzmarken entlang der Linie einblenden
    - Optional direkt das Profil extrahieren und anzeigen
    """
    z = surface.data

    fig, ax = plt.subplots(dpi=150)

    im = ax.imshow(
        z,
        cmap=cmap,
        extent=(0, surface.width_um, 0, surface.height_um)
    )
    fig.colorbar(im, ax=ax, label="z [µm]")

    # Hauptlinie einzeichnen
    ax.plot([x0, x1], [y0, y1], color="red", linewidth=2)

    dx = x1 - x0
    dy = y1 - y0
    line_length = float(np.hypot(dx, dy))
    if line_length == 0:
        raise ValueError("Start- und Endpunkt dürfen nicht identisch sein.")

    ux = dx / line_length
    uy = dy / line_length

    # Normale auf die Linie
    nx = -uy
    ny = ux

    # Markierungen entlang der Linie
    if step_label_um is not None and step_label_um > 0:
        distances = np.arange(0, line_length + 1e-9, step_label_um)

        for d in distances:
            px = x0 + d * ux
            py = y0 + d * uy

            ax.plot(
                [px - tick_half_um * nx, px + tick_half_um * nx],
                [py - tick_half_um * ny, py + tick_half_um * ny],
                color="white",
                linewidth=1.2
            )

            ax.text(
                px + text_offset_um * nx,
                py + text_offset_um * ny,
                f"{int(round(d))}",
                fontsize=9,
                color="white",
                ha="center",
                va="center",
                bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1.5)
            )

    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_title(Titel)
    ax.set_xlim(0, surface.width_um)
    ax.set_ylim(0, surface.height_um)

    profile = None
    if show_profile:
        profile = surface.get_oblique_profile_fixed(x0, y0, x1, y1)
        profile.show()

    plt.tight_layout()
    plt.show()

    return fig, ax, profile


# ============================================================
# 9) Zoom-Plot um eine Linie
# ============================================================

def plot_zoomed_line(surface, x0, y0, x1, y1, margin_um=500, step_um=100):
    """
    Zeichnet eine Linie auf eine Surface und zoomt auf ihre Umgebung.

    Was kann man damit machen?
    - Einen Bereich um eine Messlinie gezielt vergrößern
    - Markierungen in festen Abständen entlang der Linie setzen
    """
    fig, ax = surface.plot_2d()

    ax.plot([x0, x1], [y0, y1], color="black", linewidth=2)

    dx = x1 - x0
    dy = y1 - y0
    line_length = np.hypot(dx, dy)
    if line_length == 0:
        raise ValueError("Start- und Endpunkt dürfen nicht identisch sein.")

    ux = dx / line_length
    uy = dy / line_length

    nx = -uy
    ny = ux

    distances = np.arange(0, line_length + 1e-9, step_um)
    tick_half = 20
    text_offset = 35

    for d in distances:
        px = x0 + d * ux
        py = y0 + d * uy

        ax.plot(
            [px - tick_half * nx, px + tick_half * nx],
            [py - tick_half * ny, py + tick_half * ny],
            color="black",
            linewidth=1
        )

        ax.text(
            px + text_offset * nx,
            py + text_offset * ny,
            f"{int(round(d))}",
            fontsize=8,
            color="black",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1)
        )

    ax.set_title(f"Oberfläche mit Messlinie\nMesslinienlänge: {line_length:.1f} µm")

    xmin = max(0, min(x0, x1) - margin_um)
    xmax = min(surface.width_um, max(x0, x1) + margin_um)
    ymin = max(0, min(y0, y1) - margin_um)
    ymax = min(surface.height_um, max(y0, y1) + margin_um)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    plt.show()
    return fig, ax


# ============================================================
# 10) MATLAB-Flanke laden
# ============================================================

def load_flanke_from_mat(path):
    """
    Lädt eine MATLAB-Datei mit Struct 'daten' und erzeugt daraus eine Surface.

    Was kann man damit machen?
    - X/Y/Z-Meshes aus MATLAB-Dateien übernehmen
    - Orientierung korrigieren
    - Schrittweiten automatisch bestimmen
    """
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    daten = mat["daten"]

    X = np.array(daten.X, dtype=float)
    Y = np.array(daten.Y, dtype=float)
    Z = np.array(daten.Z, dtype=float)

    step_x = float(np.nanmedian(np.abs(np.diff(X[0, :]))))
    step_y = float(np.nanmedian(np.abs(np.diff(Y[:, 0]))))

    # Orientierung anpassen
    if X.shape[1] > 1 and X[0, 1] < X[0, 0]:
        X = np.fliplr(X)
        Y = np.fliplr(Y)
        Z = np.fliplr(Z)

    if Y.shape[0] > 1 and Y[1, 0] < Y[0, 0]:
        X = np.flipud(X)
        Y = np.flipud(Y)
        Z = np.flipud(Z)

    print(f"X range: {np.nanmin(X):.6f} bis {np.nanmax(X):.6f}")
    print(f"Y range: {np.nanmin(Y):.6f} bis {np.nanmax(Y):.6f}")
    print(f"Z range: {np.nanmin(Z):.6f} bis {np.nanmax(Z):.6f}")
    print(f"step_x: {step_x:.6f}")
    print(f"step_y: {step_y:.6f}")

    surface = Surface(Z, step_x, step_y)
    return surface, X, Y, Z, step_x, step_y


def crop_to_valid_region(X, Y, Z, max_trim_rows=20, max_trim_cols=20):
    """
    Schneidet ein X/Y/Z-Mesh auf den gültigen Bereich zu.

    Was kann man damit machen?
    - NaN-Randbereiche entfernen
    - Eine stabile Bounding-Box des gültigen Messfelds erhalten
    """
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    Z = np.array(Z, dtype=float)

    if X.shape != Z.shape or Y.shape != Z.shape:
        raise ValueError("X, Y und Z müssen dieselbe Form haben.")

    valid = np.isfinite(Z)
    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("Keine gültigen berechneten Punkte in Z vorhanden.")

    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1

    Xc = X[r0:r1, c0:c1]
    Yc = Y[r0:r1, c0:c1]
    Zc = Z[r0:r1, c0:c1]

    # Obere und untere Randzeilen schrittweise abschälen
    for _ in range(max_trim_rows):
        if Zc.shape[0] == 0 or np.all(np.isfinite(Zc[0, :])):
            break
        Xc, Yc, Zc = Xc[1:, :], Yc[1:, :], Zc[1:, :]

    for _ in range(max_trim_rows):
        if Zc.shape[0] == 0 or np.all(np.isfinite(Zc[-1, :])):
            break
        Xc, Yc, Zc = Xc[:-1, :], Yc[:-1, :], Zc[:-1, :]

    # Linke und rechte Randspalten schrittweise abschälen
    for _ in range(max_trim_cols):
        if Zc.shape[1] == 0 or np.all(np.isfinite(Zc[:, 0])):
            break
        Xc, Yc, Zc = Xc[:, 1:], Yc[:, 1:], Zc[:, 1:]

    for _ in range(max_trim_cols):
        if Zc.shape[1] == 0 or np.all(np.isfinite(Zc[:, -1])):
            break
        Xc, Yc, Zc = Xc[:, :-1], Yc[:, :-1], Zc[:, :-1]

    if Zc.size == 0:
        raise ValueError("Nach dem Zuschneiden blieb kein gültiger Bereich übrig.")

    return Xc, Yc, Zc


# ============================================================
# 11) Interpolierter z-Wert bei (x_um, y_um)
# ============================================================

def z_at_um(surface, x_um, y_um):
    """
    Interpoliert den z-Wert einer Surface an einer physikalischen Position.

    Was kann man damit machen?
    - Einen z-Wert an beliebigen x/y-Koordinaten abfragen
    - Punktwerte zwischen Rasterpunkten schätzen

    Hinweis:
    - Hier wird implizit eine y-Spiegelung angenommen
    """
    xpx = x_um / surface.width_um * (surface.size.x - 1)
    ypx = (1 - y_um / surface.height_um) * (surface.size.y - 1)

    return float(
        ndimage.map_coordinates(surface.data, [[ypx], [xpx]], order=1, mode='nearest')[0]
    )


# ============================================================
# 12) Perthometer-Export laden
# ============================================================

def load_perthometer_prf_txt(path_txt: str):
    """
    Liest einen Perthometer-Text-Export mit [PROFILE_VALUES].

    Was kann man damit machen?
    - x- und z-Werte aus einer Textdatei extrahieren
    - Werte von mm nach µm umrechnen
    - Den Profilschritt bestimmen
    """
    path = Path(path_txt)

    data_started = False
    xs = []
    zs = []

    with path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if line.startswith("[PROFILE_VALUES]"):
                data_started = True
                continue
            if not data_started:
                continue

            if not line or line.startswith("//"):
                continue
            if "=" not in line:
                continue

            _, right = line.split("=", 1)
            parts = right.split()
            if len(parts) < 3:
                continue

            x = float(parts[0])
            z = float(parts[2])
            xs.append(x)
            zs.append(z)

    x = np.asarray(xs, dtype=float)
    z = np.asarray(zs, dtype=float)

    if x.size < 2:
        raise ValueError("Zu wenige Punkte im [PROFILE_VALUES]-Block gefunden.")

    x_um = x * 1000.0
    z_um = z * 1000.0

    dx_um = float(np.median(np.diff(x_um)))

    return x_um, z_um, dx_um


# ============================================================
# 13) Trend-Check 1D
# ============================================================

def check_detrend(x_um, z_um, z_detrended, degree=2):
    """
    Prüft, ob ein 1D-Detrend plausibel funktioniert hat.

    Was kann man damit machen?
    - Vorher/Nachher-Polynomkoeffizienten vergleichen
    - Prüfen, ob lineare oder quadratische Trends reduziert wurden
    """
    x = np.asarray(x_um, dtype=float)
    z = np.asarray(z_um, dtype=float)
    zd = np.asarray(z_detrended, dtype=float)

    m0 = np.isfinite(x) & np.isfinite(z)
    m1 = np.isfinite(x) & np.isfinite(zd)

    c0 = np.polyfit(x[m0], z[m0], deg=degree)
    c1 = np.polyfit(x[m1], zd[m1], deg=degree)

    r0 = np.corrcoef(x[m0], z[m0])[0, 1]
    r1 = np.corrcoef(x[m1], zd[m1])[0, 1]

    print("=== Trend-Check ===")
    print(f"Polyfit original (deg={degree}): a2={c0[0]:.3e}, a1={c0[1]:.3e}, a0={c0[2]:.3e}")
    print(f"Polyfit detrended (deg={degree}): a2={c1[0]:.3e}, a1={c1[1]:.3e}, a0={c1[2]:.3e}")
    print(f"Korr(x,z) original:   r={r0:.6f}")
    print(f"Korr(x,z) detrended:  r={r1:.6f}")

    return (c0, c1, r0, r1)


# ============================================================
# 14) 2D-Trend-Check
# ============================================================

def fit_poly2_surface(x, y, z):
    """
    Fit einer quadratischen Fläche:
    z = a*x² + b*y² + c*x*y + d*x + e*y + f

    Was kann man damit machen?
    - Den 2D-Trend einer Fläche approximieren
    - Vorher/Nachher-Koeffizienten vergleichen
    """
    X, Y = np.meshgrid(x, y)

    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_flat = z.ravel()

    mask = np.isfinite(z_flat)
    x_flat = x_flat[mask]
    y_flat = y_flat[mask]
    z_flat = z_flat[mask]

    A = np.column_stack([
        x_flat**2,
        y_flat**2,
        x_flat * y_flat,
        x_flat,
        y_flat,
        np.ones_like(x_flat)
    ])

    coeffs, residuals, rank, s = np.linalg.lstsq(A, z_flat, rcond=None)
    return coeffs, residuals


def corrcoef_safe(a, b):
    """
    Robuste Korrelation mit NaN-Handling.

    Was kann man damit machen?
    - Korrelationen sicher berechnen
    - NaN zurückgeben, wenn zu wenig Information vorhanden ist
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    if len(a) < 2:
        return np.nan
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return np.nan

    return np.corrcoef(a, b)[0, 1]


def check_surface_detrend(surface_original, surface_detrended):
    """
    Prüft die Qualität eines 2D-Detrends.

    Was kann man damit machen?
    - Quadratische Trendanteile vor und nach dem Detrend vergleichen
    - Korrelation von x und y mit der Höhe vor und nach dem Detrend ausgeben
    """
    z_orig = surface_original.data
    z_det = surface_detrended.data

    ny, nx = z_orig.shape

    x = np.arange(nx) * surface_original.step_x
    y = np.arange(ny) * surface_original.step_y

    x = x - np.mean(x)
    y = y - np.mean(y)

    coeffs_orig, resid_orig = fit_poly2_surface(x, y, z_orig)
    coeffs_det, resid_det = fit_poly2_surface(x, y, z_det)

    X, Y = np.meshgrid(x, y)

    print("=== 2D-Polynomfit Originalfläche ===")
    print(f"a (x²)  = {coeffs_orig[0]:.6e}")
    print(f"b (y²)  = {coeffs_orig[1]:.6e}")
    print(f"c (xy)  = {coeffs_orig[2]:.6e}")
    print(f"d (x)   = {coeffs_orig[3]:.6e}")
    print(f"e (y)   = {coeffs_orig[4]:.6e}")
    print(f"f       = {coeffs_orig[5]:.6e}")

    print("\n=== 2D-Polynomfit detrendete Fläche ===")
    print(f"a (x²)  = {coeffs_det[0]:.6e}")
    print(f"b (y²)  = {coeffs_det[1]:.6e}")
    print(f"c (xy)  = {coeffs_det[2]:.6e}")
    print(f"d (x)   = {coeffs_det[3]:.6e}")
    print(f"e (y)   = {coeffs_det[4]:.6e}")
    print(f"f       = {coeffs_det[5]:.6e}")

    print("\n=== Korrelationen ===")
    print(f"r(x, z_orig) = {corrcoef_safe(X, z_orig):.6f}")
    print(f"r(y, z_orig) = {corrcoef_safe(Y, z_orig):.6f}")
    print(f"r(x, z_det)  = {corrcoef_safe(X, z_det):.6f}")
    print(f"r(y, z_det)  = {corrcoef_safe(Y, z_det):.6f}")


# ============================================================
# 15) Center-Crop
# ============================================================

def crop_surface_to_center_size(surface, target_width_um, target_height_um):
    """
    Croppt eine Surface mittig auf eine Zielgröße.

    Was kann man damit machen?
    - Verschiedene Datensätze auf dieselbe physikalische Größe bringen
    - Vergleichbare Ausschnitte erzeugen
    """
    width = surface.width_um
    height = surface.height_um

    if target_width_um > width or target_height_um > height:
        raise ValueError(
            f"Zielgröße ({target_width_um} x {target_height_um} µm) "
            f"ist größer als die Surface ({width:.2f} x {height:.2f} µm)."
        )

    x_center = width / 2
    y_center = height / 2

    x0 = x_center - target_width_um / 2
    x1 = x_center + target_width_um / 2
    y0 = y_center - target_height_um / 2
    y1 = y_center + target_height_um / 2

    return surface.crop((x0, x1, y0, y1))


# ============================================================
# 16) Preprocessing für Matching
# ============================================================

CUTOFF_S = 8.0
CUTOFF_L = 250.0

def preprocess_for_match(surf: Surface, cutoff_s=CUTOFF_S, cutoff_l=CUTOFF_L) -> Surface:
    """
    Führt ein standardisiertes Preprocessing für Matching aus.

    Was kann man damit machen?
    - Formkorrektur, Thresholding und Bandpassfilterung in einer Kette anwenden
    - Datensätze für einen späteren Vergleich besser vorbereiten
    """
    return (
        surf.level()
            .detrend_polynomial(2)
            .threshold(threshold=(0.25, 0.25))
            .fill_nonmeasured(method="nearest")
            .filter(filter_type="bandpass", cutoff=cutoff_s, cutoff2=cutoff_l)
            .fill_nonmeasured(method="nearest")
    )


# ============================================================
# 17) Verzeichnisse
# ============================================================

# Basisverzeichnisse für Dateien
KEYENCE_DIR = Path("/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP00/1x7/matchingAlgo")
NANOFOCUS_DIR = Path("/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben")
OUT_DIR = Path("/Users/benbinkert/PycharmProjects/Bachelorarbeit/Ergebnisse/plots_pairs_wsp00")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_nanofocus_file(stem: str) -> Path | None:
    """
    Sucht zu einem Dateistamm eine passende NanoFocus-Datei.

    Was kann man damit machen?
    - Automatisch verschiedene mögliche Dateiendungen testen
    - Schnell die erste passende Datei finden
    """
    candidates = []
    candidates += list(NANOFOCUS_DIR.glob(f"{stem}.sdf"))
    candidates += list(NANOFOCUS_DIR.glob(f"{stem}.csv"))
    candidates += list(NANOFOCUS_DIR.glob(f"{stem}.txt"))
    candidates += list(NANOFOCUS_DIR.glob(f"{stem}.*"))
    return candidates[0] if candidates else None


# ============================================================
# 18) Robuste Normierung
# ============================================================

def robust_z(img):
    """
    Robuste z-Normierung mit Median und MAD.

    Was kann man damit machen?
    - Bilder robust skalieren, auch wenn Ausreißer vorhanden sind
    - Kontrastunterschiede besser kontrollieren
    """
    img = img.astype(float)
    med = np.nanmedian(img)
    mad = np.nanmedian(np.abs(img - med)) + 1e-12

    # 1.4826 * MAD entspricht ungefähr der Standardabweichung
    return (img - med) / (1.4826 * mad)


# ============================================================
# 19) Zeilenweiser FFT-Bandpass
# ============================================================

def bandpass_rows_fft(z, step_um, wl_min_um, wl_max_um):
    """
    Führt einen FFT-Bandpass zeilenweise durch.

    Was kann man damit machen?
    - Bestimmte Wellenlängenbereiche entlang der x-Richtung isolieren
    - Zeilenweise periodische Strukturen betonen
    """
    z = z.astype(float)
    ny, nx = z.shape
    out = np.zeros_like(z)

    f = np.fft.rfftfreq(nx, d=step_um)
    wl = np.empty_like(f)
    wl[0] = np.inf
    wl[1:] = 1.0 / f[1:]

    mask = (wl >= wl_min_um) & (wl <= wl_max_um)
    mask[0] = False

    win = signal.windows.hann(nx)

    for i in range(ny):
        row = z[i, :]
        row = np.nan_to_num(row, nan=np.nanmedian(row))
        row = signal.detrend(row, type="linear")
        row = row * win
        R = np.fft.rfft(row)
        R[~mask] = 0.0
        out[i, :] = np.fft.irfft(R, n=nx)

    return out


def normalize_energy_per_row(img):
    """
    Normiert jede Zeile auf gleiche RMS-Energie.

    Was kann man damit machen?
    - Schwächere Strukturen sichtbarer machen
    - Den Einfluss sehr energiereicher Zeilen reduzieren
    """
    img = img.astype(float)
    rms = np.sqrt(np.mean(img**2, axis=1, keepdims=True)) + 1e-12
    return img / rms


# ============================================================
# 20) Keyence-artige Darstellungen
# ============================================================

def show_keyence_like_confocal(path_sdf, crop_edge_um=200.0, wl_min=200.0, wl_max=600.0):
    """
    Erzeugt mehrere Visualisierungen einer Surface:
    1) robust skalierte detrendete Oberfläche
    2) zeilenweise bandpassgefilterte Strukturansicht
    3) Gradientenbild

    Was kann man damit machen?
    - Strukturen sichtbar machen, die in der Rohansicht schwer erkennbar sind
    - Konfokal-ähnliche Ansichten für visuelle Vergleiche erzeugen
    """
    surf = Surface.load(path_sdf)

    if crop_edge_um and crop_edge_um > 0:
        surf = surf.crop((crop_edge_um, surf.width_um-crop_edge_um,
                          crop_edge_um, surf.height_um-crop_edge_um), in_units=True)

    # Milder Detrend
    s = surf.level().detrend_polynomial(2)
    z = np.nan_to_num(s.data, nan=np.nanmedian(s.data))

    # 1) robuste Darstellung der detrendeten Fläche
    z_view = robust_z(z)
    lo, hi = np.percentile(z_view, [1, 99])
    Surface(np.clip(z_view, lo, hi), s.step_x, s.step_y).show()

    # 2) Strukturband hervorheben
    band = bandpass_rows_fft(z, step_um=s.step_x, wl_min_um=wl_min, wl_max_um=wl_max)
    band = normalize_energy_per_row(band)
    band = robust_z(band)
    lo, hi = np.percentile(band, [1, 99])
    Surface(np.clip(band, lo, hi), s.step_x, s.step_y).show()

    # 3) Gradientenbild zur Kantenbetonung
    g = ndimage.gaussian_filter(band, sigma=1.0)
    gx = ndimage.sobel(g, axis=1, mode="nearest")
    gy = ndimage.sobel(g, axis=0, mode="nearest")
    grad = np.hypot(gx, gy)
    grad = robust_z(grad)
    lo, hi = np.percentile(grad, [1, 99])
    Surface(np.clip(grad, lo, hi), s.step_x, s.step_y).show()


# ============================================================
# 21) Einfache Statistik einer Surface
# ============================================================

def print_stats(name: str, surf: Surface):
    """
    Gibt Basisstatistiken einer Surface aus.

    Was kann man damit machen?
    - Schnell die wichtigsten Eckdaten einer Fläche prüfen
    - Form, Schrittweite, NaN-Anteil und Verteilung überblicken
    """
    z = surf.data.astype(float)

    print(f"\n--- {name} ---")
    print("shape:", z.shape)
    print("step:", surf.step_x, surf.step_y)
    print("width/height [µm]:", surf.width_um, surf.height_um)
    print("nan ratio:", np.isnan(z).mean())
    print("z min/max:", np.nanmin(z), np.nanmax(z))
    print("z std:", np.nanstd(z))
    print("p1/p99:", np.nanpercentile(z, [1, 99]))