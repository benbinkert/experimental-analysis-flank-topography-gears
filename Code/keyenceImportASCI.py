# asc_to_sdf.py
# ------------------------------------------------------------
# Liest Keyence-ASCII-Dateien mit x-, y- und z-Werten ein,
# baut daraus eine Surfalize-Oberfläche auf und speichert
# diese als SDF-Datei.
#
# Was kann man damit machen?
# - ASCII-Exportdaten aus Keyence in ein Surfalize-kompatibles Format umwandeln
# - Regelmäßige Raster direkt übernehmen
# - Unregelmäßige Punktwolken auf ein regelmäßiges Raster interpolieren
# - Die erzeugte SDF-Datei direkt wieder in weiteren Skripten verwenden
#
# Unterstützt:
# - reguläres Raster (direkte Rasterzuordnung)
# - unregelmäßige Punkte (Interpolation via scipy.griddata)
# ------------------------------------------------------------

from pathlib import Path
import numpy as np
from surfalize import Surface
from scipy.interpolate import griddata


# Eingabedatei: ASCII-Datei mit x, y, z
ASC_PATH = Path("/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP00/1x7_Reflektion/ASCI/WSP00_L1_S1.asc")

# Ursprünglich gesetzter Pfad, wird im Skript aber nicht weiter verwendet
OUT_PATH = Path("/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP00/1x7_Reflektion/ASCI/WSP00_L1_S1.asc")

# Ausgabedatei: gleiche Datei, aber mit Endung .sdf
OUT_SDF = ASC_PATH.with_suffix(".sdf")

# Wenn Interpolation nötig ist:
# - "auto" bestimmt die Schrittweite aus den Daten
# - alternativ eine feste Zahl angeben
GRID_STEP_MODE = "auto"

# Skalierung der x/y-Werte
# Beispiel:
# - wenn die ASCII-Datei x und y in mm enthält und Surfalize in µm arbeiten soll,
#   dann ist SCALE_XY = 1000.0 sinnvoll
SCALE_XY = 1000.0

# Skalierung der z-Werte
# Optional, falls z ebenfalls in mm vorliegt und in µm umgerechnet werden soll
SCALE_Z = 1000.0


def read_asc_xyz(path: Path) -> np.ndarray:
    """
    Liest eine ASCII-Datei mit drei Spalten (x, y, z) ein.

    Was kann man damit machen?
    - ASCII-Exportdateien unterschiedlicher Schreibweisen robust einlesen
    - Dateien mit Leerzeichen, Tab oder Semikolon als Trenner verarbeiten
    - Dezimalkomma automatisch in Dezimalpunkt umwandeln
    - Header- oder Textzeilen automatisch ignorieren

    Erwartet:
    - drei numerische Spalten: x, y, z

    Rückgabe:
    - NumPy-Array der Form (N, 3)
    """
    # Dateiinhalt als Text lesen
    txt = path.read_text(encoding="utf-8", errors="ignore")

    # Deutsches Dezimalkomma in Punkt umwandeln
    txt = txt.replace(",", ".")

    # Semikolon ebenfalls als Trennzeichen akzeptieren
    txt = txt.replace(";", " ")

    # Leere Zeilen entfernen
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]

    # Nur Zeilen behalten, deren erste drei Spalten als floats gelesen werden können
    data_lines = []
    for ln in lines:
        parts = ln.split()

        # Zeilen mit weniger als drei Einträgen ignorieren
        if len(parts) < 3:
            continue

        try:
            float(parts[0])
            float(parts[1])
            float(parts[2])
            data_lines.append(ln)
        except Exception:
            # Header- oder Textzeilen werden übersprungen
            continue

    if not data_lines:
        raise RuntimeError(f"Keine Datenzeilen gefunden in {path}")

    # Gültige Datenzeilen in ein NumPy-Array laden
    arr = np.loadtxt(data_lines)

    # Falls nur eine einzige Zeile vorliegt, auf 2D-Form bringen
    if arr.ndim == 1:
        arr = arr[None, :]

    # Falls mehr als drei Spalten vorhanden sind, nur die ersten drei verwenden
    if arr.shape[1] > 3:
        arr = arr[:, :3]

    if arr.shape[1] != 3:
        raise RuntimeError(f"Erwarte 3 Spalten (x y z), got shape {arr.shape}")

    return arr.astype(float)


def infer_step(vals: np.ndarray) -> float:
    """
    Schätzt die Schrittweite aus sortierten Unique-Werten.

    Was kann man damit machen?
    - Die typische Rasterauflösung eines Koordinatenvektors bestimmen
    - Automatisch eine sinnvolle Grid-Schrittweite für Interpolation ableiten
    - Aus leicht verrauschten Messkoordinaten eine robuste Schrittweite gewinnen

    Vorgehen:
    - Unique-Werte bilden
    - Positive Differenzen berechnen
    - Median der Differenzen als Schrittweite verwenden

    Rückgabe:
    - Geschätzte Schrittweite als float
    """
    u = np.unique(vals)

    if u.size < 2:
        return 1.0

    d = np.diff(u)
    d = d[d > 0]

    if d.size == 0:
        return 1.0

    return float(np.median(d))


def xyz_to_surface(xyz: np.ndarray) -> Surface:
    """
    Baut aus x-, y- und z-Daten eine Surfalize-Oberfläche.

    Was kann man damit machen?
    - Punktdaten in eine 2D-Oberfläche überführen
    - Regelmäßige Raster direkt übernehmen
    - Unregelmäßige Messpunkte auf ein regelmäßiges Grid interpolieren

    Ablauf:
    1. x, y, z optional skalieren
    2. Prüfen, ob ein vollständiges regelmäßiges Raster vorliegt
    3. Bei regelmäßigem Raster direkt ein Grid aufbauen
    4. Sonst per griddata auf ein regelmäßiges Raster interpolieren

    Rückgabe:
    - Surface-Objekt für Surfalize
    """
    # Skaliere Koordinaten und Höhenwerte
    x = xyz[:, 0] * SCALE_XY
    y = xyz[:, 1] * SCALE_XY
    z = xyz[:, 2] * SCALE_Z

    # Auf feste Genauigkeit runden, damit fast gleiche Koordinaten
    # sicher als gleiche Rasterpunkte erkannt werden
    xr = np.round(x, 12)
    yr = np.round(y, 12)

    # Eindeutige x- und y-Werte bestimmen
    ux = np.unique(xr)
    uy = np.unique(yr)

    # Erwartete und tatsächliche Punktanzahl
    n_expected = ux.size * uy.size
    n_actual = xr.size

    # Unique-Werte sortieren
    ux.sort()
    uy.sort()

    # Typische Schrittweiten abschätzen
    step_x = infer_step(ux)
    step_y = infer_step(uy)

    # --------------------------------------------------------
    # Fall 1: perfektes regelmäßiges Raster
    # --------------------------------------------------------
    if n_expected == n_actual:
        # Zuordnung von Koordinate -> Index im Raster
        x_to_ix = {v: i for i, v in enumerate(ux)}
        y_to_iy = {v: i for i, v in enumerate(uy)}

        # Leeres Raster anlegen
        grid = np.full((uy.size, ux.size), np.nan, dtype=float)

        # Messwerte an die passende Rasterposition schreiben
        for xi, yi, zi in zip(xr, yr, z):
            ix = x_to_ix[xi]
            iy = y_to_iy[yi]
            grid[iy, ix] = zi

        # Falls die y-Richtung invertiert werden soll:
        # grid = np.flipud(grid)

        surf = Surface(grid, step_x, step_y)
        return surf

    # --------------------------------------------------------
    # Fall 2: unregelmäßige Punkte -> Interpolation auf Grid
    # --------------------------------------------------------
    if GRID_STEP_MODE == "auto":
        gx = step_x
        gy = step_y
    else:
        gx = float(GRID_STEP_MODE)
        gy = float(GRID_STEP_MODE)

    # Grenzen des Koordinatenbereichs bestimmen
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    # Zielraster aufbauen
    xg = np.arange(x_min, x_max + 0.5 * gx, gx)
    yg = np.arange(y_min, y_max + 0.5 * gy, gy)
    XG, YG = np.meshgrid(xg, yg)

    # Interpolationsdaten vorbereiten
    points = np.column_stack([x, y])
    values = z

    # Lineare Interpolation auf das Zielgrid
    grid = griddata(points, values, (XG, YG), method="linear")

    # Randbereiche, die linear nicht gefüllt wurden,
    # mit nearest neighbor ergänzen
    if np.isnan(grid).any():
        grid_nn = griddata(points, values, (XG, YG), method="nearest")
        grid = np.where(np.isnan(grid), grid_nn, grid)

    surf = Surface(grid, gx, gy)
    return surf


def main():
    """
    Hauptfunktion zum Einlesen, Umwandeln und Speichern.

    Was kann man damit machen?
    - Eine Keyence-ASCII-Datei direkt in eine SDF-Datei umwandeln
    - Größe und Schrittweite der erzeugten Oberfläche kontrollieren
    - Direkt prüfen, ob die gespeicherte SDF-Datei korrekt wieder geladen werden kann
    """
    # ASCII-Datei einlesen
    xyz = read_asc_xyz(ASC_PATH)

    # Punktdaten in eine Surface umwandeln
    surf = xyz_to_surface(xyz)

    print("Loaded ASCII -> Surface")
    print(f"  shape px: {surf.data.shape}")
    print(f"  step_x, step_y: {surf.step_x:.6g}, {surf.step_y:.6g}")
    print(f"  size (units of x/y): {surf.width_um:.2f} x {surf.height_um:.2f} (Surfalize nennt es width_um/height_um)")

    # Surface als binäre SDF-Datei speichern
    surf.save(str(OUT_SDF), format="sdf", binary=True)
    print(f"Saved: {OUT_SDF}")

    # Test: SDF-Datei erneut laden
    surf2 = Surface.load(str(OUT_SDF))
    print("Reload OK")
    print(f"  shape px: {surf2.data.shape}, step: {surf2.step_x:.6g},{surf2.step_y:.6g}")


if __name__ == "__main__":
    main()