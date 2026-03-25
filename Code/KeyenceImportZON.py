from pathlib import Path
from SurfaceTopography import read_topography
from surfalize import Surface
import numpy as np


def zon_to_sdf_folder(
    folder_in: str,
    folder_out: str,
    pattern: str = "*.zon",
    overwrite: bool = False
):
    """
    Wandelt alle .zon-Dateien eines Ordners in .sdf-Dateien um.

    Was kann man damit machen?
    - Keyence- oder andere .zon-Dateien gesammelt einlesen
    - Die Höhenmatrix in ein surfalize-kompatibles Surface-Format überführen
    - Mehrere Dateien in einem Schritt als .sdf abspeichern
    - Dabei dieselbe Korrekturlogik für Orientierung und Schrittweite verwenden

    Parameter:
    - folder_in:
      Eingabeordner mit den .zon-Dateien
    - folder_out:
      Ausgabeordner für die erzeugten .sdf-Dateien
    - pattern:
      Suchmuster für die Eingabedateien
    - overwrite:
      Falls False, werden vorhandene .sdf-Dateien nicht überschrieben
    """
    folder_in = Path(folder_in)
    folder_out = Path(folder_out)

    # Ausgabeordner anlegen, falls er noch nicht existiert
    folder_out.mkdir(parents=True, exist_ok=True)

    # Alle passenden .zon-Dateien suchen
    files = sorted(folder_in.glob(pattern))
    if not files:
        print(f"Keine .zon gefunden in: {folder_in}")
        return

    # Jede Datei einzeln verarbeiten
    for f in files:
        out_path = folder_out / (f.stem + ".sdf")

        # Bereits vorhandene Dateien optional überspringen
        if out_path.exists() and not overwrite:
            print(f"SKIP (exists): {out_path.name}")
            continue

        # ----------------------------------------------------
        # 1) .zon-Datei mit SurfaceTopography einlesen
        # ----------------------------------------------------
        t = read_topography(str(f))

        # Höhenwerte laden
        z = t.heights()

        # Physikalische Größen des Messfelds in Metern
        sx, sy = t.physical_sizes

        # ----------------------------------------------------
        # 2) Einheiten von m nach µm umrechnen
        # ----------------------------------------------------
        sx_um = float(sx) * 1e6
        sy_um = float(sy) * 1e6
        z_um = np.asarray(z, dtype=float) * 1e6

        # ----------------------------------------------------
        # 3) Orientierungskorrektur
        #    transpose + vertikal spiegeln
        #    so wie in deinem bisherigen Keyence-Workflow
        # ----------------------------------------------------
        z_corr = np.flipud(z_um.T)

        # Zeilen- und Spaltenanzahl der korrigierten Matrix
        rows, cols = z_corr.shape

        # ----------------------------------------------------
        # 4) Schrittweiten in x und y aus Feldgröße ableiten
        # ----------------------------------------------------
        dx_um = sx_um / (cols - 1)
        dy_um = sy_um / (rows - 1)

        # ----------------------------------------------------
        # 5) surfalize Surface erzeugen
        #    mit zusätzlichem flipud wie in deinem bisherigen Ablauf
        # ----------------------------------------------------
        s = Surface(np.flipud(z_corr), dx_um, dy_um)

        # ----------------------------------------------------
        # 6) Als .sdf speichern
        # ----------------------------------------------------
        s.save(str(out_path))

        print(
            f"OK: {f.name}  ->  {out_path.name}   "
            f"(dx={dx_um:.3f} µm, dy={dy_um:.3f} µm)"
        )


if __name__ == "__main__":
    zon_to_sdf_folder(
        folder_in="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP00/1x7_Reflektion",
        folder_out="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP00/1x7_Reflektion/Ergebnisse",
        pattern="WSP00_L*_*.zon",   # nimmt L1/L13/L14/L44 usw.
        overwrite=False
    )