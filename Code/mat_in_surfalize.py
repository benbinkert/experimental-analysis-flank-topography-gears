# mat_in_surfalize.py
# ------------------------------------------------------------
# Exportiert Flanken-Meshes aus WST_TOPO-*.mat nach Surfalize-SDF.
#
# Was kann man damit machen?
# - Simulations- oder MAT-Dateien mit Flankenmesh einlesen
# - Rechte und linke Flanke getrennt als SDF speichern
# - Sowohl klassische .mat-Dateien als auch MATLAB-v7.3-Dateien verarbeiten
# - Orientierung der Flanken beim Export gezielt anpassen
#
# Unterstützt:
# - Classic MAT über scipy.io.loadmat
# - MATLAB v7.3 / HDF5 über h5py
#   A) Reference-Datasets der Form (N,1)
#   B) direkt numerische 2D-Datasets
#   C) optional numerische 3D-Datasets mit Slice [idx0,:,:]
#
# Exportiert:
# - <base>_Rechts.sdf
# - <base>_Links.sdf
#
# Annahmen:
# - X = Radius-Mesh in mm
# - Y = Breite-/Axial-Mesh in mm
# - Z = Abweichung in mm
# - Für Surfalize erfolgt die Umrechnung in µm
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import h5py

from surfalize import Surface


# ============================================================
# Optionaler Crop-Helfer
# Damit kann das Mesh auf den Bereich reduziert werden, in dem
# tatsächlich gültige Z-Werte vorhanden sind
# ============================================================

def crop_to_valid_region(X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    """
    Cropt X-, Y- und Z-Mesh auf den kleinsten Bereich mit gültigen Z-Werten.

    Was kann man damit machen?
    - Leere Randbereiche mit nur NaN-Werten entfernen
    - Die exportierte Fläche auf den real gemessenen oder simulierten Bereich beschränken
    - Nachgelagerte Verarbeitung robuster machen

    Rückgabe:
    - Gecroppte Arrays X, Y, Z
    """
    mask = np.isfinite(Z)

    if not np.any(mask):
        return X, Y, Z

    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]

    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1

    return X[r0:r1, c0:c1], Y[r0:r1, c0:c1], Z[r0:r1, c0:c1]


# ============================================================
# Debug-Helfer für HDF5-Struktur
# Damit kann man sich den inneren Aufbau einer MATLAB-v7.3-Datei
# ansehen, wenn unklar ist, wo die Meshdaten liegen
# ============================================================

def print_hdf5_structure(path: str | Path):
    """
    Gibt die Struktur einer HDF5-/MAT-v7.3-Datei rekursiv aus.

    Was kann man damit machen?
    - Dataset- und Gruppenstruktur einer Datei inspizieren
    - Prüfen, ob Daten als Referenzen oder numerische Arrays vorliegen
    - Pfade und Shapes vor dem eigentlichen Export kontrollieren
    """
    path = str(path)

    def visitor(name, obj):
        t = type(obj).__name__
        shp = getattr(obj, "shape", "")
        dt = getattr(obj, "dtype", "")
        print(f"{name} -> {t} shape={shp} dtype={dt}")

    with h5py.File(path, "r") as f:
        f.visititems(visitor)


# ============================================================
# Mesh -> Surface
# Kernfunktion zur Umrechnung von X/Y/Z-Meshes in eine
# Surfalize-Surface mit Schrittweiten in µm
# ============================================================

def _surface_from_mesh_large(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    show_plot: bool = True,
    title: str = "",
    do_crop: bool = True,
    transform: str = "flipud",
):
    """
    Erzeugt ein Surfalize-Surface aus X-, Y- und Z-Meshes.

    Was kann man damit machen?
    - Ein Flankenmesh direkt in ein Surface-Objekt umwandeln
    - Die Schrittweiten automatisch aus dem Mesh bestimmen
    - Die Orientierung des Z-Bildes für den Export anpassen
    - Vor dem Export eine Plausibilitätsvisualisierung anzeigen

    Annahmen:
    - X, Y, Z liegen in mm vor
    - Schrittweiten und Z-Werte werden für Surfalize in µm umgerechnet

    Parameter:
    - show_plot:
      Zeigt das Mesh vor dem Export als pcolormesh
    - do_crop:
      Cropt vor der Umwandlung auf den gültigen Bereich
    - transform:
      Steuert die Orientierung von Z im Export
      Optionen: "none", "flipud", "fliplr", "flipud_fliplr"

    Rückgabe:
    - Surface-Objekt
    """
    z_valid = Z[np.isfinite(Z)]
    if z_valid.size:
        print(f"{title} | Z mm: {np.min(z_valid)} ... {np.max(z_valid)}")
    else:
        print(f"{title} | Z mm: (keine finite Werte)")

    print(f"{title} | Shapes: X{X.shape} Y{Y.shape} Z{Z.shape}")

    # Optionaler Diagnoseplot
    if show_plot:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        pcm = ax.pcolormesh(Y, X, Z, shading="auto", cmap="jet")
        ax.set_xlim(np.nanmin(Y), np.nanmax(Y))
        ax.set_ylim(np.nanmin(X), np.nanmax(X))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Werkstückbreite in mm")
        ax.set_ylabel("Werkstückradius in mm")
        ax.set_title(title)
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Abweichung in mm")
        plt.tight_layout()
        plt.show(block=False)

    # Optionaler Crop auf den gültigen Bereich
    if do_crop:
        Xc, Yc, Zc = crop_to_valid_region(X, Y, Z)
    else:
        Xc, Yc, Zc = X, Y, Z

    # Schrittweiten robust aus dem Mesh bestimmen
    # Y entspricht der x-Richtung
    # X entspricht der y-Richtung
    step_x_new_mm = float(np.nanmedian(np.abs(np.diff(Yc[0, :]))))
    step_y_new_mm = float(np.nanmedian(np.abs(np.diff(Xc[:, 0]))))

    step_x_new_um = 1000.0 * step_x_new_mm
    step_y_new_um = 1000.0 * step_y_new_mm

    # Z in µm umrechnen und Orientierung anpassen
    if transform == "none":
        Z_new_um = 1000.0 * Zc
    elif transform == "flipud":
        Z_new_um = 1000.0 * np.flipud(Zc)
    elif transform == "fliplr":
        Z_new_um = 1000.0 * np.fliplr(Zc)
    elif transform == "flipud_fliplr":
        Z_new_um = 1000.0 * np.flipud(np.fliplr(Zc))
    else:
        raise ValueError(f"Unbekannte transform-Option: {transform}")

    # Surface erzeugen
    surf = Surface(Z_new_um, step_x_new_um, step_y_new_um)
    print(f"{title} | Surface: {surf.data.shape}, step_x={surf.step_x:.6f} µm, step_y={surf.step_y:.6f} µm")
    return surf


# ============================================================
# MAT-v7.3-/HDF5-Helfer
# Damit lassen sich MATLAB-v7.3-Dateien sicher erkennen und
# Referenz- oder numerische Datasets robust lesen
# ============================================================

def is_mat_v73(mat_path: str | Path) -> bool:
    """
    Prüft, ob eine MAT-Datei als HDF5 / MATLAB v7.3 gelesen werden kann.

    Was kann man damit machen?
    - Automatisch zwischen klassischer MAT und v7.3 unterscheiden
    - Den passenden Leseweg wählen
    """
    try:
        with h5py.File(mat_path, "r"):
            return True
    except OSError:
        return False


def _read_scalar_h5(f: h5py.File, path: str) -> float:
    """
    Liest einen skalaren Wert aus einer HDF5-Datei.

    Was kann man damit machen?
    - Einzelwerte wie z_topo auslesen
    - MATLAB-Parameter als Python-float übernehmen
    """
    arr = np.array(f[path][()]).squeeze()
    return float(arr)


def _is_ref_dtype(d: h5py.Dataset) -> bool:
    """
    Prüft, ob ein HDF5-Dataset Referenzen statt numerischer Daten enthält.

    Was kann man damit machen?
    - Zwischen Reference-Containern und direkten numerischen Datasets unterscheiden
    - Den passenden Ausleseweg wählen
    """
    return (d.dtype == object) or ("ref" in str(d.dtype).lower())


def _deref_one(f: h5py.File, ref):
    """
    Dereferenziert genau eine HDF5-Referenz.

    Was kann man damit machen?
    - Aus einem Referenz-Container auf das eigentliche Ziel-Dataset zugreifen
    - Unterschiedliche Ref-Formate robust behandeln
    """
    if isinstance(ref, np.ndarray):
        ref = ref.squeeze()
        if ref.shape != ():
            ref = ref.flat[0]
    return f[ref]


def _read_mesh_any(f: h5py.File, dset: h5py.Dataset, idx0: int | None = None) -> np.ndarray:
    """
    Liest Mesh-Daten aus mehreren möglichen Dataset-Layouts.

    Was kann man damit machen?
    - MAT-v7.3-Dateien robust verarbeiten, auch wenn das interne Layout variiert
    - Referenz-Container, 2D- und 3D-Datasets einheitlich behandeln

    Unterstützte Fälle:
    - (N,1) Reference-Dataset -> dereferenzieren
    - 3D numeric (N,Ny,Nx) -> Slice [idx0,:,:]
    - 2D numeric (Ny,Nx) -> direkt
    - 1D numeric -> direkt

    Rückgabe:
    - Numerisches NumPy-Array
    """
    if isinstance(dset, h5py.Group):
        raise TypeError("Dataset erwartet, Group bekommen.")

    # Fall 1: Referenz-Container der Form (N,1)
    if _is_ref_dtype(dset) and dset.ndim == 2 and dset.shape[1] == 1:
        if idx0 is None:
            raise ValueError("idx0 nötig für (N,1) Reference-Dataset.")
        ref = dset[idx0, 0]
        obj = _deref_one(f, ref)
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"Referenz zeigt nicht auf Dataset, sondern: {type(obj)}")
        return np.array(obj[()], dtype=float)

    # Fall 2: 3D numerisches Dataset
    if (not _is_ref_dtype(dset)) and dset.ndim == 3:
        if idx0 is None:
            raise ValueError("idx0 nötig für 3D Dataset.")
        return np.array(dset[idx0, :, :], dtype=float)

    # Fall 3: 2D numerisches Dataset
    if (not _is_ref_dtype(dset)) and dset.ndim == 2:
        return np.array(dset[()], dtype=float)

    # Fall 4: 1D numerisches Dataset
    if (not _is_ref_dtype(dset)) and dset.ndim == 1:
        return np.array(dset[()], dtype=float)

    raise ValueError(f"Unbekanntes Dataset-Layout: shape={dset.shape}, dtype={dset.dtype}")


# ============================================================
# Export für MATLAB v7.3 / HDF5
# Liest beide Flanken robust aus einer HDF5-basierten MAT-Datei
# und speichert sie als SDF
# ============================================================

def export_flanks_from_mat_large(
    mat_path: str,
    output_dir: str | None = None,
    show_plots: bool = True,
    show_surfaces: bool = False,
    do_crop: bool = True,
    transform_right: str = "flipud",
    transform_left: str = "none",
):
    """
    Exportiert rechte und linke Flanke aus einer MATLAB-v7.3-Datei.

    Was kann man damit machen?
    - HDF5-basierte MAT-Dateien mit WST_TOPO-Struktur verarbeiten
    - Den gewünschten z_topo-Eintrag auswählen
    - Beide Flanken getrennt als SDF speichern

    Rückgabe:
    - Tuple mit Pfaden zu rechter und linker SDF-Datei
    """
    mat_path = str(mat_path)

    if output_dir is None:
        output_dir = os.path.dirname(mat_path)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(mat_path))[0]

    with h5py.File(mat_path, "r") as f:
        # MATLAB ist 1-basiert, Python 0-basiert
        z_topo = int(_read_scalar_h5(f, "WST_TOPO/Par/z_topo"))
        idx0 = z_topo - 1
        print(f"z_topo = {z_topo}")

        ist = f["WST_TOPO/Ist"]

        # Rechte Flanke
        Xr = _read_mesh_any(f, ist["r_r_mesh"], idx0=idx0)
        Yr = _read_mesh_any(f, ist["z_r_mesh"], idx0=idx0)
        Zr = _read_mesh_any(f, ist["Abw_r_mesh"], idx0=idx0)

        # Linke Flanke
        Xl = _read_mesh_any(f, ist["r_l_mesh"], idx0=idx0)
        Yl = _read_mesh_any(f, ist["z_l_mesh"], idx0=idx0)
        Zl = _read_mesh_any(f, ist["Abw_l_mesh"], idx0=idx0)

    print(f"[Right] Xr:{Xr.shape} Yr:{Yr.shape} Zr:{Zr.shape}")
    print(f"[Left ] Xl:{Xl.shape} Yl:{Yl.shape} Zl:{Zl.shape}")

    # Rechte Flanke in Surface umwandeln und speichern
    surf_r = _surface_from_mesh_large(
        Xr, Yr, Zr,
        show_plot=show_plots,
        title=f"{base_name} - Rechte Flanke",
        do_crop=do_crop,
        transform=transform_right,
    )
    right_path = os.path.join(output_dir, f"{base_name}_Rechts.sdf")
    surf_r.save(right_path)
    if show_surfaces:
        surf_r.show()

    # Linke Flanke in Surface umwandeln und speichern
    surf_l = _surface_from_mesh_large(
        Xl, Yl, Zl,
        show_plot=show_plots,
        title=f"{base_name} - Linke Flanke",
        do_crop=do_crop,
        transform=transform_left,
    )
    left_path = os.path.join(output_dir, f"{base_name}_Links.sdf")
    surf_l.save(left_path)
    if show_surfaces:
        surf_l.show()

    print(f"Gespeichert:\n  {right_path}\n  {left_path}")
    return right_path, left_path


# ============================================================
# Export für klassische MAT-Dateien
# Liest die WST_TOPO-Struktur über scipy.io.loadmat aus
# ============================================================

def export_flanks_from_mat(
    mat_path: str,
    output_dir: str | None = None,
    show_plots: bool = True,
    show_surfaces: bool = False,
    do_crop: bool = True,
    transform_right: str = "flipud",
    transform_left: str = "none",
):
    """
    Exportiert rechte und linke Flanke aus einer klassischen MAT-Datei.

    Was kann man damit machen?
    - Nicht-HDF5-MAT-Dateien direkt verarbeiten
    - Den gewünschten z_topo-Eintrag auswählen
    - Beide Flanken getrennt als SDF speichern

    Rückgabe:
    - Tuple mit Pfaden zu rechter und linker SDF-Datei
    """
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    WST = mat["WST_TOPO"]

    # MATLAB 1-basiert
    z_target = int(WST.Par.z_topo)
    Ist = np.atleast_1d(WST.Ist)[z_target - 1]

    if output_dir is None:
        output_dir = os.path.dirname(mat_path)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(mat_path))[0]

    # Rechte Flanke
    Xr = np.array(Ist.r_r_mesh, dtype=float)
    Yr = np.array(Ist.z_r_mesh, dtype=float)
    Zr = np.array(Ist.Abw_r_mesh, dtype=float)

    # Linke Flanke
    Xl = np.array(Ist.r_l_mesh, dtype=float)
    Yl = np.array(Ist.z_l_mesh, dtype=float)
    Zl = np.array(Ist.Abw_l_mesh, dtype=float)

    surf_r = _surface_from_mesh_large(
        Xr, Yr, Zr,
        show_plot=show_plots,
        title=f"{base_name} - Rechte Flanke",
        do_crop=do_crop,
        transform=transform_right,
    )
    right_path = os.path.join(output_dir, f"{base_name}_Rechts.sdf")
    surf_r.save(right_path)
    if show_surfaces:
        surf_r.show()

    surf_l = _surface_from_mesh_large(
        Xl, Yl, Zl,
        show_plot=show_plots,
        title=f"{base_name} - Linke Flanke",
        do_crop=do_crop,
        transform=transform_left,
    )
    left_path = os.path.join(output_dir, f"{base_name}_Links.sdf")
    surf_l.save(left_path)
    if show_surfaces:
        surf_l.show()

    print(f"Gespeichert:\n  {right_path}\n  {left_path}")
    return right_path, left_path


# ============================================================
# Automatische Wahl des passenden MAT-Lesewegs
# ============================================================

def export_flanks_from_mat_auto(
    mat_path: str,
    output_dir: str | None = None,
    show_plots: bool = True,
    show_surfaces: bool = False,
    do_crop: bool = True,
    transform_right: str = "flipud",
    transform_left: str = "none",
):
    """
    Wählt automatisch den passenden Leseweg für eine MAT-Datei.

    Was kann man damit machen?
    - Ohne manuelle Prüfung direkt klassische oder v7.3-MAT-Dateien verarbeiten
    - Den Export über eine einzige Einstiegfunktion starten
    """
    if is_mat_v73(mat_path):
        print("MATLAB v7.3 erkannt -> benutze HDF5/h5py")
        return export_flanks_from_mat_large(
            mat_path,
            output_dir=output_dir,
            show_plots=show_plots,
            show_surfaces=show_surfaces,
            do_crop=do_crop,
            transform_right=transform_right,
            transform_left=transform_left,
        )
    else:
        print("Klassische .mat-Datei erkannt -> benutze scipy.io.loadmat")
        return export_flanks_from_mat(
            mat_path,
            output_dir=output_dir,
            show_plots=show_plots,
            show_surfaces=show_surfaces,
            do_crop=do_crop,
            transform_right=transform_right,
            transform_left=transform_left,
        )


# ============================================================
# Hauptfunktion
# Startet den Export für eine konkrete Datei
# ============================================================

def mat_in_surfalize():
    """
    Startet den MAT-zu-SDF-Export für eine konkrete Datei.

    Was kann man damit machen?
    - Eine Simulationsdatei direkt in zwei Flanken-SDF-Dateien umwandeln
    - Optional Diagnoseplots und Surface-Anzeige aktivieren
    """
    # Optional vorher Struktur ausgeben, wenn die Datei unbekannt ist
    # print_hdf5_structure("/path/to/your.mat")

    export_flanks_from_mat_auto(
        "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Simulation/WSP00/WST_WSP00_L1_HighResolution_2.mat",
        output_dir="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Simulation/WSP00",
        show_plots=True,
        show_surfaces=True,
        do_crop=True,
        transform_right="flipud",
        transform_left="none",
    )


if __name__ == "__main__":
    mat_in_surfalize()