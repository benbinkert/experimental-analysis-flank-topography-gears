from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import csv
import numpy as np
from scipy.ndimage import gaussian_filter1d


# ============================================================
# Auswertung von Welligkeitsprofilen aus Perthometer- und KMG-Daten
#
# Was kann man damit machen?
# - Perthometer-Profile aus Textdateien einlesen
# - KMG-Profile aus exportierten CSV-Dateien einlesen
# - Aus allen KMG-Unterordnern rekursiv alle Lead-Profile finden
# - Welligkeitsprofile mit einer Grenzwellenlänge λc filtern
# - Welligkeitskenngrößen Wa, Wq, Wp, Wv, Wz, Wsk und Wku berechnen
# - Ergebnisse für Perthometer und KMG in einer gemeinsamen CSV speichern
#
# Hinweis:
# - Für Perthometer werden alle Wiederholungen pro Flanke verarbeitet
# - Für KMG werden alle Lead-Dateien verarbeitet
# - Für KMG werden sowohl das dev-Signal als auch das z-Signal ausgewertet
# ============================================================


# ============================================================
# Datencontainer für 1D-Profile
# ============================================================

@dataclass
class Profile1D:
    """
    Ein einfacher Container für ein 1D-Profil.

    Attribute:
    - x_um:
      Profilkoordinate in µm
    - z_um:
      Höhen- oder Abweichungssignal in µm
    - label:
      Freie Bezeichnung des Profils
    """
    x_um: np.ndarray
    z_um: np.ndarray
    label: str = ""

    @property
    def dx_um(self) -> float:
        """
        Mittlere Schrittweite des Profils in µm.
        """
        if len(self.x_um) < 2:
            return float("nan")
        return float(np.median(np.diff(self.x_um)))

    @property
    def length_um(self) -> float:
        """
        Gesamtlänge des Profils in µm.
        """
        if len(self.x_um) < 2:
            return 0.0
        return float(self.x_um[-1] - self.x_um[0])


# ============================================================
# Perthometer import
# ============================================================

def load_perthometer_profile(path_txt: str) -> Profile1D:
    """
    Liest ein Perthometer-Profil aus einer Textdatei ein.

    Erwartet:
    - einen Block [PROFILE_VALUES]
    - Datenzeilen mit mindestens x und z

    Umrechnung:
    - x von mm nach µm
    - z von mm nach µm

    Rückgabe:
    - Profile1D-Objekt
    """
    path = Path(path_txt)
    data_started = False
    xs_mm: list[float] = []
    zs_mm: list[float] = []

    with path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # Ab hier beginnen die Profilwerte
            if line.startswith("[PROFILE_VALUES]"):
                data_started = True
                continue

            # Vor dem Datenblock wird alles ignoriert
            if not data_started:
                continue

            # Leere Zeilen und Kommentarzeilen überspringen
            if not line or line.startswith("//"):
                continue

            # Nur Zeilen mit "=" verarbeiten
            if "=" not in line:
                continue

            _, right = line.split("=", 1)
            parts = right.split()

            # Es werden mindestens x und z erwartet
            if len(parts) < 3:
                continue

            x = float(parts[0])
            z = float(parts[2])
            xs_mm.append(x)
            zs_mm.append(z)

    if len(xs_mm) < 2:
        raise ValueError(f"Zu wenige Punkte in Perthometer Datei: {path_txt}")

    x_mm = np.asarray(xs_mm, dtype=float)
    z_mm = np.asarray(zs_mm, dtype=float)

    # x-Achse auf 0 setzen und nach µm umrechnen
    x_um = (x_mm - x_mm[0]) * 1000.0

    # z nach µm umrechnen
    z_um = z_mm * 1000.0

    return Profile1D(x_um=x_um, z_um=z_um, label=path.stem)


def load_perthometer_repetitions(folder: str, base_id: str) -> list[Profile1D]:
    """
    Sucht in einem Ordner nach bis zu drei Wiederholungsmessungen
    für eine gegebene Flanke.

    Beispiel:
    - base_id = WSP00_L13_S1
    - sucht nach:
      WSP00_L13_S1_1.*
      WSP00_L13_S1_2.*
      WSP00_L13_S1_3.*

    Rückgabe:
    - Liste aus Profile1D-Objekten
    """
    folder_p = Path(folder)
    reps: list[Profile1D] = []

    for k in (1, 2, 3):
        candidates = sorted(folder_p.glob(f"{base_id}_{k}.*"))
        if not candidates:
            continue
        reps.append(load_perthometer_profile(str(candidates[0])))

    if not reps:
        raise FileNotFoundError(
            f"Keine Perthometer Wiederholungen gefunden für {base_id} in {folder_p}"
        )

    return reps


def find_wsp00_perthometer_base_ids(perth_dir: str) -> list[str]:
    """
    Durchsucht einen Perthometer-Ordner nach Basis-IDs vom Typ:
    WSP00_L*_S1 oder WSP00_L*_S2

    Beispiel:
    - WSP00_L13_S1_1.txt -> Basis-ID = WSP00_L13_S1

    Rückgabe:
    - sortierte Liste aller gefundenen Basis-IDs
    """
    folder = Path(perth_dir)
    patt = re.compile(r"^(WSP00_L\d+_S[12])_(\d+)$")

    base_ids = set()
    for p in folder.iterdir():
        if not p.is_file():
            continue

        m = patt.match(p.stem)
        if m:
            base_ids.add(m.group(1))

    return sorted(base_ids)


# ============================================================
# KMG import
# ============================================================

def _read_kmg_csv_with_meta(path: Path) -> tuple[list[str], list[list[str]], dict]:
    """
    Liest eine KMG-CSV mit Metazeilen und Datenbereich ein.

    Erwartet:
    - Metazeilen beginnen mit '#'
    - danach eine Header-Zeile
    - danach Datenzeilen

    Rückgabe:
    - header: Spaltennamen
    - data_rows: Datenzeilen als Stringlisten
    - meta: Dictionary mit Metainformationen
    """
    meta: dict[str, str] = {}
    header: list[str] = []
    data_rows: list[list[str]] = []
    in_data = False

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Metazeilen auslesen
            if line.startswith("#"):
                m = re.match(r"^#\s*([^:]+)\s*:\s*(.*)$", line)
                if m:
                    meta[m.group(1).strip().lower()] = m.group(2).strip()
                continue

            parts = [p.strip() for p in line.split(",")]

            # Erste Nicht-Metazeile ist der Header
            if not in_data:
                header = parts
                in_data = True
            else:
                if len(parts) >= len(header):
                    data_rows.append(parts)

    if not header:
        raise ValueError(f"Kein CSV-Header gefunden: {path}")
    if not data_rows:
        raise ValueError(f"Keine CSV-Datenzeilen gefunden: {path}")

    return header, data_rows, meta


def load_kmg_block_csv(path_csv: str, channel: str = "dev") -> tuple[Profile1D, dict]:
    """
    Liest ein einzelnes KMG-Profil aus einer CSV-Datei.

    channel:
    - "dev": verwendet die Spalte dev_um bzw. dev
    - "z":   verwendet die Spalte z_um bzw. z

    Rückgabe:
    - Profile1D
    - info-Dictionary mit Metadaten
    """
    path = Path(path_csv)
    header, rows, meta = _read_kmg_csv_with_meta(path)

    col = {name: i for i, name in enumerate(header)}

    if "s_um" not in col:
        raise ValueError(f"Spalte 's_um' fehlt: {path}")

    dev_key = "dev_um" if "dev_um" in col else ("dev" if "dev" in col else None)
    if dev_key is None:
        raise ValueError(f"Spalte 'dev_um'/'dev' fehlt: {path}")

    z_key = "z_um" if "z_um" in col else ("z" if "z" in col else None)
    if z_key is None:
        raise ValueError(f"Spalte 'z_um'/'z' fehlt: {path}")

    s_vals: list[float] = []
    y_vals: list[float] = []

    for r in rows:
        try:
            s = float(r[col["s_um"]])

            # Je nach gewünschtem Signal dev oder z verwenden
            yy = float(r[col[z_key]]) if channel.lower() == "z" else float(r[col[dev_key]])
        except Exception:
            continue

        s_vals.append(s)
        y_vals.append(yy)

    if len(s_vals) < 2:
        raise ValueError(f"Zu wenige gültige Punkte in: {path}")

    prof = Profile1D(
        x_um=np.asarray(s_vals, dtype=float),
        z_um=np.asarray(y_vals, dtype=float),
        label=path.stem + f"_KMG_{channel.lower()}"
    )

    info = {
        "file": str(path),
        "filename": path.name,
        "prefix": meta.get("prefix", ""),
        "kind": meta.get("kind", ""),
        "no": meta.get("no", ""),
        "pos": meta.get("pos", ""),
        "side": meta.get("side", ""),
        "flank": meta.get("flank", ""),
        "id": meta.get("id", ""),
        "units": meta.get("units", ""),
        "channel": channel.lower(),
    }

    return prof, info


def load_all_kmg_lead_blocks(folder: str, channel: str = "dev") -> list[tuple[Profile1D, dict]]:
    """
    Sucht rekursiv in einem Ordner nach allen KMG-CSV-Dateien,
    liest sie ein und filtert nur Lead-Blöcke.

    Wichtig:
    - nutzt rglob("*.csv"), also auch Unterordner wie L1, L40 usw.
    - summary-Dateien werden übersprungen

    Rückgabe:
    - Liste aus (Profile1D, info)-Tupeln
    """
    folder_p = Path(folder)
    out: list[tuple[Profile1D, dict]] = []

    for p in sorted(folder_p.rglob("*.csv")):
        # Summary-Dateien nicht als Profile behandeln
        if p.name.lower().startswith("summary"):
            continue

        try:
            prof, info = load_kmg_block_csv(str(p), channel=channel)
        except Exception as e:
            print(f"[SKIP] {p.name}: {e}")
            continue

        # Nur Lead-Dateien weiterverwenden
        if (info.get("kind", "") or "").lower() != "lead":
            continue

        flank = info.get("flank", "")

        # Falls flank fehlt, aus side ableiten
        if flank not in ("S1", "S2"):
            side = (info.get("side", "") or "").lower()
            flank = "S1" if side == "right" else "S2"
            info["flank"] = flank

        if flank in ("S1", "S2"):
            out.append((prof, info))

    return out


# ============================================================
# Signalverarbeitung
# ============================================================

def level_profile(p: Profile1D, mode: str = "mean") -> Profile1D:
    """
    Zentriert ein Profil um seinen Mittelwert oder Median.

    mode:
    - "mean"
    - "median"
    """
    z = p.z_um.astype(float).copy()
    m = np.isfinite(z)

    if np.count_nonzero(m) == 0:
        return Profile1D(p.x_um.copy(), z, p.label + "_level(empty)")

    z0 = float(np.median(z[m])) if mode == "median" else float(np.mean(z[m]))
    return Profile1D(p.x_um.copy(), z - z0, p.label + f"_level({mode})")


def detrend_poly(p: Profile1D, degree: int = 2) -> Profile1D:
    """
    Entfernt einen polynomialen Trend aus einem Profil.

    degree:
    - Polynomgrad, standardmäßig 2
    """
    x = p.x_um.astype(float)
    z = p.z_um.astype(float)
    m = np.isfinite(z)

    if np.count_nonzero(m) < degree + 1:
        raise ValueError("Zu wenige gültige Punkte für Polynomfit")

    # x normieren für stabileren Polynomfit
    x0 = x[m].mean()
    xn = x - x0
    s = np.max(np.abs(xn[m])) + 1e-12
    xn = xn / s

    coeff = np.polyfit(xn[m], z[m], degree)
    trend = np.polyval(coeff, xn)

    return Profile1D(p.x_um.copy(), z - trend, p.label + f"_det{degree}")


def _sigma_px_from_cutoff_um(cutoff_um: float, dx_um: float) -> float:
    """
    Wandelt eine Grenzwellenlänge λc in eine Gauß-Sigma-Breite in Pixel um.
    """
    sigma_um = cutoff_um * np.sqrt(np.log(2.0)) / (2.0 * np.pi)
    sigma_px = sigma_um / (dx_um + 1e-12)
    return float(max(sigma_px, 0.5))


def gaussian_lowpass(p: Profile1D, cutoff_um: float, label_suffix: str = "") -> Profile1D:
    """
    Führt einen gaußschen Tiefpass auf ein Profil aus.

    Falls NaN-Werte vorhanden sind:
    - werden diese vorher linear interpoliert
    """
    dx = p.dx_um
    sigma_px = _sigma_px_from_cutoff_um(cutoff_um, dx)

    z = p.z_um.astype(float)

    if np.isnan(z).any():
        m = np.isfinite(z)
        z = z.copy()
        z[~m] = np.interp(p.x_um[~m], p.x_um[m], z[m])

    z_lp = gaussian_filter1d(z, sigma=sigma_px, mode="nearest")

    suf = label_suffix if label_suffix else f"_LP(lambda_c={cutoff_um:g}um)"
    return Profile1D(p.x_um.copy(), z_lp, p.label + suf)


def waviness_only(
    p: Profile1D,
    lambda_c_um: float = 800.0,
    do_level: bool = True,
    do_detrend: bool = False,
    detrend_degree: int = 2,
) -> Profile1D:
    """
    Berechnet aus einem Profil das Welligkeitsprofil.

    Schritte:
    - optional leveln
    - optional detrenden
    - anschließend Tiefpass mit λc
    """
    q = p

    if do_level:
        q = level_profile(q, mode="mean")

    if do_detrend:
        q = detrend_poly(q, degree=detrend_degree)

    return gaussian_lowpass(
        q,
        cutoff_um=lambda_c_um,
        label_suffix=f"_LP(lambda_c={lambda_c_um:g}um)"
    )


# ============================================================
# Kennwerte
# ============================================================

def centered_z(p: Profile1D) -> np.ndarray:
    """
    Zentriert das Profilsignal um seinen Mittelwert.
    """
    z = np.asarray(p.z_um, dtype=float)
    m = np.isfinite(z)

    if np.count_nonzero(m) == 0:
        return z.copy()

    zc = z.copy()
    zc[m] = z[m] - np.mean(z[m])
    return zc


def waviness_metrics_dict(p: Profile1D, prefix: str = "W") -> dict[str, float]:
    """
    Berechnet Welligkeitskenngrößen aus einem Profil.

    Kennwerte:
    - Wa
    - Wq
    - Wp
    - Wv
    - Wz
    - Wsk
    - Wku
    """
    z = centered_z(p)
    m = np.isfinite(z)
    z = z[m]

    if z.size == 0:
        return {
            f"{prefix}a_um": np.nan,
            f"{prefix}q_um": np.nan,
            f"{prefix}p_um": np.nan,
            f"{prefix}v_um": np.nan,
            f"{prefix}z_um": np.nan,
            f"{prefix}sk": np.nan,
            f"{prefix}ku": np.nan,
        }

    wa = float(np.mean(np.abs(z)))
    wq = float(np.sqrt(np.mean(z ** 2)))
    wp = float(np.max(z))
    wv = float(abs(np.min(z)))
    wz = wp + wv

    if wq > 0:
        wsk = float(np.mean(z ** 3) / (wq ** 3))
        wku = float(np.mean(z ** 4) / (wq ** 4))
    else:
        wsk = np.nan
        wku = np.nan

    return {
        f"{prefix}a_um": wa,
        f"{prefix}q_um": wq,
        f"{prefix}p_um": wp,
        f"{prefix}v_um": wv,
        f"{prefix}z_um": wz,
        f"{prefix}sk": wsk,
        f"{prefix}ku": wku,
    }


# ============================================================
# Perthometer-Auswertung
# ============================================================

def process_one_perthometer_flank(
    perth_dir: str,
    base_id: str,
    lambda_c_um: float = 800.0,
) -> list[dict]:
    """
    Verarbeitet alle Wiederholungen einer Perthometer-Flanke.

    Für jede Wiederholung:
    - einlesen
    - leveln
    - detrenden
    - Welligkeitsprofil berechnen
    - Welligkeitskenngrößen speichern

    Rückgabe:
    - Liste von Ergebniszeilen
    """
    perth_reps = load_perthometer_repetitions(perth_dir, base_id=base_id)
    rows: list[dict] = []

    for idx, prof in enumerate(perth_reps, start=1):
        perth_w = waviness_only(
            prof,
            lambda_c_um=lambda_c_um,
            do_level=True,
            do_detrend=True,
            detrend_degree=2
        )

        row = {
            "profil_id": f"{base_id}_{idx}",
            "messsystem": "Perthometer",
            "signal": "z",
            "lambda_c_um": lambda_c_um,
            "source_file": Path(prof.label).name,
            "flank": base_id.split("_")[-1],
            "pos_mm": "",
        }
        row.update(waviness_metrics_dict(perth_w, prefix="W"))
        rows.append(row)

    return rows


# ============================================================
# KMG-Auswertung
# ============================================================

def _parse_pos_mm_from_info(info: dict) -> float:
    """
    Extrahiert den Positionswert pos aus dem info-Dictionary als float.
    """
    s = str(info.get("pos", "")).strip()
    if not s:
        return float("nan")

    m = re.search(r"[+-]?\d+(?:\.\d+)?", s)
    if not m:
        return float("nan")

    return float(m.group(0))


def process_all_kmg_lead_files(
    kmg_dir: str,
    lambda_c_um: float = 800.0,
) -> list[dict]:
    """
    Verarbeitet alle KMG-Lead-Dateien in einem Verzeichnisbaum.

    Für jede Datei:
    - dev-Signal auswerten
    - z-Signal auswerten
    - jeweils Welligkeitskenngrößen berechnen

    Rückgabe:
    - Liste aller Ergebniszeilen
    """
    folder_name = Path(kmg_dir).name

    # Alle Lead-Dateien jeweils für dev und z laden
    dev_blocks = load_all_kmg_lead_blocks(kmg_dir, channel="dev")
    z_blocks = load_all_kmg_lead_blocks(kmg_dir, channel="z")

    # Nach Dateinamen mappen, damit dev und z derselben Datei zusammengehören
    dev_map = {info["filename"]: (prof, info) for prof, info in dev_blocks}
    z_map = {info["filename"]: (prof, info) for prof, info in z_blocks}

    common_filenames = sorted(set(dev_map.keys()) & set(z_map.keys()))
    rows: list[dict] = []

    for fname in common_filenames:
        dev_prof, dev_info = dev_map[fname]
        z_prof, z_info = z_map[fname]

        flank = dev_info.get("flank", "")
        pos_mm = _parse_pos_mm_from_info(dev_info)

        # ----------------------------------------------------
        # dev-Signal
        # ----------------------------------------------------
        kmg_dev_w = waviness_only(
            dev_prof,
            lambda_c_um=lambda_c_um,
            do_level=True,
            do_detrend=False
        )

        row_dev = {
            "profil_id": f"{folder_name}_{flank}_{Path(fname).stem}",
            "messsystem": "KMG",
            "signal": "dev",
            "lambda_c_um": lambda_c_um,
            "source_file": fname,
            "flank": flank,
            "pos_mm": pos_mm,
        }
        row_dev.update(waviness_metrics_dict(kmg_dev_w, prefix="W"))
        rows.append(row_dev)

        # ----------------------------------------------------
        # z-Signal
        # ----------------------------------------------------
        kmg_z_w = waviness_only(
            z_prof,
            lambda_c_um=lambda_c_um,
            do_level=True,
            do_detrend=True,
            detrend_degree=2
        )

        row_z = {
            "profil_id": f"{folder_name}_{flank}_{Path(fname).stem}",
            "messsystem": "KMG",
            "signal": "z",
            "lambda_c_um": lambda_c_um,
            "source_file": fname,
            "flank": flank,
            "pos_mm": pos_mm,
        }
        row_z.update(waviness_metrics_dict(kmg_z_w, prefix="W"))
        rows.append(row_z)

        print(f"[OK] {folder_name} | {flank} | {fname}")

    return rows


# ============================================================
# CSV-Export
# ============================================================

def save_waviness_metrics_csv(out_csv_path: str, rows: list[dict]):
    """
    Speichert alle Ergebniszeilen in einer CSV-Datei.

    Die Ausgabe enthält:
    - Profil-ID
    - Messsystem
    - Signal
    - λc
    - Quelldatei
    - Flanke
    - Position
    - Welligkeitskenngrößen
    """
    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "profil_id",
        "messsystem",
        "signal",
        "lambda_c_um",
        "source_file",
        "flank",
        "pos_mm",
        "Wa_um",
        "Wq_um",
        "Wp_um",
        "Wv_um",
        "Wz_um",
        "Wsk",
        "Wku",
    ]

    normalized_rows = []
    for row in rows:
        normalized_rows.append({k: row.get(k, "") for k in fieldnames})

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


# ============================================================
# Hauptprogramm
# ============================================================

if __name__ == "__main__":
    # Ordner mit Perthometer-Daten
    PERTH_DIR = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Mahr XCR20"

    # Basisordner mit KMG-Ergebnissen
    # Da rekursiv gesucht wird, reicht hier der WSP00-Hauptordner
    KMG_DIRS = [
        "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Zeiss KMG/Ergebnisse/WSP00",
    ]

    # Ziel-CSV
    OUT_CSV = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Zeiss KMG/Ergebnisse/waviness_metrics_perthometer_and_all_kmg_leads_wsp00.csv"

    # Grenzwellenlänge für Welligkeitsfilter
    LAMBDA_C_UM = 800.0

    # Hier werden alle Zeilen gesammelt
    all_rows: list[dict] = []

    # --------------------------------------------------------
    # Perthometer: alle Einzelmessungen pro Flanke
    # --------------------------------------------------------
    perth_base_ids = find_wsp00_perthometer_base_ids(PERTH_DIR)
    print(f"Gefundene WSP00-Perthometer-Flanken: {len(perth_base_ids)}")

    for base_id in perth_base_ids:
        try:
            rows = process_one_perthometer_flank(
                perth_dir=PERTH_DIR,
                base_id=base_id,
                lambda_c_um=LAMBDA_C_UM,
            )
            all_rows.extend(rows)
            print(f"[OK] Perthometer {base_id} | {len(rows)} Wiederholungen")
        except Exception as e:
            print(f"[WARN] Perthometer {base_id} übersprungen: {e}")

    # --------------------------------------------------------
    # KMG: alle Lead-Dateien rekursiv durchsuchen
    # --------------------------------------------------------
    for kmg_dir in KMG_DIRS:
        try:
            rows = process_all_kmg_lead_files(
                kmg_dir=kmg_dir,
                lambda_c_um=LAMBDA_C_UM,
            )
            all_rows.extend(rows)
        except Exception as e:
            print(f"[WARN] KMG {Path(kmg_dir).name} übersprungen: {e}")

    # --------------------------------------------------------
    # Gemeinsame CSV speichern
    # --------------------------------------------------------
    save_waviness_metrics_csv(OUT_CSV, all_rows)

    print(f"\nCSV gespeichert: {OUT_CSV}")
    print(f"Anzahl Zeilen: {len(all_rows)}")