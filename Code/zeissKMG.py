from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import csv
import math
from typing import List, Optional, Tuple, Dict


# ============================================================
# Parser für Zeiss GearPro / KMG TXT-Exporte
#
# Was kann man damit machen?
# - TXT-Exporte von Zeiss/KMG einlesen
# - Headerblöcke wie "lead" und "profile" erkennen
# - Die Nummer hinter "no." als Zahnlücke interpretieren
# - Rechte und linke Flanke auf S1 / S2 abbilden
# - Daten optional nach Lücke getrennt exportieren
# - Prüfen, ob Blöcke einer Lücke ähnlich lang sind
#
# Unterstützte Header-Beispiele:
#   { lead no. 40 138.47 right id 45 }
#   { profile no. 40 -7.28 left id 22 }
#
# Datenzeilenformat:
#   x y z u v w dev
#
# Mapping:
# - right -> S1 = rechte Flanke
# - left  -> S2 = linke Flanke
#
# Einheiten:
# - Falls die Eingabedatei in mm vorliegt, werden x, y, z und dev
#   nach µm umgerechnet, wenn ASSUME_MM_AND_CONVERT_TO_UM=True gesetzt ist
# ============================================================


# =========================
# SETTINGS
# =========================

# Eingabedatei
INPUT_TXT = Path(
    "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Zeiss KMG/WSP00_L40/Verzahnung_Innen_Rad_0000003193_V1_1_Ver.txt"
)

# Basisordner für Exportdateien
OUT_DIR = Path(
    "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Zeiss KMG/Ergebnisse"
)

# Optionaler Prefix, z. B. "WSP00"
# Wenn None, wird versucht, den Prefix automatisch aus Pfad oder Dateiname abzuleiten
FORCE_PREFIX: Optional[str] = None

# Einheiteneinstellung:
# True  -> Eingabewerte werden als mm interpretiert und nach µm umgerechnet
# False -> Werte bleiben in Rohform
ASSUME_MM_AND_CONVERT_TO_UM = True
MM_TO_UM = 1000.0

# Optionaler Filter:
# - "lead"    -> nur lead-Blöcke
# - "profile" -> nur profile-Blöcke
# - None      -> alle Blöcke
ONLY_KIND: Optional[str] = None


# =========================
# REGEX
# =========================

# Header-RegEx:
# erkennt z. B.
# { lead no. 40 138.47 right id 45 }
HDR_RE = re.compile(
    r"^\{\s*(lead|profile)\s+no\.\s*(\d+)\s+([+-]?\d+(?:\.\d+)?)\s+(left|right)\s+id\s+(\d+)\s*\}\s*$",
    re.IGNORECASE,
)

# Datenzeilen-RegEx:
# erwartet genau 7 numerische Spalten:
# x y z u v w dev
DATA_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s*$"
)


# =========================
# DATA CLASSES
# =========================

@dataclass
class BlockHeader:
    """
    Metadaten eines Zeiss-Blocks.

    Felder:
    - kind:
      "lead" oder "profile"
    - gap_no:
      Zahnlücke, interpretiert aus "no."
    - pos:
      Positionswert aus dem Header
    - side:
      "left" oder "right"
    - block_id:
      ID aus dem Header
    - s_label:
      "S1" oder "S2"
    - prefix:
      z. B. "WSP00"
    """
    kind: str
    gap_no: int
    pos: float
    side: str
    block_id: int
    s_label: str
    prefix: str


@dataclass
class Block:
    """
    Ein kompletter Datenblock bestehend aus:
    - Header
    - Messpunkten mit 7 Spalten:
      x, y, z, u, v, w, dev
    """
    header: BlockHeader
    rows: List[Tuple[float, float, float, float, float, float, float]]


# =========================
# HELPERS
# =========================

def side_to_flank(side: str) -> str:
    """
    Wandelt die Seitenbezeichnung in die interne Flankenbezeichnung um.

    Was kann man damit machen?
    - right -> S1
    - left  -> S2
    """
    return "S1" if side.lower() == "right" else "S2"


def maybe_um(v: float) -> float:
    """
    Rechnet einen Wert optional von mm in µm um.

    Was kann man damit machen?
    - Alle relevanten Größen einheitlich in µm ausgeben
    """
    return v * MM_TO_UM if ASSUME_MM_AND_CONVERT_TO_UM else v


def cumulative_arclength_um(x_um, y_um, z_um) -> List[float]:
    """
    Berechnet die kumulative Bogenlänge entlang einer 3D-Punktfolge.

    Was kann man damit machen?
    - Den realen Weg entlang einer Messlinie bestimmen
    - Die gesamte Linienlänge eines Blocks berechnen

    Rückgabe:
    - Liste der kumulativen Weglängen s in µm
    """
    s = [0.0]
    for i in range(1, len(x_um)):
        dx = x_um[i] - x_um[i - 1]
        dy = y_um[i] - y_um[i - 1]
        dz = z_um[i] - z_um[i - 1]
        s.append(s[-1] + math.sqrt(dx * dx + dy * dy + dz * dz))
    return s


def safe_std(values: List[float]) -> float:
    """
    Berechnet die Standardabweichung robust für Listen.

    Was kann man damit machen?
    - Auch bei nur einem Wert sauber 0.0 zurückgeben
    """
    n = len(values)
    if n <= 1:
        return 0.0

    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def derive_prefix_from_path(path: Path) -> str:
    """
    Versucht, den Prefix wie 'WSP00' aus Pfad oder Dateiname abzuleiten.

    Was kann man damit machen?
    - Automatisch den Proben-/Werkstücknamen bestimmen
    - Exportnamen konsistent erzeugen

    Falls nichts gefunden wird:
    - Rückgabe = "WSP"
    """
    patt_wsp = re.compile(r"(WSP\d+)", re.IGNORECASE)

    for p in [*path.parts[::-1], path.stem]:
        m = patt_wsp.search(str(p))
        if m:
            return m.group(1).upper()

    return "WSP"


def sanitize_pos_for_filename(pos: float) -> str:
    """
    Wandelt einen Positionswert in einen dateinamenfreundlichen String um.

    Beispiel:
    - 138.47 -> 138p47
    - -7.28  -> m7p28
    """
    return f"{pos:.2f}".replace(".", "p").replace("-", "m")


# =========================
# PARSER
# =========================

def parse_zeiss_txt(path: Path, prefix: str) -> List[Block]:
    """
    Liest die Zeiss-TXT-Datei ein und zerlegt sie in Blöcke.

    Was kann man damit machen?
    - Header und zugehörige Datenzeilen gruppieren
    - Für jeden Block ein Block-Objekt erzeugen
    - Optional nur lead- oder profile-Blöcke behalten

    Parameter:
    - path:
      Eingabedatei
    - prefix:
      z. B. "WSP00"

    Rückgabe:
    - Liste von Block-Objekten
    """
    blocks: List[Block] = []
    current: Optional[Block] = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # ------------------------------------------------
            # 1) Prüfen, ob die Zeile ein neuer Header ist
            # ------------------------------------------------
            m = HDR_RE.match(line)
            if m:
                # Den bisherigen Block abschließen, falls Daten vorhanden sind
                if current is not None and len(current.rows) > 0:
                    blocks.append(current)

                kind = m.group(1).lower()
                gap_no = int(m.group(2))
                pos = float(m.group(3))
                side = m.group(4).lower()
                bid = int(m.group(5))

                hdr = BlockHeader(
                    kind=kind,
                    gap_no=gap_no,
                    pos=pos,
                    side=side,
                    block_id=bid,
                    s_label=side_to_flank(side),
                    prefix=prefix,
                )
                current = Block(header=hdr, rows=[])
                continue

            # ------------------------------------------------
            # 2) Prüfen, ob die Zeile eine Datenzeile ist
            # ------------------------------------------------
            m2 = DATA_RE.match(line)
            if m2:
                if current is None:
                    # Falls Datenzeilen vor einem Header auftauchen, ignorieren
                    continue

                vals = tuple(float(m2.group(i)) for i in range(1, 8))
                current.rows.append(vals)
                continue

    # Letzten Block übernehmen
    if current is not None and len(current.rows) > 0:
        blocks.append(current)

    # Optional nach lead/profile filtern
    if ONLY_KIND in ("lead", "profile"):
        blocks = [b for b in blocks if b.header.kind == ONLY_KIND]

    return blocks


# =========================
# EXPORT
# =========================

def export_blocks_split_by_gap(blocks: List[Block], out_base: Path):
    """
    Exportiert alle Blöcke getrennt nach Zahnlücke.

    Zielstruktur:
      out_base/<PREFIX>/L<gap_no>/...

    Zusätzlich werden erzeugt:
    - summary_blocks_all.csv
    - summary_blocks_L<gap>.csv für jede Lücke

    Was kann man damit machen?
    - Einzelblöcke getrennt abspeichern
    - Überblick über alle exportierten Blöcke behalten
    """
    if not blocks:
        print("[WARN] Keine Blöcke gefunden.")
        return

    prefix = blocks[0].header.prefix
    base = out_base / prefix
    base.mkdir(parents=True, exist_ok=True)

    # Gesamt-Summary über alle Lücken
    summary_all = base / "summary_blocks_all.csv"
    with summary_all.open("w", newline="", encoding="utf-8") as fall:
        wall = csv.writer(fall)
        wall.writerow([
            "source_file",
            "prefix",
            "gap_no",
            "kind",
            "pos_value",
            "pos_unit",
            "side",
            "S",
            "id",
            "n_points",
            "s_um_total",
            "dev_min",
            "dev_max",
            "dev_std",
            "out_file",
        ])

        # Nach Zahnlücke gruppieren
        by_gap: Dict[int, List[Block]] = {}
        for b in blocks:
            by_gap.setdefault(b.header.gap_no, []).append(b)

        for gap_no, gap_blocks in sorted(by_gap.items()):
            gap_dir = base / f"L{gap_no}"
            gap_dir.mkdir(parents=True, exist_ok=True)

            # Summary nur für diese Lücke
            summary_gap = gap_dir / f"summary_blocks_L{gap_no}.csv"
            with summary_gap.open("w", newline="", encoding="utf-8") as fgap:
                wgap = csv.writer(fgap)
                wgap.writerow([
                    "source_file",
                    "prefix",
                    "gap_no",
                    "kind",
                    "pos_value",
                    "pos_unit",
                    "side",
                    "S",
                    "id",
                    "n_points",
                    "s_um_total",
                    "dev_min",
                    "dev_max",
                    "dev_std",
                    "out_file",
                ])

                for b in gap_blocks:
                    hdr = b.header

                    # Spalten entpacken
                    x, y, z, u, v, w_norm, dev = zip(*b.rows)

                    # Optional mm -> µm umrechnen
                    x_um = [maybe_um(xx) for xx in x]
                    y_um = [maybe_um(yy) for yy in y]
                    z_um = [maybe_um(zz) for zz in z]
                    dev_um = [maybe_um(dd) for dd in dev]

                    # Kumulative Bogenlänge berechnen
                    s_um = cumulative_arclength_um(x_um, y_um, z_um)

                    # Statistiken
                    n = len(dev_um)
                    dev_min = min(dev_um)
                    dev_max = max(dev_um)
                    dev_std = safe_std(dev_um)

                    pos_str = sanitize_pos_for_filename(hdr.pos)

                    # Dateiname für den Einzelblock
                    out_name = (
                        f"{hdr.prefix}_L{hdr.gap_no}_"
                        f"{hdr.kind}_pos{pos_str}_"
                        f"{hdr.side}_id{hdr.block_id}_{hdr.s_label}.csv"
                    )
                    out_path = gap_dir / out_name

                    # Einzeldatei schreiben
                    with out_path.open("w", newline="", encoding="utf-8") as fout:
                        wr = csv.writer(fout)
                        wr.writerow([f"# prefix: {hdr.prefix}"])
                        wr.writerow([f"# gap_no: {hdr.gap_no}"])
                        wr.writerow([f"# kind: {hdr.kind}"])
                        wr.writerow([f"# pos: {hdr.pos} ({'mm' if ASSUME_MM_AND_CONVERT_TO_UM else 'raw'})"])
                        wr.writerow([f"# side: {hdr.side}"])
                        wr.writerow([f"# flank: {hdr.s_label}"])
                        wr.writerow([f"# id: {hdr.block_id}"])
                        wr.writerow([f"# units: {'µm' if ASSUME_MM_AND_CONVERT_TO_UM else 'raw'}"])
                        wr.writerow([])

                        wr.writerow([
                            "s_um",
                            "dev_um" if ASSUME_MM_AND_CONVERT_TO_UM else "dev",
                            "x_um" if ASSUME_MM_AND_CONVERT_TO_UM else "x",
                            "y_um" if ASSUME_MM_AND_CONVERT_TO_UM else "y",
                            "z_um" if ASSUME_MM_AND_CONVERT_TO_UM else "z",
                            "u", "v", "w",
                        ])

                        for i in range(n):
                            wr.writerow([s_um[i], dev_um[i], x_um[i], y_um[i], z_um[i], u[i], v[i], w_norm[i]])

                    # Eine Summary-Zeile bauen
                    row = [
                        INPUT_TXT.name,
                        hdr.prefix,
                        hdr.gap_no,
                        hdr.kind,
                        hdr.pos,
                        "mm" if ASSUME_MM_AND_CONVERT_TO_UM else "raw",
                        hdr.side,
                        hdr.s_label,
                        hdr.block_id,
                        n,
                        s_um[-1] if s_um else 0.0,
                        dev_min,
                        dev_max,
                        dev_std,
                        out_name,
                    ]

                    wgap.writerow(row)
                    wall.writerow(row)

    print(f"[OK] Exportiert: {len(blocks)} Blöcke")
    print(f"[OK] Basisordner: {base}")
    print(f"[OK] Gesamt-Summary: {summary_all}")


def check_block_lengths(blocks: List[Block], tol_um: float = 5.0):
    """
    Prüft, ob Blöcke gleicher Art und gleicher Lücke ähnliche Längen haben.

    Was kann man damit machen?
    - Kontrollieren, ob Blöcke innerhalb einer Lücke konsistent sind
    - Auffällige Längenunterschiede schnell erkennen

    Parameter:
    - tol_um:
      Toleranz in µm, bis zu der die Längen als 'gleich' gelten
    """
    if not blocks:
        print("[WARN] Keine Blöcke vorhanden.")
        return

    # Gruppierung nach (Lücke, Typ)
    grouped: Dict[Tuple[int, str], List[Tuple[BlockHeader, int, float]]] = {}

    for b in blocks:
        x, y, z, u, v, w_norm, dev = zip(*b.rows)

        x_um = [maybe_um(xx) for xx in x]
        y_um = [maybe_um(yy) for yy in y]
        z_um = [maybe_um(zz) for zz in z]

        s_um = cumulative_arclength_um(x_um, y_um, z_um)
        total_len = s_um[-1] if s_um else 0.0
        n_points = len(b.rows)

        key = (b.header.gap_no, b.header.kind)
        grouped.setdefault(key, []).append((b.header, n_points, total_len))

    print("\n=== Längenprüfung der Blöcke ===")
    for (gap_no, kind), items in sorted(grouped.items()):
        lengths = [it[2] for it in items]
        npts = [it[1] for it in items]

        lmin = min(lengths)
        lmax = max(lengths)
        spread = lmax - lmin

        nmin = min(npts)
        nmax = max(npts)

        status = "OK" if spread <= tol_um else "NICHT GLEICH"

        print(f"\nLücke {gap_no}, Typ {kind}: {status}")
        print(f"  Punkte: min={nmin}, max={nmax}")
        print(f"  Länge [µm]: min={lmin:.3f}, max={lmax:.3f}, Δ={spread:.3f}")

        for hdr, n_points, total_len in sorted(items, key=lambda t: (t[0].side, t[0].pos)):
            print(
                f"    id={hdr.block_id:>3} | "
                f"{hdr.side:<5} | "
                f"pos={hdr.pos:>8.2f} | "
                f"S={hdr.s_label} | "
                f"n={n_points:>6} | "
                f"L={total_len:>10.3f} µm"
            )


# =========================
# RUN
# =========================

if __name__ == "__main__":
    # Prüfen, ob die Eingabedatei existiert
    if not INPUT_TXT.exists():
        raise FileNotFoundError(INPUT_TXT)

    # Prefix bestimmen:
    # entweder manuell vorgegeben oder automatisch aus Pfad ableiten
    prefix = FORCE_PREFIX if FORCE_PREFIX else derive_prefix_from_path(INPUT_TXT)

    # Blöcke einlesen
    blocks = parse_zeiss_txt(INPUT_TXT, prefix=prefix)

    # Längenprüfung durchführen
    check_block_lengths(blocks, tol_um=5.0)

    # Export aktivieren, wenn die Einzeldateien geschrieben werden sollen
    export_blocks_split_by_gap(blocks, OUT_DIR)