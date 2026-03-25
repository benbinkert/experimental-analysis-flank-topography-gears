from pathlib import Path
import numpy as np
import pandas as pd
from surfalize import Surface
import re


# ============================================================
# ISO-Flächenkennwerte, die für jede Oberfläche berechnet werden
# ============================================================
ISO_PARAMETERS = (
    'Sa', 'Sq', 'Sp', 'Sv', 'Sz', 'Ssk', 'Sku', 'Sdr', 'Sdq', 'Sal', 'Str',
    'Sk', 'Spk', 'Svk', 'Smr1', 'Smr2', 'Sxp', 'Vmp', 'Vmc', 'Vvv', 'Vvc'
)

# Ordner für kombinierte Ergebnisdateien
OUT_DIR = Path("/Users/benbinkert/PycharmProjects/Bachelorarbeit/Ergebnisse/Messsystemvergleich")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_combined_vergleich_csv(
    df00_nano: pd.DataFrame,
    df03_nano: pd.DataFrame,
    df_key_vergl_00: pd.DataFrame,
    df_key_vergl_03: pd.DataFrame,
    out_name: str = "Daten-Messsystemvergleich_vergleich.csv"
) -> Path:
    """
    Fasst mehrere bereits berechnete Vergleichs-DataFrames in einer CSV zusammen.

    Was kann man damit machen?
    - Ergebnisse von NanoFocus und Keyence gemeinsam abspeichern
    - WSP00 und WSP03 in einer Datei zusammenführen
    - Später einfacher filtern, gruppieren und statistisch auswerten

    Parameter:
    - df00_nano:
      Vergleichsdaten für WSP00 aus NanoFocus
    - df03_nano:
      Vergleichsdaten für WSP03 aus NanoFocus
    - df_key_vergl_00:
      Vergleichsdaten für WSP00 aus Keyence
    - df_key_vergl_03:
      Vergleichsdaten für WSP03 aus Keyence
    - out_name:
      Name der Ausgabedatei

    Rückgabe:
    - Pfad zur gespeicherten CSV-Datei
    """
    # Defensive Kopien, damit Original-DataFrames nicht verändert werden
    a = df00_nano.copy()
    b = df03_nano.copy()
    c = df_key_vergl_00.copy()
    d = df_key_vergl_03.copy()

    # Dataset-Spalte ergänzen, damit später nach WSP00 / WSP03 gefiltert werden kann
    a["dataset"] = "WSP00"
    b["dataset"] = "WSP03"
    c["dataset"] = "WSP00"
    d["dataset"] = "WSP03"

    # Alle DataFrames auf dieselben Spalten bringen
    # Fehlende Spalten werden automatisch mit NaN aufgefüllt
    all_cols = sorted(set(a.columns) | set(b.columns) | set(c.columns) | set(d.columns))
    a = a.reindex(columns=all_cols)
    b = b.reindex(columns=all_cols)
    c = c.reindex(columns=all_cols)
    d = d.reindex(columns=all_cols)

    # Alles untereinander zusammenfügen
    df_all = pd.concat([a, b, c, d], ignore_index=True)

    # Spaltenreihenfolge etwas schöner sortieren
    preferred_front = ["dataset", "messsystem", "filtercase", "datei", "cropped_to", "cropped_to"]
    front = [x for x in preferred_front if x in df_all.columns]
    rest = [x for x in df_all.columns if x not in front]
    df_all = df_all[front + rest]

    # CSV speichern
    out_path = OUT_DIR / out_name
    df_all.to_csv(out_path, index=False, sep=";")

    print(f"[OK] Kombi-CSV gespeichert:\n{out_path}")
    print(f"[OK] Zeilen: {len(df_all)} | Spalten: {len(df_all.columns)}")

    return out_path


def _nano_match_name(keyence_filename: str) -> str | None:
    """
    Extrahiert den gemeinsamen Probennamen aus einem Keyence-Dateinamen.

    Was kann man damit machen?
    - Einen Keyence-Dateinamen auf das passende NanoFocus-Pendant abbilden
    - Dateinamen wie WSP00_L1_S1.sdf oder WSP00_L1_S1_irgendwas.sdf erkennen
    - Den Basisnamen für das Matching beider Messsysteme gewinnen

    Rückgabe:
    - Basisname wie z. B. 'WSP00_L1_S1'
    - None, falls kein passendes Muster gefunden wird
    """
    m = re.search(r"(WSP\d+_L\d+_S[12])", keyence_filename)
    return m.group(1) if m else None


def batch_to_csv_keyence_cropped_to_nano(
    key_folder: str,
    key_pattern: str,
    nano_folder: str,
    nano_ext: str = ".nms",
    out_csv: str = "ISO_Keyence_vergleich_cropped.csv",
    filtercase: str = "vergleich",
    params=ISO_PARAMETERS
):
    """
    Lädt Keyence-Dateien, croppt sie auf das Fenster der passenden NanoFocus-Dateien
    und berechnet daraus ISO-Kennwerte.

    Was kann man damit machen?
    - Keyence und NanoFocus auf dieselbe Auswertefläche bringen
    - Vergleichbare Kennwerte für beide Messsysteme erzeugen
    - Direkt eine CSV für den Messsystemvergleich speichern

    Ablauf:
    1. Keyence-Dateien suchen
    2. Passende NanoFocus-Datei anhand des Dateinamens finden
    3. Keyence-Fläche auf Größe der NanoFocus-Fläche croppen
    4. Vorverarbeitung und ISO-Kennwertberechnung durchführen
    5. Ergebnisse als CSV speichern

    Rückgabe:
    - DataFrame mit allen berechneten Kennwerten
    """
    key_folder = Path(key_folder)
    nano_folder = Path(nano_folder)

    files = sorted(key_folder.glob(key_pattern))
    rows = []

    for f in files:
        # Passenden Basisnamen aus der Keyence-Datei extrahieren
        base = _nano_match_name(f.name)
        if base is None:
            print(f"[SKIP] kein WSP*_L*_S* in {f.name}")
            continue

        # Passende NanoFocus-Datei zusammensetzen
        nano_path = nano_folder / f"{base}{nano_ext}"
        if not nano_path.exists():
            print(f"[SKIP] Nano fehlt: {nano_path.name} (für {f.name})")
            continue

        # Beide Oberflächen laden
        nano = Surface.load(str(nano_path))
        key = Surface.load(str(f))

        # Keyence wird auf dieselbe Fenstergröße wie NanoFocus gecroppt
        crop_to = (float(nano.width_um), float(nano.height_um))

        # Keyence-Oberfläche vorverarbeiten
        key_p = preprocess_surface(
            key,
            messsystem="Keyence",
            filtercase=filtercase,
            crop_to_um=crop_to
        )

        # Ergebniszeile aufbauen
        row = {
            "messsystem": "Keyence",
            "filtercase": filtercase,
            "datei": f.name,
            "cropped_to": nano_path.name
        }
        row.update(compute_iso_params(key_p, params=params))
        rows.append(row)

    # DataFrame erzeugen und speichern
    df = pd.DataFrame(rows)
    out_path = key_folder / out_csv
    df.to_csv(out_path, index=False, sep=";")

    print(f"\nGespeichert: {out_path}")
    return df


def symmetric_crop_to(surf: Surface, target_w_um: float, target_h_um: float) -> Surface:
    """
    Schneidet eine Oberfläche symmetrisch auf eine Zielgröße zu.

    Was kann man damit machen?
    - Größere Flächen auf ein kleineres, zentriertes Fenster reduzieren
    - Für den Messsystemvergleich identische Auswertebereiche herstellen
    - Randbereiche gezielt entfernen

    Parameter:
    - surf:
      Eingangsoberfläche
    - target_w_um, target_h_um:
      Zielbreite und Zielhöhe in µm

    Rückgabe:
    - Gecroppte Oberfläche
    """
    if surf.width_um <= target_w_um or surf.height_um <= target_h_um:
        raise ValueError(
            f"Surface zu klein für Crop: surf={surf.width_um:.1f}x{surf.height_um:.1f} µm, "
            f"target={target_w_um:.1f}x{target_h_um:.1f} µm"
        )

    # Randabstände links/rechts und oben/unten symmetrisch bestimmen
    dx = (surf.width_um - target_w_um) / 2.0
    dy = (surf.height_um - target_h_um) / 2.0

    # Croppen in physikalischen Einheiten
    return surf.crop((dx, dx + target_w_um, dy, dy + target_h_um), in_units=True)


def preprocess_surface(
    surface: Surface,
    messsystem: str,
    filtercase: str,
    crop_to_um: tuple[float, float] | None = None
) -> Surface:
    """
    Führt die komplette Vorverarbeitung einer Oberfläche durch.

    Was kann man damit machen?
    - Eine Rohoberfläche standardisiert vorbereiten
    - Messsystemabhängige Filter anwenden
    - Vergleichbare Kennwerte für verschiedene Datensätze erzeugen

    Enthaltene Schritte:
    1. Optionales Cropping auf eine Zielgröße
    2. Leveln der Oberfläche
    3. Polynomiales Detrending
    4. Thresholding
    5. Auffüllen nicht gemessener Werte
    6. Filterung abhängig von Messsystem und Filterfall

    Parameter:
    - surface:
      Eingangsoberfläche
    - messsystem:
      'Nanofocus' oder 'Keyence'
    - filtercase:
      'prozessfrei', 'vergleich' oder 'gesamt'
    - crop_to_um:
      Optionales Crop-Ziel in µm als (Breite, Höhe)

    Rückgabe:
    - Vorverarbeitete Oberfläche
    """
    # Optional zuerst croppen,
    # damit Randartefakte bereits vor der weiteren Verarbeitung reduziert werden
    if crop_to_um is not None:
        tw, th = crop_to_um
        surface = symmetric_crop_to(surface, tw, th)

    # Grundlegende Vorverarbeitung
    surface = surface.level()
    surface = Surface.detrend_polynomial(surface, degree=2)
    surface = surface.threshold(threshold=(0.25, 0.25))
    surface = surface.fill_nonmeasured_rowwise_linear()

    # S-Filter abhängig vom Messsystem wählen
    if messsystem == "Nanofocus":
        cutoff_s = 1.6
    elif messsystem == "Keyence":
        cutoff_s = 8.0
    else:
        raise ValueError("messsystem muss 'Nanofocus' oder 'Keyence' sein")

    # L-Filter bzw. Filterstrategie abhängig vom Auswertefall
    if filtercase == "prozessfrei":
        cutoff_l = 100.0
        surface = surface.filter(filter_type="bandpass", cutoff=cutoff_s, cutoff2=cutoff_l)

    elif filtercase == "vergleich":
        cutoff_l = 250.0
        # Hier bewusst cutoff=8 gesetzt,
        # damit beide Systeme im Vergleich auf derselben unteren Filtergrenze laufen
        surface = surface.filter(filter_type="bandpass", cutoff=8, cutoff2=cutoff_l)

    elif filtercase == "gesamt":
        surface = surface.filter(filter_type="lowpass", cutoff=cutoff_s)

    else:
        raise ValueError("filtercase muss 'prozessfrei', 'vergleich' oder 'gesamt' sein")

    # Nach der Filterung eventuell erneut nicht gemessene Werte auffüllen
    surface = surface.fill_nonmeasured_rowwise_linear()
    return surface


def compute_iso_params(surface: Surface, params=ISO_PARAMETERS) -> dict:
    """
    Berechnet eine Liste von ISO-Kennwerten für eine Oberfläche.

    Was kann man damit machen?
    - Eine Oberfläche automatisiert über viele Kennwerte auswerten
    - Funktions- und Attributzugriffe einheitlich behandeln
    - Fehlerhafte Kennwerte sauber als NaN auffangen

    Parameter:
    - surface:
      Vorverarbeitete Oberfläche
    - params:
      Liste oder Tupel von Parameternamen

    Rückgabe:
    - Dictionary {Parametername: Wert}
    """
    out = {}

    for p in params:
        val = np.nan
        try:
            attr = getattr(surface, p, None)

            # Falls der Kennwert als Methode vorliegt -> aufrufen
            if callable(attr):
                val = float(attr())

            # Falls der Kennwert als Attribut vorliegt -> direkt lesen
            elif attr is not None:
                val = float(attr)

        except Exception:
            val = np.nan

        out[p] = val

    return out


def batch_to_csv(
    folder: str,
    pattern: str,
    out_csv: str,
    messsystem: str,
    filtercase: str,
    params=ISO_PARAMETERS
):
    """
    Lädt alle Dateien eines Ordners, verarbeitet sie und speichert die ISO-Kennwerte als CSV.

    Was kann man damit machen?
    - Ganze Messreihen automatisiert auswerten
    - Für ein bestimmtes Messsystem und einen Filterfall Kennwerte erzeugen
    - Direkt eine strukturierte CSV für die weitere Auswertung speichern

    Parameter:
    - folder:
      Eingabeordner
    - pattern:
      Dateimuster, z. B. 'WSP00_L*_S*.nms'
    - out_csv:
      Name der Ausgabedatei
    - messsystem:
      'Nanofocus' oder 'Keyence'
    - filtercase:
      'prozessfrei', 'vergleich' oder 'gesamt'
    - params:
      Zu berechnende ISO-Kennwerte

    Rückgabe:
    - DataFrame mit den berechneten Kennwerten
    """
    folder = Path(folder)
    files = sorted(folder.glob(pattern))

    if not files:
        print(f"Keine Dateien gefunden: {folder} / {pattern}")
        return pd.DataFrame(columns=["messsystem", "filtercase", "datei", *params])

    rows = []

    for f in files:
        print(f"== {messsystem} | {filtercase} | {f.name} ==")

        # Oberfläche laden und vorverarbeiten
        surf = Surface.load(str(f))
        surf = preprocess_surface(surf, messsystem=messsystem, filtercase=filtercase)

        # Kennwerte in einer Zeile speichern
        row = {"messsystem": messsystem, "filtercase": filtercase, "datei": f.name}
        row.update(compute_iso_params(surf, params=params))
        rows.append(row)

    # DataFrame mit definierter Spaltenreihenfolge erzeugen
    df = pd.DataFrame(rows, columns=["messsystem", "filtercase", "datei", *params])

    # CSV speichern
    out_path = folder / out_csv
    df.to_csv(out_path, index=False, sep=";")

    print(f"\nGespeichert: {out_path}")
    return df


# ============================================================
# Hauptfunktion:
# Hier werden die gewünschten Auswertungen für verschiedene
# Datensätze, Messsysteme und Filterfälle gestartet
# ============================================================

def kenngroesenBerechnung():
    """
    Führt die ausgewählten Batch-Auswertungen aus.

    Was kann man damit machen?
    - Vergleichsdatensätze für NanoFocus und Keyence erzeugen
    - Keyence-Daten auf NanoFocus-Fenster croppen
    - Alle Vergleichsergebnisse in einer gemeinsamen CSV zusammenführen

    Hinweis:
    - Einige Blöcke sind aktuell auskommentiert
    - Dadurch lässt sich gezielt nur ein bestimmter Auswertefall laufen
    """
    """
    # Nanofocus WSP00
    df00_nano = batch_to_csv(
        folder="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben",
        pattern="WSP00_L*_S*.nms",
        out_csv="ISO_WSP00_Nanofocus_prozessfrei.csv",
        messsystem="Nanofocus",
        filtercase="prozessfrei"
    )

    # Nanofocus WSP03
    df03_nano = batch_to_csv(
        folder="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP03/ErodierteProben",
        pattern="WSP03_L*_S*.nms",
        out_csv="ISO_WSP03_Nanofocus_prozessfrei.csv",
        messsystem="Nanofocus",
        filtercase="prozessfrei"
    )

    # Keyence WSP00
    df00_key = batch_to_csv(
        folder="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP00/ErodierteProben/Speicher",
        pattern="WSP00_L*_*.sdf",
        out_csv="ISO_WSP00_Keyence_prozessfrei.csv",
        messsystem="Keyence",
        filtercase="prozessfrei"
    )

    # Keyence WSP03
    df00_key = batch_to_csv(
        folder="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP03/Speicher",
        pattern="WSP03_L*_*.sdf",
        out_csv="ISO_WSP03_Keyence_prozessfrei.csv",
        messsystem="Keyence",
        filtercase="prozessfrei"
    )
    """

    # --- NanoFocus WSP00 Vergleich ---
    df00_nano = batch_to_csv(
        folder="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben",
        pattern="WSP00_L*_S*.nms",
        out_csv="ISO_WSP00_Nanofocus_vergleich.csv",
        messsystem="Nanofocus",
        filtercase="vergleich"
    )

    # --- NanoFocus WSP03 Vergleich ---
    df03_nano = batch_to_csv(
        folder="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP03/ErodierteProben",
        pattern="WSP03_L*_S*.nms",
        out_csv="ISO_WSP03_Nanofocus_vergleich.csv",
        messsystem="Nanofocus",
        filtercase="vergleich"
    )

    # --- Keyence WSP00 Vergleich (auf NanoFocus-Fenster gecroppt) ---
    df_key_vergl_00 = batch_to_csv_keyence_cropped_to_nano(
        key_folder="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP00/1x7/ErgebnisseMatInSDF",
        key_pattern="WSP00_L*_S*.sdf",
        nano_folder="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben",
        nano_ext=".nms",
        out_csv="ISO_WSP00_Keyence_vergleich_cropped_to_nano.csv",
        filtercase="vergleich",
    )

    # --- Keyence WSP03 Vergleich (auf NanoFocus-Fenster gecroppt) ---
    df_key_vergl_03 = batch_to_csv_keyence_cropped_to_nano(
        key_folder="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP03/Speicher",
        key_pattern="WSP03_L*_S*.sdf",
        nano_folder="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP03/ErodierteProben",
        nano_ext=".nms",
        out_csv="ISO_WSP03_Keyence_vergleich_cropped_to_nano.csv",
        filtercase="vergleich",
    )

    # --- Alles zusammen in eine gemeinsame Vergleichs-CSV speichern ---
    save_combined_vergleich_csv(
        df00_nano=df00_nano,
        df03_nano=df03_nano,
        df_key_vergl_00=df_key_vergl_00,
        df_key_vergl_03=df_key_vergl_03,
        out_name="Daten-Messsystemvergleich_vergleich.csv"
    )

    """
    # Nanofocus WSP00 gesamt
    df00_nano = batch_to_csv(
        folder="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben",
        pattern="WSP00_L*_S*.nms",
        out_csv="ISO_WSP00_Nanofocus_gesamt.csv",
        messsystem="Nanofocus",
        filtercase="gesamt"
    )

    # Nanofocus WSP03 gesamt
    df03_nano = batch_to_csv(
        folder="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP03/ErodierteProben",
        pattern="WSP03_L*_S*.nms",
        out_csv="ISO_WSP03_Nanofocus_gesamt.csv",
        messsystem="Nanofocus",
        filtercase="gesamt"
    )

    # Keyence WSP00 gesamt
    df00_key = batch_to_csv(
        folder="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP00/ErodierteProben/Speicher",
        pattern="WSP00_L*_*.sdf",
        out_csv="ISO_WSP00_Keyence_gesamt.csv",
        messsystem="Keyence",
        filtercase="gesamt"
    )

    # Keyence WSP03 gesamt
    df03_key = batch_to_csv(
        folder="/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP03/Speicher",
        pattern="WSP03_L*_*.sdf",
        out_csv="ISO_WSP03_Keyence_gesamt.csv",
        messsystem="Keyence",
        filtercase="gesamt"
    )
    """


if __name__ == "__main__":
    kenngroesenBerechnung()