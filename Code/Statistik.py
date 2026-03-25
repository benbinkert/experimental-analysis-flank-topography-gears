import pandas as pd
import numpy as np


# ============================================================
# Statistik und Messsystemvergleich aus einer gemeinsamen CSV
#
# Was kann man damit machen?
# - Die gemeinsame Vergleichs-CSV einlesen
# - Für Teilmengen Mittelwert und Standardabweichung berechnen
# - WSP00 und WSP03 getrennt auswerten
# - NanoFocus und Keyence direkt miteinander vergleichen
# - Absolute und relative Unterschiede der Mittelwerte bestimmen
# ============================================================


# Pfad zur CSV-Datei mit den Vergleichsdaten
dateipfad = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Ergebnisse/Messystemvergleich/Daten-Messsystemvergleich_vergleich.csv"

# CSV einlesen
df = pd.read_csv(dateipfad, sep=";")


# Spaltennamen zentral definieren
# Damit kann man sie später leichter anpassen, falls sich die CSV-Struktur ändert
COL_SYS = "messsystem"
COL_FILE = "datei"


def stats_for(df_sub: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Kennstatistiken für alle numerischen Spalten einer Teilmenge.

    Was kann man damit machen?
    - Mittelwerte aller numerischen Kennwerte berechnen
    - Standardabweichungen mit ausgeben
    - Die Größe der betrachteten Teilmenge dokumentieren

    Parameter:
    - df_sub:
      Teil-DataFrame, für den die Statistik berechnet werden soll

    Rückgabe:
    - DataFrame mit:
      * Mittelwert
      * Standardabweichung
      * n
    """
    # Nur numerische Spalten auswählen
    num = df_sub.select_dtypes(include="number")

    # Statistik-DataFrame aufbauen
    return pd.DataFrame({
        "Mittelwert": num.mean(numeric_only=True),
        "Standardabweichung": num.std(numeric_only=True),

        # n ist hier für alle Zeilen gleich,
        # wird aber bewusst als Spalte mitgeführt
        "n": len(df_sub)
    })


def subset(df_all: pd.DataFrame, wsp: str, system: str) -> pd.DataFrame:
    """
    Filtert die Gesamtdaten auf ein bestimmtes WSP und ein bestimmtes Messsystem.

    Was kann man damit machen?
    - Nur WSP00 oder nur WSP03 auswählen
    - Gleichzeitig nach NanoFocus oder Keyence filtern
    - Eine saubere Teilmenge für weitere Statistikfunktionen erzeugen

    Parameter:
    - df_all:
      Gesamter DataFrame
    - wsp:
      Suchstring wie 'WSP00' oder 'WSP03'
    - system:
      Messsystemname, z. B. 'Nanofocus' oder 'Keyence'

    Rückgabe:
    - Gefilterter DataFrame
    """
    # Zeilen finden, deren Dateiname das gewünschte WSP enthält
    m_wsp = df_all[COL_FILE].astype(str).str.contains(wsp, na=False)

    # Zeilen finden, deren Messsystem zum gewünschten System passt
    m_sys = df_all[COL_SYS].astype(str).str.lower().eq(system.lower())

    # Beide Bedingungen kombinieren
    return df_all[m_wsp & m_sys].copy()


# ==========================================================
# A) ALT: Nur Nanofocus -> WSP00 vs WSP03
# ==========================================================

# Nur NanoFocus-Daten herausfiltern
df_nanofocus = df[df[COL_SYS].astype(str).str.lower().eq("nanofocus")].copy()

# NanoFocus-Daten für WSP00
df_wsp00_nano = df_nanofocus[
    df_nanofocus[COL_FILE].astype(str).str.contains("WSP00", na=False)
].copy()

# NanoFocus-Daten für WSP03
df_wsp03_nano = df_nanofocus[
    df_nanofocus[COL_FILE].astype(str).str.contains("WSP03", na=False)
].copy()

# Statistik ausgeben
print("=== Statistik WSP00 | Nanofocus ===")
print(stats_for(df_wsp00_nano).to_string())

print("\n=== Statistik WSP03 | Nanofocus ===")
print(stats_for(df_wsp03_nano).to_string())


# ==========================================================
# B) NEU: Messsystemvergleich pro WSP: Nano vs Keyence
# Ausgabe:
# - mean_nano
# - std_nano
# - mean_keyence
# - std_keyence
# - diff_abs
# - diff_rel
# ==========================================================

def compare_table(wsp: str) -> pd.DataFrame:
    """
    Erstellt eine Vergleichstabelle für ein WSP zwischen NanoFocus und Keyence.

    Was kann man damit machen?
    - Mittelwerte und Standardabweichungen beider Messsysteme nebeneinander stellen
    - Absolute Unterschiede der Mittelwerte berechnen
    - Relative Unterschiede bezogen auf NanoFocus bestimmen

    Parameter:
    - wsp:
      z. B. 'WSP00' oder 'WSP03'

    Rückgabe:
    - Vergleichstabelle mit Kennwerten pro Parameter
    """
    # Statistik für NanoFocus berechnen
    nano_stats = stats_for(subset(df, wsp, "Nanofocus"))[
        ["Mittelwert", "Standardabweichung"]
    ].copy()

    # Statistik für Keyence berechnen
    key_stats = stats_for(subset(df, wsp, "Keyence"))[
        ["Mittelwert", "Standardabweichung"]
    ].copy()

    # Beide Tabellen über den Parameterindex zusammenführen
    out = nano_stats.join(
        key_stats,
        how="outer",
        lsuffix="_Nano",
        rsuffix="_Key"
    )

    # Spaltennamen vereinheitlichen
    out = out.rename(columns={
        "Mittelwert_Nano": "mean_nano",
        "Standardabweichung_Nano": "std_nano",
        "Mittelwert_Key": "mean_keyence",
        "Standardabweichung_Key": "std_keyence",
    })

    # Absoluter Unterschied: Keyence minus NanoFocus
    out["diff_abs"] = out["mean_keyence"] - out["mean_nano"]

    # Relativer Unterschied bezogen auf NanoFocus
    out["diff_rel"] = out["diff_abs"] / out["mean_nano"]

    return out


# Vergleichstabellen ausgeben
print("\n=== Vergleich WSP00: mean/std Nano vs Keyence + diff ===")
print(compare_table("WSP00").to_string())

print("\n=== Vergleich WSP03: mean/std Nano vs Keyence + diff ===")
print(compare_table("WSP03").to_string())