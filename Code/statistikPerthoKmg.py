import pandas as pd
import numpy as np
import re


# ============================================================
# Statistik pro Flanke
#
# Was kann man damit machen?
# - CSV mit Perthometer- und KMG-Daten einlesen
# - Aus profil_id eine einheitliche Flanken-ID erzeugen
# - Nur Perthometer und KMG-z behalten
# - Pro Flanke und Messsystem:
#     * Anzahl n
#     * Mittelwert
#     * Standardabweichung
#   berechnen
# - Ergebnis zusätzlich als neue CSV speichern
# ============================================================


dateipfad = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Ergebnisse/KMGvsPertho/waviness_metrics_perthometer_and_all_kmg_leads_wsp00.csv"
out_csv = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Ergebnisse/KMGvsPertho/statistik_flanken_nur_z.csv"

# Deine Datei ist kommagetrennt
df = pd.read_csv(dateipfad, sep=",")


def extract_flanken_id(profil_id: str) -> str | None:
    """
    Erzeugt aus profil_id eine einheitliche Flanken-ID.

    Beispiele:
    - WSP00_L13_S1_1 -> WSP00_L13_S1
    - WSP00_L1_S2_3  -> WSP00_L1_S2
    - WSP00_S2_WSP00_L1_lead_pos136p57_left_id34_S2 -> WSP00_L1_S2
    """
    s = str(profil_id)

    # Perthometer
    m1 = re.search(r"(WSP\d+_L\d+_S[12])(?:_\d+)?$", s)
    if m1:
        return m1.group(1)

    # KMG
    m2 = re.search(r"(WSP\d+_L\d+).*?(S[12])$", s)
    if m2:
        return f"{m2.group(1)}_{m2.group(2)}"

    return None


# Einheitliche Flanken-ID
df["flanken_id"] = df["profil_id"].apply(extract_flanken_id)
df = df[df["flanken_id"].notna()].copy()


# ============================================================
# Nur Perthometer + KMG z behalten
# ============================================================
df_filtered = df[
    (df["messsystem"].eq("Perthometer")) |
    ((df["messsystem"].eq("KMG")) & (df["signal"].eq("z")))
].copy()


def gruppenstatistik(df_sub: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet n, Mittelwert und Standardabweichung
    für alle Welligkeitskenngrößen je Flanke und Messsystem.
    """
    kennwerte = ["Wa_um", "Wq_um", "Wp_um", "Wv_um", "Wz_um", "Wsk", "Wku"]

    agg_dict = {}
    for k in kennwerte:
        agg_dict[k] = ["mean", "std"]

    out = df_sub.groupby(["flanken_id", "messsystem"]).agg(agg_dict)

    # MultiIndex-Spalten glätten
    out.columns = [f"{col}_{stat}" for col, stat in out.columns]

    # n ergänzen
    n = df_sub.groupby(["flanken_id", "messsystem"]).size().rename("n")
    out = out.join(n)

    return out.reset_index()


ergebnis = gruppenstatistik(df_filtered)

# schönere Spaltenreihenfolge
spalten = [
    "flanken_id",
    "messsystem",
    "n",
    "Wa_um_mean", "Wa_um_std",
    "Wq_um_mean", "Wq_um_std",
    "Wp_um_mean", "Wp_um_std",
    "Wv_um_mean", "Wv_um_std",
    "Wz_um_mean", "Wz_um_std",
    "Wsk_mean", "Wsk_std",
    "Wku_mean", "Wku_std",
]
ergebnis = ergebnis[spalten]

print(ergebnis.to_string(index=False))

# CSV speichern
ergebnis.to_csv(out_csv, index=False, sep=";")
print(f"\nCSV gespeichert: {out_csv}")