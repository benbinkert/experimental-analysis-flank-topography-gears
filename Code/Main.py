

from Code.Plots import Filter_Flankenlinie
from Code.Plots import *
from Filter import *
from GeometrischHilfen import *
import matplotlib.pyplot as plt
from Unterprogramme import *
from surfalize import Profile
import csv
from pathlib import Path
import numpy as np
from scipy import signal


def stats(name, arr):
    arr = arr.astype(float)
    print(name, "min/max", np.nanmin(arr), np.nanmax(arr),
          "std", np.nanstd(arr),
          "p1/p99", np.nanpercentile(arr, [1, 99]))
def bandpass_rows_fft(z, step_um, wl_min_um, wl_max_um):
    """
    Zeilenweiser Bandpass in Wellenlängen [wl_min, wl_max] (µm) entlang x.
    """
    z = z.astype(float)
    ny, nx = z.shape
    out = np.zeros_like(z)

    # Frequenzen in cycles/µm
    f = np.fft.rfftfreq(nx, d=step_um)
    wl = np.where(f > 0, 1.0 / f, np.inf)  # µm

    mask = (wl >= wl_min_um) & (wl <= wl_max_um)   # keep only this band
    mask[0] = False  # DC weg

    win = signal.windows.hann(nx)

    for i in range(ny):
        row = z[i, :]
        row = signal.detrend(row, type="linear")
        row = row * win
        R = np.fft.rfft(row)
        R[~mask] = 0.0
        out[i, :] = np.fft.irfft(R, n=nx)

    return out

def main():
    from surfalize import Surface
    from scipy.io import loadmat
    """
    plot_wsp00_wsp03_onecol_S1S2(
        luecke_label="L14",
        paths={
            "WSP00_S1": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben/WSP00_L14_S1.nms",
            "WSP03_S1": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP03/ErodierteProben/WSP03_L14_S1.nms",
            "WSP00_S2": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben/WSP00_L14_S2.nms",
            "WSP03_S2": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP03/ErodierteProben/WSP03_L14_S2.nms",
        },
        suptitle_extra="Setup Gesamtbild",
        preprocess_fn=load_preprocess_Gesamtbild

    )

    plot_wsp00_wsp03_onecol_S1S2(
        luecke_label="L13",
        paths={
            "WSP00_S1": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben/WSP00_L13_S1.nms",
            "WSP03_S1": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP03/ErodierteProben/WSP03_L13_S1.nms",
            "WSP00_S2": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben/WSP00_L13_S2.nms",
            "WSP03_S2": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP03/ErodierteProben/WSP03_L13_S2.nms",
        },
        suptitle_extra="Setup Gesamtbild",
        preprocess_fn=load_preprocess_Gesamtbild

    )

    
    plot_wsp00_wsp03_onecol_S1S2(
        luecke_label="L14",
        paths={
            "WSP00_S1": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben/WSP00_L14_S1.nms",
            "WSP03_S1": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP03/ErodierteProben/WSP03_L14_S1.nms",
            "WSP00_S2": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben/WSP00_L14_S2.nms",
            "WSP03_S2": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP03/ErodierteProben/WSP03_L14_S2.nms",
        },
        suptitle_extra="Setup Prozessfrei",
        preprocess_fn=load_preprocess_Prozessfrei,
        preprocess_kwargs={"cutoff_s": 1.6, "cutoff_l": 100.0}
    )

    plot_wsp00_wsp03_onecol_S1S2(
        luecke_label="L13",
        paths={
            "WSP00_S1": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben/WSP00_L13_S1.nms",
            "WSP03_S1": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP03/ErodierteProben/WSP03_L13_S1.nms",
            "WSP00_S2": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben/WSP00_L13_S2.nms",
            "WSP03_S2": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP03/ErodierteProben/WSP03_L13_S2.nms",
        },
        suptitle_extra="Setup Prozessfrei",
        preprocess_fn=load_preprocess_Prozessfrei,
        preprocess_kwargs={"cutoff_s": 1.6, "cutoff_l": 100.0}
    )
    
        x_um, z_um, dx_um = load_perthometer_prf_txt(
        "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Mahr XCR20/WSP00_L1_S1_1.txt")
    
    length_um = float(x_um[-1] - x_um[0])

    p = Profile(
        height_data=z_um,
        step=dx_um,  # <- direkt nutzen
        length_um=length_um,
        axis_data=x_um,
        axis_label="x [µm]",
        title="Perthometer Profil"
    )

    p.show_real(False)
    profile = p.level()
    #profile.show_real()
    profile = profile.detrend_polynomial(degree=2)
    #profile.show_real()

    # Originale Signale
    x_um = p.axis_data  # echte x-Achse in µm
    z_um = p.data  # Höhen in µm

    # Detrend anwenden (so wie du es machst)
    p_level = p.level()  # degree=0
    p_det2 = p_level.detrend_polynomial(2)  # degree=2

    x = x_um
    z = z_um

    coeff = np.polyfit(x, z, 2)
    trend = np.polyval(coeff, x)

    plt.figure(figsize=(10, 4))
    plt.plot(x, z, label="Originalprofil")
    plt.plot(x, trend, label="Polynomfit 2. Grades")
    plt.xlabel("x [µm]")
    plt.ylabel("z [µm]")
    plt.title("Originalprofil mit angepasstem Trend 2. Grades")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
  

    plot_messsystemvergleich_onecol_S1S2(
        luecke_label="L14",
        paths={
            "Nanofocus_S1": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben/WSP00_L14_S1.nms",
            "Nanofocus_S2": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben/WSP00_L14_S2.nms",
            "Keyence_S1": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP00/1x7/WSP00_L14_S1.sdf",
            "Keyence_S2": "//Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP00/1x7/WSP00_L14_S2.sdf",
        },
        suptitle_extra="Setup Messsystemvergleich",
        preprocess_fn=load_preprocess_Vergleich,
        preprocess_kwargs={"cutoff_s": 16 ,"cutoff_l": 400},
        scale_mode="per_surface"
    )

    plot_messsystemvergleich_onecol_S1S2(
        luecke_label="L13",
        paths={
            "Nanofocus_S1": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben/WSP00_L13_S1.nms",
            "Nanofocus_S2": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben/WSP00_L13_S2.nms",
            "Keyence_S1": "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP00/1x7/WSP00_L13_S1.sdf",
            "Keyence_S2": "//Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP00/1x7/WSP00_L13_S2.sdf",
        },
        suptitle_extra="Setup Messsystemvergleich",
        preprocess_fn=load_preprocess_Vergleich,
        preprocess_kwargs={"cutoff_s": 16 ,"cutoff_l": 400},
        scale_mode="per_surface"
    )
    
    path2 = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben/WSP00_L14_S2.nms"
    surface2 = Surface.load(path2)
    surface2.show(False)
    path = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP00/1x7/Ergebnisse/WSP00_L14_S2.sdf"
    surface = Surface.load(path)
    surface.show()
    surface = surface.level()
    surface.show()
    surface = surface.detrend_polynomial(degree=2)
    surface.show()
    surface = surface.threshold(threshold=(0.5, 0.5))
    surface.show()
    surface = surface.fill_nonmeasured(method="nearest")
    surface.show()
    surface_lowpass = surface.filter(filter_type="bandpass", cutoff= 16 , cutoff2=400)
    print("a")
    surface_lowpass.show()
    """

    # Pfade anpassen, falls nötig
    path = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben/WSP00_L14_S2.nms"
    path2 = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP00/1x7/ErgebnisseMatInSDF/WSP00_L14_S2.sdf"
    path3 = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Keyence/WSP03/Refelktion/Ergebnisse/WSP03_L13_S2.sdf"
    path4 = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP03/ErodierteProben/WSP03_L13_S2.nms"
    """
    surface4 = Surface.load(path4)
    surface4 = surface4.level()
    surface4 = surface4.detrend_polynomial(degree=2)
    surface4 = surface4.threshold(threshold=(0.5, 0.5))
    surface4 = surface4.fill_nonmeasured(method="nearest")
    surface4 = surface4.filter(filter_type="lowpass", cutoff= 16)
    surface4.show(False)

    surface3 = Surface.load(path3)
    surface3 = surface3.crop((0,surface3.width_um,600,surface3.height_um))
    surface3 = surface3.level()
    surface3 = surface3.detrend_polynomial(degree=2)
    surface3 = surface3.threshold(threshold=(0.5, 0.5))
    surface3 = surface3.fill_nonmeasured(method="nearest")
    surface3 = surface3.filter(filter_type="lowpass", cutoff= 16)
    surface3.show()
    
    surfaceNano = Surface.load(path)
    surfaceNano = surfaceNano.level()
    surfaceNano = surfaceNano.detrend_polynomial(degree=2)
    surfaceNano = surfaceNano.threshold(threshold=(0.5, 0.5))
    surfaceNano = surfaceNano.fill_nonmeasured(method="nearest")
    surfaceNano_lowpass = surfaceNano.filter(filter_type="lowpass", cutoff= 50)
    surfaceNano_lowpass.show(False)
    2

INPUT_CSV = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Mahr XCR20/Ergebnisse/perthometer_nanofocus_R_W.csv"
OUTPUT_CSV = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Mahr XCR20/Ergebnisse/perthometer_nanofocus_R_W_vergleich.csv"


def to_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return float(value)


def safe_rel_diff(nano, perth):
    if perth is None or nano is None:
        return None
    if perth == 0:
        return None
    return (nano - perth) / perth


def safe_factor(nano, perth):
    if perth is None or nano is None:
        return None
    if perth == 0:
        return None
    return nano / perth


def safe_abs_diff(nano, perth):
    if perth is None or nano is None:
        return None
    return nano - perth


def flank_label_from_profile_id(profile_id: str):
    if profile_id.endswith("_S1"):
        return "Rechte Flanke"
    if profile_id.endswith("_S2"):
        return "Linke Flanke"
    return ""


def parse_profile_id(profile_id: str):
    # z. B. WSP00_L13_S1
    parts = profile_id.split("_")
    out = {
        "wsp": "",
        "luecke": "",
        "seite": "",
    }
    if len(parts) == 3:
        out["wsp"] = parts[0].replace("WSP", "WSP ")
        out["luecke"] = parts[1].replace("L", "")
        out["seite"] = parts[2]
    return out
input_path = Path(INPUT_CSV)
output_path = Path(OUTPUT_CSV)
output_path.parent.mkdir(parents=True, exist_ok=True)

with input_path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Gruppieren nach (profil_id, anteil)
grouped = {}
for row in rows:
    key = (row["profil_id"], row["anteil"])
    grouped.setdefault(key, {})
    grouped[key][row["messsystem"]] = row

metrics = [
    "Ra_um", "Rq_um", "Rp_um", "Rv_um", "Rz_um", "Rsk", "Rku",
    "Wa_um", "Wq_um", "Wp_um", "Wv_um", "Wz_um", "Wsk", "Wku"
]

out_rows = []

for (profil_id, anteil), systems in sorted(grouped.items()):
    perth = systems.get("Perthometer")
    nano = systems.get("NanoFocus")

    if perth is None or nano is None:
        continue

    parsed = parse_profile_id(profil_id)

    out_row = {
        "profil_id": profil_id,
        "wsp": parsed["wsp"],
        "luecke": parsed["luecke"],
        "seite": parsed["seite"],
        "flanke": flank_label_from_profile_id(profil_id),
        "anteil": anteil,
        "Nis_um": perth.get("Nis_um", ""),
        "Nic_um": perth.get("Nic_um", ""),
    }

    for metric in metrics:
        perth_val = to_float(perth.get(metric, ""))
        nano_val = to_float(nano.get(metric, ""))

        # Nur die Kennwerte berechnen, die in der jeweiligen Zeile auch existieren
        if perth_val is None and nano_val is None:
            continue

        out_row[f"{metric}_Perthometer"] = perth_val
        out_row[f"{metric}_NanoFocus"] = nano_val
        out_row[f"{metric}_diff_abs"] = safe_abs_diff(nano_val, perth_val)
        out_row[f"{metric}_diff_rel"] = safe_rel_diff(nano_val, perth_val)
        out_row[f"{metric}_factor"] = safe_factor(nano_val, perth_val)

    out_rows.append(out_row)

# Feldnamen sammeln
fieldnames = []
base_fields = ["profil_id", "wsp", "luecke", "seite", "flanke", "anteil", "Nis_um", "Nic_um"]
fieldnames.extend(base_fields)

for metric in metrics:
    fieldnames.extend([
        f"{metric}_Perthometer",
        f"{metric}_NanoFocus",
        f"{metric}_diff_abs",
        f"{metric}_diff_rel",
        f"{metric}_factor",
    ])

with output_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(out_rows)

print(f"Fertig. Vergleichs-CSV gespeichert unter:\n{output_path}")

    """
    #plot_WSP00_WSP03_15mu("")

    filepath = '/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Simulation/WSP00/WST_WSP00_L1_HighResolution_2_Rechts.sdf'
    s00 = Surface.load(filepath)
    s00 = s00.level()
    s00 = Surface.detrend_polynomial(s00, 2)
    s00 = s00.threshold(threshold=(0.25, 0.25))
    s00 = s00.fill_nonmeasured_rowwise_linear()
    s00 = s00.filter(filter_type='lowpass', cutoff=1.6)
    s00 = s00.fill_nonmeasured_rowwise_linear()
    s00.show()

    # ROI für WSP00 (in µm)
    x_min00, x_max00, y_min00, y_max00 = (s00.width_um/2 -50, s00.width_um/2+50, s00.height_um/2-50, s00.height_um/2+50)
    s00_roi = s00.crop((x_min00, x_max00, y_min00, y_max00))
    #s00_roi.show()

if __name__ == "__main__":
    main()
