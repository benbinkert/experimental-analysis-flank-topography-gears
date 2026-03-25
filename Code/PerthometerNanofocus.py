from pathlib import Path
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from surfalize import Surface


# ============================================================
# Einfache Profilverarbeitung für Perthometer- und NanoFocus-Daten
#
# Was kann man damit machen?
# - Perthometer-Profile aus TXT-Dateien einlesen
# - Passende horizontale NanoFocus-Profile ausschneiden
# - Beide Systeme auf dieselbe Profillänge bringen
# - Vorverarbeitung, Rauheits- und Welligkeitstrennung durchführen
# - Kennwerte berechnen und als CSV speichern
# - Direkt Vergleichsplots für Perthometer vs. NanoFocus erzeugen
# ============================================================


# ============================================================
# 1) EINFACHE PROFILKLASSE - OHNE surfalize.Profile
# ============================================================

class SimpleProfile:
    """
    Einfache Profilklasse für 1D-Profile mit x- und z-Werten.

    Was kann man damit machen?
    - 1D-Profile unabhängig von surfalize.Profile speichern
    - Grundlegende Kennwerte wie Ra, Rq, Rz usw. berechnen
    - Profile direkt plotten und kopieren
    """
    def __init__(self, z_um, x_um, title=""):
        self.z = np.asarray(z_um, dtype=float)
        self.x = np.asarray(x_um, dtype=float)
        self.title = title

        if self.z.ndim != 1 or self.x.ndim != 1:
            raise ValueError("x und z müssen 1D-Arrays sein.")
        if len(self.z) != len(self.x):
            raise ValueError("x und z müssen gleich lang sein.")
        if len(self.x) < 2:
            raise ValueError("Profil braucht mindestens 2 Punkte.")

    @property
    def step_um(self):
        """
        Typische Schrittweite des Profils in µm.

        Was kann man damit machen?
        - Die Abtastung des Profils bestimmen
        - Filterparameter in Pixel umrechnen
        """
        return float(np.median(np.diff(self.x)))

    @property
    def length_um(self):
        """
        Profillänge in µm.

        Was kann man damit machen?
        - Profile verschiedener Systeme auf dieselbe Länge bringen
        - Vergleichsbereiche prüfen
        """
        return float(self.x[-1] - self.x[0])

    def copy(self, title=None):
        """
        Erstellt eine Kopie des Profils.

        Was kann man damit machen?
        - Verarbeitungsschritte durchführen, ohne das Original zu verändern
        """
        return SimpleProfile(
            z_um=self.z.copy(),
            x_um=self.x.copy(),
            title=self.title if title is None else title
        )

    def centered(self):
        """
        Zentriert das Profil auf Mittelwert 0.

        Was kann man damit machen?
        - Höhenoffset entfernen
        - Profile besser vergleichbar machen
        """
        z = self.z - np.nanmean(self.z)
        return SimpleProfile(z, self.x, self.title)

    # -------- Kennwerte --------
    def Ra(self):
        """
        Mittlere arithmetische Abweichung.

        Was kann man damit machen?
        - Klassischen Rauheitsmittelwert berechnen
        """
        z = self.z - np.nanmean(self.z)
        return float(np.nanmean(np.abs(z)))

    def Rq(self):
        """
        Quadratischer Mittenrauwert.

        Was kann man damit machen?
        - Energieähnliche Streuung des Profils berechnen
        """
        z = self.z - np.nanmean(self.z)
        return float(np.sqrt(np.nanmean(z ** 2)))

    def Rp(self):
        """
        Maximale Profilspitze relativ zum Mittelwert.

        Was kann man damit machen?
        - Größte positive Auslenkung im Profil bestimmen
        """
        z = self.z - np.nanmean(self.z)
        return float(np.nanmax(z))

    def Rv(self):
        """
        Maximale Profiltiefe relativ zum Mittelwert.

        Was kann man damit machen?
        - Größte negative Auslenkung im Profil bestimmen
        """
        z = self.z - np.nanmean(self.z)
        return float(abs(np.nanmin(z)))

    def Rz(self):
        """
        Gesamthöhe des Profils als Rp + Rv.

        Was kann man damit machen?
        - Peak-to-valley-artigen Kennwert berechnen
        """
        return self.Rp() + self.Rv()

    def Rsk(self):
        """
        Schiefe des Profils.

        Was kann man damit machen?
        - Beurteilen, ob Spitzen oder Täler überwiegen
        """
        rq = self.Rq()
        if rq == 0:
            return np.nan
        z = self.z - np.nanmean(self.z)
        return float(np.nanmean(z ** 3) / (rq ** 3))

    def Rku(self):
        """
        Kurtosis des Profils.

        Was kann man damit machen?
        - Einschätzen, wie spitz oder flach die Verteilung ist
        """
        rq = self.Rq()
        if rq == 0:
            return np.nan
        z = self.z - np.nanmean(self.z)
        return float(np.nanmean(z ** 4) / (rq ** 4))

    # W-Kennwerte identisch berechnet, nur anders benannt
    def Wa(self): return self.Ra()
    def Wq(self): return self.Rq()
    def Wp(self): return self.Rp()
    def Wv(self): return self.Rv()
    def Wz(self): return self.Rz()
    def Wsk(self): return self.Rsk()
    def Wku(self): return self.Rku()

    def plot(self, ax=None, lw=0.7, color="k"):
        """
        Zeichnet das Profil.

        Was kann man damit machen?
        - Einzelprofile schnell visualisieren
        - Vergleichsdarstellungen mit Matplotlib erzeugen
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3), dpi=150)

        ax.plot(self.x, self.z, lw=lw, color=color)

        ax.set_xlim(float(self.x[0]), float(self.x[-1]))
        ax.margins(x=0)

        ax.set_xlabel("x [µm]")
        ax.set_ylabel("z [µm]")

        if self.title:
            ax.set_title(self.title)

        return ax


# ============================================================
# 2) PROFIL-OPERATIONEN
# ============================================================

def level_profile(profile: SimpleProfile) -> SimpleProfile:
    """
    Zieht den Mittelwert vom Profil ab.

    Was kann man damit machen?
    - Höhenoffset entfernen
    - Ein Profil auf 0 mitteln
    """
    z = profile.z - np.nanmean(profile.z)
    return SimpleProfile(z, profile.x, profile.title)


def detrend_profile_polynomial(profile: SimpleProfile, degree=2) -> SimpleProfile:
    """
    Entfernt einen polynomialen Trend aus dem Profil.

    Was kann man damit machen?
    - Langsame Formanteile oder Krümmungen herausrechnen
    - Das Profil auf Rauheits-/Welligkeitsanalyse vorbereiten

    Parameter:
    - degree:
      Polynomgrad des Trends
    """
    x = profile.x
    z = profile.z

    mask = np.isfinite(z)
    if np.count_nonzero(mask) < degree + 1:
        raise ValueError("Zu wenige gültige Punkte für Polynomfit.")

    x_valid = x[mask]
    z_valid = z[mask]

    # x normieren für stabileren Polynomfit
    x0 = x_valid.mean()
    x_norm = x_valid - x0
    denom = np.max(np.abs(x_norm))
    if denom == 0:
        raise ValueError("x ist konstant.")
    x_norm /= denom

    coeffs = np.polyfit(x_norm, z_valid, deg=degree)
    x_all = (x - x0) / denom
    trend = np.polyval(coeffs, x_all)

    z_det = np.where(np.isfinite(z), z - trend, np.nan)
    return SimpleProfile(z_det, x, profile.title)


def threshold_percentile(profile: SimpleProfile, lower=0.25, upper=0.25) -> SimpleProfile:
    """
    Entfernt extreme Werte über untere und obere Perzentile.

    Was kann man damit machen?
    - Ausreißer im Profil maskieren
    - Extremspitzen oder extreme Täler vor dem Füllen ausschließen
    """
    z = profile.z.copy()
    mask = np.isfinite(z)
    z_valid = z[mask]

    if z_valid.size == 0:
        raise ValueError("Keine gültigen Punkte vorhanden.")

    low_val = np.percentile(z_valid, lower)
    high_val = np.percentile(z_valid, 100.0 - upper)

    z[z < low_val] = np.nan
    z[z > high_val] = np.nan

    return SimpleProfile(z, profile.x, profile.title)


def fill_nonmeasured_linear(profile: SimpleProfile) -> SimpleProfile:
    """
    Füllt NaN-Bereiche linear auf.

    Was kann man damit machen?
    - Profil nach Thresholding wieder vollständig machen
    - Filterung auf lückenfreien Daten ermöglichen
    """
    z = profile.z.copy()
    x = profile.x

    mask = np.isfinite(z)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Zu wenige gültige Punkte zum Interpolieren.")

    z[~mask] = np.interp(x[~mask], x[mask], z[mask])
    return SimpleProfile(z, x, profile.title)


def preprocess_profile(profile: SimpleProfile,
                       do_level=True,
                       do_detrend=True,
                       degree=2,
                       do_threshold=True,
                       threshold_lower=0.25,
                       threshold_upper=0.25,
                       do_fill=True) -> SimpleProfile:
    """
    Führt die komplette Vorverarbeitung eines Profils aus.

    Was kann man damit machen?
    - Mehrere Verarbeitungsschritte in fester Reihenfolge anwenden
    - Perthometer- und NanoFocus-Profile einheitlich vorbereiten

    Enthaltene Schritte:
    - Leveln
    - Detrending
    - Thresholding
    - lineares Füllen
    """
    out = profile.copy()

    if do_level:
        out = level_profile(out)

    if do_detrend:
        out = detrend_profile_polynomial(out, degree=degree)

    if do_threshold:
        out = threshold_percentile(
            out,
            lower=threshold_lower,
            upper=threshold_upper
        )

    if do_fill:
        out = fill_nonmeasured_linear(out)

    return out


# ============================================================
# 3) GAUSS-FILTER FÜR PROFILE
# ============================================================

def _sigma_from_cutoff_um(cutoff_um: float, step_um: float) -> float:
    """
    Rechnet eine Gauß-Cutoff-Wellenlänge in eine Gauß-Sigma in Pixel um.

    Was kann man damit machen?
    - Filterparameter von µm in Pixel übertragen
    - Lowpass-/Highpass-Filter für Profile parametrisieren
    """
    sigma_um = cutoff_um * np.sqrt(np.log(2.0)) / (2.0 * np.pi)
    sigma_px = sigma_um / step_um
    return max(float(sigma_px), 0.5)


def gaussian_lowpass(profile: SimpleProfile, cutoff_um: float) -> SimpleProfile:
    """
    Wendet einen Gauß-Lowpass auf ein Profil an.

    Was kann man damit machen?
    - Langsame Anteile bzw. Welligkeit extrahieren
    - Kurzwellige Anteile unterdrücken
    """
    sigma_px = _sigma_from_cutoff_um(cutoff_um, profile.step_um)
    z_lp = gaussian_filter1d(profile.z, sigma=sigma_px, mode="nearest")
    return SimpleProfile(z_lp, profile.x, profile.title)


def gaussian_highpass(profile: SimpleProfile, cutoff_um: float) -> SimpleProfile:
    """
    Wendet einen Gauß-Highpass auf ein Profil an.

    Was kann man damit machen?
    - Langwellige Anteile entfernen
    - Kurzwellige Rauheitsanteile isolieren
    """
    lp = gaussian_lowpass(profile, cutoff_um)
    z_hp = profile.z - lp.z
    return SimpleProfile(z_hp, profile.x, profile.title)


def split_roughness_waviness(profile: SimpleProfile, nis_um=2.5, nic_um=800.0):
    """
    Trennt ein Profil in S-gefiltertes Profil, Rauheit und Welligkeit.

    Was kann man damit machen?
    - Kurzwellige Störanteile unter Nis entfernen
    - Welligkeit als langwelligen Anteil > Nic extrahieren
    - Rauheit als Restanteil unterhalb Nic bestimmen

    Rückgabe:
    - profile_s:
      S-gefiltertes Profil
    - roughness:
      Rauheitsprofil
    - waviness:
      Welligkeitsprofil
    """
    # S-Filter: sehr kurze Wellenlängen entfernen
    profile_s = gaussian_lowpass(profile, cutoff_um=nis_um)

    # Welligkeit: langwelliger Anteil
    waviness = gaussian_lowpass(profile_s, cutoff_um=nic_um)

    # Rauheit: Restprofil
    roughness = SimpleProfile(
        z_um=profile_s.z - waviness.z,
        x_um=profile_s.x,
        title=profile_s.title
    )

    return profile_s, roughness, waviness


def profile_metrics_dict(profile: SimpleProfile, prefix="R"):
    """
    Baut ein Dictionary mit Profilkennwerten auf.

    Was kann man damit machen?
    - Rauheits- oder Welligkeitskennwerte kompakt exportieren
    - CSV-Zeilen für spätere Auswertung erzeugen

    Parameter:
    - prefix='R':
      Rauheitskennwerte
    - prefix='W':
      Welligkeitskennwerte
    """
    if prefix.upper() == "R":
        return {
            "Ra_um": profile.Ra(),
            "Rq_um": profile.Rq(),
            "Rp_um": profile.Rp(),
            "Rv_um": profile.Rv(),
            "Rz_um": profile.Rz(),
            "Rsk": profile.Rsk(),
            "Rku": profile.Rku(),
        }
    elif prefix.upper() == "W":
        return {
            "Wa_um": profile.Wa(),
            "Wq_um": profile.Wq(),
            "Wp_um": profile.Wp(),
            "Wv_um": profile.Wv(),
            "Wz_um": profile.Wz(),
            "Wsk": profile.Wsk(),
            "Wku": profile.Wku(),
        }
    else:
        raise ValueError("prefix muss 'R' oder 'W' sein.")


# ============================================================
# 4) EINLESEN PERTHOMETER
# ============================================================

def load_perthometer_prf_txt(path_txt: str):
    """
    Liest eine Perthometer-TXT-Datei mit [PROFILE_VALUES] ein.

    Was kann man damit machen?
    - Exportierte Perthometer-Profile in x- und z-Koordinaten umwandeln
    - x und z von mm nach µm umrechnen
    - Ein einheitliches 1D-Profil für die weitere Verarbeitung erzeugen

    Rückgabe:
    - x_um:
      x-Achse in µm
    - z_um:
      Höhenwerte in µm
    - dx_um:
      typische Schrittweite in µm
    """
    path = Path(path_txt)

    data_started = False
    xs = []
    zs = []

    with path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # Start des Datenblocks erkennen
            if line.startswith("[PROFILE_VALUES]"):
                data_started = True
                continue
            if not data_started:
                continue

            # Kommentare und ungeeignete Zeilen überspringen
            if not line or line.startswith("//"):
                continue
            if "=" not in line:
                continue

            _, right = line.split("=", 1)
            parts = right.split()
            if len(parts) < 3:
                continue

            x = float(parts[0])   # mm
            z = float(parts[2])   # mm
            xs.append(x)
            zs.append(z)

    x = np.asarray(xs, dtype=float)
    z = np.asarray(zs, dtype=float)

    if x.size < 2:
        raise ValueError(f"Zu wenige Punkte im [PROFILE_VALUES]-Block gefunden: {path_txt}")

    # x relativ ab 0 setzen und in µm umrechnen
    x_um = (x - x[0]) * 1000.0
    z_um = z * 1000.0
    dx_um = float(np.median(np.diff(x_um)))

    return x_um, z_um, dx_um


def load_perthometer_groups(folder):
    """
    Liest mehrere Perthometer-Dateien ein und gruppiert sie nach Profil-ID.

    Was kann man damit machen?
    - Wiederholungsmessungen zu einer gemeinsamen Gruppe zusammenfassen
    - Aus Dateinamen wie WSP00_L1_S1_1, WSP00_L1_S1_2 usw. Profilgruppen bilden
    - Später Mittelprofile über Wiederholungen berechnen

    Rückgabe:
    - Dictionary mit gruppierten Profilen
    """
    folder = Path(folder)
    pattern = re.compile(r"^(WSP\d+_L\d+_S\d+)_(\d+)$")

    raw_groups = {}

    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue

        m = pattern.match(path.stem)
        if not m:
            continue

        group_key = m.group(1)
        repetition = int(m.group(2))

        x_um, z_um, dx_um = load_perthometer_prf_txt(str(path))

        raw_groups.setdefault(group_key, []).append({
            "repetition": repetition,
            "file": str(path),
            "x_um": x_um,
            "z_um": z_um,
            "dx_um": dx_um,
        })

    grouped = {}

    for key, entries in raw_groups.items():
        entries.sort(key=lambda e: e["repetition"])

        dx_values = np.array([e["dx_um"] for e in entries], dtype=float)
        dx_ref = float(np.median(dx_values))

        # Kürzeste Messung bestimmt die gemeinsame Länge
        n_min = min(len(e["z_um"]) for e in entries)

        z_list = []
        files = []

        for e in entries:
            z_list.append(e["z_um"][:n_min])
            files.append(e["file"])

        x_ref_um = np.arange(n_min, dtype=float) * dx_ref

        grouped[key] = {
            "x_um": x_ref_um,
            "profiles_z_um": z_list,
            "dx_um": dx_ref,
            "files": files,
            "n_points": n_min,
        }

    return grouped


# ============================================================
# 5) NANOFOCUS LADEN UND PASSENDES PROFIL AUSSCHNEIDEN
# ============================================================

def load_nanofocus_surfaces(folder):
    """
    Lädt alle NanoFocus-.nms-Dateien eines Ordners.

    Was kann man damit machen?
    - Alle verfügbaren NanoFocus-Oberflächen auf einmal einlesen
    - Über Dateinamen passend zu Perthometer-Profilen zuordnen
    """
    folder = Path(folder)
    surfaces = {}

    for path in sorted(folder.glob("*.nms")):
        surfaces[path.stem] = Surface.load(str(path))

    return surfaces


def extract_nanofocus_horizontal_profile_matching_perthometer(
    surface: Surface,
    perthometer_x_um: np.ndarray,
    y_um=None,
    center_crop_x=True
):
    """
    Schneidet aus einer NanoFocus-Oberfläche ein horizontales Profil aus,
    das zur Perthometer-Profillänge passt.

    Was kann man damit machen?
    - Direkt vergleichbare NanoFocus-Profile für Perthometer-Messungen erzeugen
    - Ein zentrales horizontales Profil auf dieselbe Länge wie das Perthometer-Profil bringen

    Parameter:
    - y_um:
      Höhe des horizontalen Schnitts
    - center_crop_x:
      Wenn True, wird der x-Bereich zentriert ausgeschnitten

    Rückgabe:
    - Dictionary mit x/z-Profil und Metadaten
    """
    perthometer_length_um = float(perthometer_x_um[-1] - perthometer_x_um[0])
    target_length_um = min(perthometer_length_um, surface.width_um)

    if y_um is None:
        y_um = surface.height_um / 2.0

    if center_crop_x:
        x0_um = (surface.width_um - target_length_um) / 2.0
    else:
        x0_um = 0.0
    x1_um = x0_um + target_length_um

    profile_nf = surface.get_horizontal_profile(y=y_um)

    x_nf_um = np.arange(profile_nf.data.size, dtype=float) * profile_nf.step

    mask = (x_nf_um >= x0_um) & (x_nf_um <= x1_um)
    x_nf_crop_um = x_nf_um[mask] - x0_um
    z_nf_crop_um = profile_nf.data[mask]

    if x_nf_crop_um.size < 2:
        raise ValueError("Zu wenige Punkte im ausgeschnittenen NanoFocus-Profil.")

    return {
        "x_um": x_nf_crop_um,
        "z_um": z_nf_crop_um,
        "y_um": y_um,
        "x0_um": x0_um,
        "x1_um": x1_um,
        "dx_um": float(np.median(np.diff(x_nf_crop_um))),
    }


def resample_perthometer_group_to_length(perth_group, max_length_um):
    """
    Kürzt eine Perthometer-Gruppe auf eine maximale Länge.

    Was kann man damit machen?
    - Perthometer-Wiederholungen auf denselben x-Bereich begrenzen
    - Später direkt mit NanoFocus vergleichen
    """
    x_um = perth_group["x_um"]
    dx_um = perth_group["dx_um"]

    n_keep = min(len(x_um), int(np.floor(max_length_um / dx_um)) + 1)
    x_new = x_um[:n_keep]
    z_list_new = [z[:n_keep] for z in perth_group["profiles_z_um"]]

    return {
        "x_um": x_new,
        "profiles_z_um": z_list_new,
        "dx_um": dx_um,
        "files": perth_group["files"],
        "n_points": n_keep,
    }


def interpolate_nanofocus_to_x(nf_profile, x_target_um):
    """
    Interpoliert ein NanoFocus-Profil auf eine Ziel-x-Achse.

    Was kann man damit machen?
    - NanoFocus und Perthometer auf exakt dieselben x-Stützstellen bringen
    - Punktweise Vergleichbarkeit herstellen
    """
    x_nf = nf_profile["x_um"]
    z_nf = nf_profile["z_um"]

    max_x = min(x_target_um[-1], x_nf[-1])
    mask = x_target_um <= max_x
    x_use = x_target_um[mask]

    z_interp = np.interp(x_use, x_nf, z_nf)

    return {
        "x_um": x_use,
        "z_um": z_interp,
    }


def truncate_perthometer_group_to_x(perth_group, x_target_um):
    """
    Kürzt eine Perthometer-Gruppe auf die Länge einer Ziel-x-Achse.

    Was kann man damit machen?
    - Nach der Interpolation beide Systeme auf exakt dieselbe Punktzahl bringen
    """
    n = len(x_target_um)
    return {
        "x_um": perth_group["x_um"][:n],
        "profiles_z_um": [z[:n] for z in perth_group["profiles_z_um"]],
        "dx_um": perth_group["dx_um"],
        "files": perth_group["files"],
        "n_points": n,
    }


def mean_profile_from_list(profile_list, title=None):
    """
    Bildet das Mittelprofil aus mehreren Profilen.

    Was kann man damit machen?
    - Wiederholungsmessungen des Perthometers mitteln
    - Ein repräsentatives Vergleichsprofil erzeugen
    """
    x_um = profile_list[0].x
    z_stack = np.vstack([p.z for p in profile_list])
    z_mean = np.mean(z_stack, axis=0)

    return SimpleProfile(
        z_um=z_mean,
        x_um=x_um,
        title=title or ""
    )


# ============================================================
# 6) KOMBINIERTE VERARBEITUNG PERTHOMETER + NANOFOCUS
# ============================================================

def build_combined_perthometer_nanofocus(
    perth_folder: str,
    nanofocus_folder: str,
    y_mode: str = "center",
    nis_um: float = 2.5,
    nic_um: float = 800.0
):
    """
    Baut die komplette kombinierte Datengrundlage aus Perthometer und NanoFocus auf.

    Was kann man damit machen?
    - Passende Profile aus beiden Systemen zusammenführen
    - Vorverarbeitete, Rauheits- und Welligkeitsprofile erzeugen
    - Eine gemeinsame Basis für Plots und CSV-Export schaffen

    Rückgabe:
    - Dictionary mit allen kombinierten Profilen und Metadaten
    """
    perth_groups = load_perthometer_groups(perth_folder)
    nanofocus_surfaces = load_nanofocus_surfaces(nanofocus_folder)

    combined = {}

    for key, perth in perth_groups.items():
        if key not in nanofocus_surfaces:
            print(f"Warnung: Keine passende NanoFocus-Datei für {key} gefunden.")
            continue

        surface = nanofocus_surfaces[key]

        if y_mode == "center":
            y_um = surface.height_um / 2.0
        else:
            y_um = surface.height_um / 2.0

        # Passendes NanoFocus-Profil ausschneiden
        nf_profile_raw = extract_nanofocus_horizontal_profile_matching_perthometer(
            surface=surface,
            perthometer_x_um=perth["x_um"],
            y_um=y_um,
            center_crop_x=True
        )

        # Perthometer auf Länge des NanoFocus-Profils begrenzen
        perth_cut = resample_perthometer_group_to_length(
            perth,
            max_length_um=nf_profile_raw["x_um"][-1]
        )

        # NanoFocus auf dieselbe x-Achse interpolieren
        nf_interp = interpolate_nanofocus_to_x(
            nf_profile_raw,
            perth_cut["x_um"]
        )

        # Perthometer ebenfalls auf dieselbe Punktzahl kürzen
        perth_final = truncate_perthometer_group_to_x(
            perth_cut,
            nf_interp["x_um"]
        )

        # Perthometer-Wiederholungen als SimpleProfile anlegen
        perth_profiles = []
        for i, z in enumerate(perth_final["profiles_z_um"], start=1):
            perth_profiles.append(
                SimpleProfile(
                    z_um=z,
                    x_um=perth_final["x_um"],
                    title=f"{key} – Perthometer {i}"
                )
            )

        # NanoFocus-Profil anlegen
        nf_profile = SimpleProfile(
            z_um=nf_interp["z_um"],
            x_um=nf_interp["x_um"],
            title=f"{key} – NanoFocus"
        )

        # Vorverarbeitung auf Perthometer-Wiederholungen
        perth_profiles_processed = [
            preprocess_profile(
                p,
                do_level=True,
                do_detrend=True,
                degree=2,
                do_threshold=True,
                threshold_upper=0.25,
                threshold_lower=0.25,
                do_fill=True
            )
            for p in perth_profiles
        ]

        # Vorverarbeitung auf NanoFocus
        nf_profile_processed = preprocess_profile(
            nf_profile,
            do_level=True,
            do_detrend=True,
            degree=2,
            do_threshold=True,
            threshold_upper=0.25,
            threshold_lower=0.25,
            do_fill=True
        )

        # Mittelprofil über Perthometer-Wiederholungen
        perth_mean_processed = mean_profile_from_list(
            perth_profiles_processed,
            title=f"{key} – Perthometer Mittel"
        )

        # Rauheit/Welligkeit trennen
        _, perth_roughness, perth_waviness = split_roughness_waviness(
            perth_mean_processed,
            nis_um=nis_um,
            nic_um=nic_um
        )

        _, nf_roughness, nf_waviness = split_roughness_waviness(
            nf_profile_processed,
            nis_um=nis_um,
            nic_um=nic_um
        )

        combined[key] = {
            "perthometer_raw": perth_profiles,
            "nanofocus_raw": nf_profile,
            "perthometer_processed": perth_profiles_processed,
            "nanofocus_processed": nf_profile_processed,
            "perthometer_mean_processed": perth_mean_processed,
            "perthometer_roughness": perth_roughness,
            "perthometer_waviness": perth_waviness,
            "nanofocus_roughness": nf_roughness,
            "nanofocus_waviness": nf_waviness,
            "nanofocus_meta": nf_profile_raw,
        }

    return combined


# ============================================================
# 7) PLOTS
# ============================================================

def flank_label_from_key(key: str):
    """
    Wandelt den S1/S2-Teil eines Keys in eine Flankenbezeichnung um.

    Was kann man damit machen?
    - Rechte und linke Flanke sprachlich sauber benennen
    """
    if key.endswith("_S1"):
        return "Rechte Flanke"
    elif key.endswith("_S2"):
        return "Linke Flanke"
    return "Unbekannte Flanke"


def format_plot_title(key: str):
    """
    Erzeugt einen lesbaren Titel aus einer Profil-ID.

    Was kann man damit machen?
    - Aus technischen Dateinamen einen sauberen Plot-Titel erzeugen
    """
    m = re.match(r"^(WSP\d+)_L(\d+)_S([12])$", key)
    if not m:
        return key

    wsp = m.group(1)
    luecke = m.group(2)
    seite = m.group(3)

    wsp_num = wsp.replace("WSP", "")
    flanke = "Rechte Flanke" if seite == "1" else "Linke Flanke"

    return f"WSP {wsp_num}, Lücke {luecke}, {flanke}"


def _common_ylim(p1: SimpleProfile, p2: SimpleProfile, pad_ratio=0.05):
    """
    Bestimmt gemeinsame y-Achsen-Grenzen für zwei Profile.

    Was kann man damit machen?
    - Zwei Profile mit identischer y-Skalierung plotten
    - Vergleichsplots optisch fair machen
    """
    z = np.concatenate([p1.z, p2.z])
    z = z[np.isfinite(z)]

    if z.size == 0:
        return -1.0, 1.0

    zmin = float(np.min(z))
    zmax = float(np.max(z))

    if zmin == zmax:
        pad = 1e-6 if zmin == 0 else abs(zmin) * 0.05
        return zmin - pad, zmax + pad

    pad = (zmax - zmin) * pad_ratio
    return zmin - pad, zmax + pad


def plot_perthometer_vs_nanofocus_rw(combined_data, key, save_path=None):
    """
    Zeichnet Perthometer und NanoFocus für ein Profil in einer 3x2-Vergleichsdarstellung.

    Was kann man damit machen?
    - Vorverarbeitete Profile beider Systeme vergleichen
    - Rauheits- und Welligkeitsanteile direkt gegenüberstellen
    - Abbildungen für Bericht oder Anhang erzeugen
    """
    if key not in combined_data:
        raise KeyError(f"{key} nicht in combined_data enthalten.")

    data = combined_data[key]

    perth_profile = data["perthometer_mean_processed"]
    perth_r = data["perthometer_roughness"]
    perth_w = data["perthometer_waviness"]

    nf_profile = data["nanofocus_processed"]
    nf_r = data["nanofocus_roughness"]
    nf_w = data["nanofocus_waviness"]

    ylim_pre = _common_ylim(perth_profile, nf_profile)
    ylim_r = _common_ylim(perth_r, nf_r)
    ylim_w = _common_ylim(perth_w, nf_w)

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), dpi=150, sharex="col")

    # Vorverarbeitete Profile
    perth_profile.plot(ax=axes[0, 0], color="k", lw=0.6)
    axes[0, 0].set_title("Perthometer – vorverarbeitet")
    axes[0, 0].set_ylim(*ylim_pre)

    nf_profile.plot(ax=axes[0, 1], color="k", lw=0.6)
    axes[0, 1].set_title("NanoFocus – vorverarbeitet")
    axes[0, 1].set_ylim(*ylim_pre)

    # Rauheit
    perth_r.plot(ax=axes[1, 0], color="k", lw=0.6)
    axes[1, 0].set_title("Perthometer – Rauheit R")
    axes[1, 0].set_ylim(*ylim_r)

    nf_r.plot(ax=axes[1, 1], color="k", lw=0.6)
    axes[1, 1].set_title("NanoFocus – Rauheit R")
    axes[1, 1].set_ylim(*ylim_r)

    # Welligkeit
    perth_w.plot(ax=axes[2, 0], color="k", lw=0.6)
    axes[2, 0].set_title("Perthometer – Welligkeit W")
    axes[2, 0].set_ylim(*ylim_w)

    nf_w.plot(ax=axes[2, 1], color="k", lw=0.6)
    axes[2, 1].set_title("NanoFocus – Welligkeit W")
    axes[2, 1].set_ylim(*ylim_w)

    fig.suptitle(format_plot_title(key), fontsize=13)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()


# ============================================================
# 8) CSV
# ============================================================

def save_combined_metrics_csv(combined_data, out_csv_path, nis_um=2.5, nic_um=800.0):
    """
    Speichert berechnete Rauheits- und Welligkeitskennwerte als CSV.

    Was kann man damit machen?
    - Ergebnisse aus Perthometer und NanoFocus in einer Datei sammeln
    - Später tabellarisch oder statistisch weiterverarbeiten
    """
    rows = []

    for key, data in sorted(combined_data.items()):
        perth_r = data["perthometer_roughness"]
        perth_w = data["perthometer_waviness"]
        nf_r = data["nanofocus_roughness"]
        nf_w = data["nanofocus_waviness"]

        # Perthometer Rauheit
        row_r_perth = {
            "profil_id": key,
            "messsystem": "Perthometer",
            "anteil": "R",
            "Nis_um": nis_um,
            "Nic_um": nic_um,
        }
        row_r_perth.update(profile_metrics_dict(perth_r, prefix="R"))
        rows.append(row_r_perth)

        # Perthometer Welligkeit
        row_w_perth = {
            "profil_id": key,
            "messsystem": "Perthometer",
            "anteil": "W",
            "Nis_um": nis_um,
            "Nic_um": nic_um,
        }
        row_w_perth.update(profile_metrics_dict(perth_w, prefix="W"))
        rows.append(row_w_perth)

        # NanoFocus Rauheit
        row_r_nf = {
            "profil_id": key,
            "messsystem": "NanoFocus",
            "anteil": "R",
            "Nis_um": nis_um,
            "Nic_um": nic_um,
        }
        row_r_nf.update(profile_metrics_dict(nf_r, prefix="R"))
        rows.append(row_r_nf)

        # NanoFocus Welligkeit
        row_w_nf = {
            "profil_id": key,
            "messsystem": "NanoFocus",
            "anteil": "W",
            "Nis_um": nis_um,
            "Nic_um": nic_um,
        }
        row_w_nf.update(profile_metrics_dict(nf_w, prefix="W"))
        rows.append(row_w_nf)

    if rows:
        all_fields = set()
        for row in rows:
            all_fields.update(row.keys())

        preferred_order = [
            "profil_id", "messsystem", "anteil", "Nis_um", "Nic_um",
            "Ra_um", "Rq_um", "Rp_um", "Rv_um", "Rz_um", "Rsk", "Rku",
            "Wa_um", "Wq_um", "Wp_um", "Wv_um", "Wz_um", "Wsk", "Wku"
        ]

        fieldnames = [f for f in preferred_order if f in all_fields]

        out_csv_path = Path(out_csv_path)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)

        with out_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


# ============================================================
# 9) DEBUG EINZELPROFIL
# ============================================================

def debug_single_perthometer_profile(path_txt: str, nis_um=2.5, nic_um=800.0):
    """
    Führt eine Schritt-für-Schritt-Diagnose für ein einzelnes Perthometer-Profil durch.

    Was kann man damit machen?
    - Prüfen, wie sich die Vorverarbeitung auf das Profil auswirkt
    - Kennwerte und Signalgrößen nach jedem Schritt kontrollieren
    - Probleme beim Detrending, Thresholding oder Filtern erkennen
    """
    x_um, z_um, dx_um = load_perthometer_prf_txt(path_txt)

    print("RAW")
    print("dx_um:", dx_um)
    print("Laenge [um]:", x_um[-1] - x_um[0])
    print("z min/max [um]:", np.min(z_um), np.max(z_um))
    print("z peak-to-peak [um]:", np.ptp(z_um))
    print("z std [um]:", np.std(z_um))

    p_raw = SimpleProfile(z_um=z_um, x_um=x_um, title="raw")

    p_level = level_profile(p_raw)
    print("\nLEVEL")
    print("z min/max [um]:", np.min(p_level.z), np.max(p_level.z))
    print("z peak-to-peak [um]:", np.ptp(p_level.z))
    print("z std [um]:", np.std(p_level.z))

    p_det = detrend_profile_polynomial(p_level, degree=2)
    print("\nDETREND")
    print("z min/max [um]:", np.min(p_det.z), np.max(p_det.z))
    print("z peak-to-peak [um]:", np.ptp(p_det.z))
    print("z std [um]:", np.std(p_det.z))

    p_thr = threshold_percentile(p_det, upper=0.25, lower=0.25)
    z_thr = p_thr.z[np.isfinite(p_thr.z)]
    print("\nTHRESHOLD")
    print("z min/max [um]:", np.min(z_thr), np.max(z_thr))
    print("z peak-to-peak [um]:", np.ptp(z_thr))
    print("z std [um]:", np.std(z_thr))

    p_fill = fill_nonmeasured_linear(p_thr)
    print("\nFILL")
    print("z min/max [um]:", np.min(p_fill.z), np.max(p_fill.z))
    print("z peak-to-peak [um]:", np.ptp(p_fill.z))
    print("z std [um]:", np.std(p_fill.z))

    p_s, p_r, p_w = split_roughness_waviness(p_fill, nis_um=nis_um, nic_um=nic_um)

    print("\nROUGHNESS")
    print("z min/max [um]:", np.min(p_r.z), np.max(p_r.z))
    print("z peak-to-peak [um]:", np.ptp(p_r.z))
    print("z std [um]:", np.std(p_r.z))
    print("Ra:", p_r.Ra())
    print("Rq:", p_r.Rq())
    print("Rz:", p_r.Rz())

    print("\nWAVINESS")
    print("z min/max [um]:", np.min(p_w.z), np.max(p_w.z))
    print("z peak-to-peak [um]:", np.ptp(p_w.z))
    print("z std [um]:", np.std(p_w.z))
    print("Wa:", p_w.Wa())
    print("Wq:", p_w.Wq())
    print("Wz:", p_w.Wz())


# ============================================================
# 10) MAIN
# ============================================================

if __name__ == "__main__":
    perth_folder = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Mahr XCR20"
    nanofocus_folder = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Nanofocus/WSP00/ErodierteProben"
    out_csv_path = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Mahr XCR20/Ergebnisse/perthometer_nanofocus_R_W.csv"

    # Kombinierte Datensätze aufbauen
    combined = build_combined_perthometer_nanofocus(
        perth_folder=perth_folder,
        nanofocus_folder=nanofocus_folder,
        y_mode="center",
        nis_um=2.5,
        nic_um=800.0
    )

    print("Gefundene kombinierte Gruppen:")
    for key in sorted(combined.keys()):
        print(" ", key)

    # Beispielplot für eine Gruppe
    plot_perthometer_vs_nanofocus_rw(combined, "WSP00_L1_S2")

    # CSV mit Kennwerten speichern
    save_combined_metrics_csv(combined, out_csv_path, nis_um=2.5, nic_um=800.0)

    # Einzelprofil-Debug
    debug_single_perthometer_profile(
        "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Data/Mahr XCR20/WSP00_L1_S1_1.txt",
        nis_um=2.5,
        nic_um=800.0
    )