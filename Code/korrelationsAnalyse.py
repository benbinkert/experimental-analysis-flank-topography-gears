# korrelationsAnalyse.py
# ------------------------------------------------------------
# Führt eine Korrelationsanalyse für ISO-Kennwerte durch.
#
# Was kann man damit machen?
# - Spearman-Korrelationen zwischen allen Kennwerten berechnen
# - Permutationstests für die Signifikanz durchführen
# - p-Werte mit Benjamini-Hochberg per FDR korrigieren
# - nur starke und signifikante Zusammenhänge clustern
# - Heatmaps mit ausgegrauten nicht-signifikanten Zellen erzeugen
# - Cluster zusätzlich als rote Boxen in der Heatmap markieren
#
# Enthalten:
# - Spearman-ρ
# - Permutationstest
# - FDR-Korrektur
# - hierarchisches Clustering mit d = 1 - |ρ|
# - Export von ρ-, p-, q- und n-Matrizen
# ------------------------------------------------------------

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# ============================================================
# KONFIGURATION
# Hier werden Einlesepfad, Schwellwerte und Ausgabeverzeichnis
# für die gesamte Korrelationsanalyse festgelegt
# ============================================================

# Eingabe-CSV mit allen Kennwerten
csv_path = "/Users/benbinkert/PycharmProjects/Bachelorarbeit/Ergebnisse/Gesamt/Gesamt-EineCSV.csv"
sep = ";"

# Optionaler Filterfall
# None bedeutet: alle Filterfälle gemeinsam auswerten
FILTERCASE = "gesamt"

# Mindestanzahl gemeinsamer gültiger Werte pro Parameterpaar
min_n = 6

# Signifikanzniveau nach FDR-Korrektur
alpha_fdr = 0.10

# Mindeststärke einer Korrelation, damit sie für Clusterkanten zählt
rho_thr_cluster = 0.70

# Bis zu dieser Stichprobengröße wird exakt über alle Permutationen getestet
exact_perm_max_n = 8

# Anzahl Monte-Carlo-Permutationen für größere Stichproben
mc_B = 20000

# Ausgabeordner für Tabellen, JSON und Heatmaps
out_dir = Path("out_corr")
out_dir.mkdir(parents=True, exist_ok=True)

# Fachliche Gruppierung der ISO-Kennwerte
GROUPS = [
    ("Höhen", ["Sa", "Sq", "Sp", "Sv", "Sz", "Ssk", "Sku"]),
    ("Hybrid", ["Sdr", "Sdq"]),
    ("Lateral", ["Sal", "Str"]),
    ("Funktional (plateauartig)", ["Sk", "Spk", "Svk", "Smrk1", "Smrk2", "Sxp"]),
    ("Funktional (Volumen)", ["Vmp", "Vmc", "Vvv", "Vvc"]),
]

# Schöne LaTeX-artige Ticklabels für die Heatmap
TICK_LABELS = {
    "Sa": r"$S_a$", "Sq": r"$S_q$", "Sp": r"$S_p$", "Sv": r"$S_v$", "Sz": r"$S_z$",
    "Ssk": r"$S_{sk}$", "Sku": r"$S_{ku}$",
    "Sdr": r"$S_{dr}$", "Sdq": r"$S_{dq}$",
    "Sal": r"$S_{al}$", "Str": r"$S_{tr}$",
    "Sk": r"$S_k$", "Spk": r"$S_{pk}$", "Svk": r"$S_{vk}$",
    "Smrk1": r"$S_{mrk1}$", "Smrk2": r"$S_{mrk2}$", "Sxp": r"$S_{xp}$",
    "Vmp": r"$V_{mp}$", "Vmc": r"$V_{mc}$", "Vvv": r"$V_{vv}$", "Vvc": r"$V_{vc}$",
}


# ============================================================
# FDR-KORREKTUR
# Benjamini-Hochberg-Korrektur für mehrere Tests
# ============================================================

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Führt eine Benjamini-Hochberg-FDR-Korrektur auf einer Menge von p-Werten durch.

    Was kann man damit machen?
    - Viele Paarvergleiche gleichzeitig absichern
    - Die Rate falsch-positiver Befunde kontrollieren
    - Aus ungefilterten p-Werten korrigierte q-Werte berechnen

    Parameter:
    - pvals:
      Array mit p-Werten

    Rückgabe:
    - Array mit q-Werten in derselben Reihenfolge
    """
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size

    # p-Werte aufsteigend sortieren
    order = np.argsort(pvals)
    ranked = pvals[order]

    # BH-Formel anwenden
    q = ranked * n / (np.arange(1, n + 1))

    # Monotone Korrektur von hinten nach vorne
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    # Wieder in Originalreihenfolge zurückschreiben
    out = np.empty_like(q)
    out[order] = q
    return out


# ============================================================
# SCHNELLE SPEARMAN-BERECHNUNG
# ============================================================

def zscore_1d(x: np.ndarray) -> np.ndarray:
    """
    Standardisiert einen 1D-Vektor auf Mittelwert 0 und Standardabweichung 1.

    Was kann man damit machen?
    - Rangdaten für Korrelationsberechnungen normieren
    - Innere Produkte direkt für Spearman ρ nutzen
    - Numerisch stabile Korrelationen berechnen

    Rückgabe:
    - Z-standardisierter Vektor
    """
    x = x.astype(float)
    mu = x.mean()
    sd = x.std(ddof=1)

    if sd <= 0:
        return x * 0.0

    return (x - mu) / sd


def spearman_rho_fast(x: np.ndarray, y: np.ndarray) -> float:
    """
    Berechnet Spearman-ρ über Rangbildung und Standardisierung.

    Was kann man damit machen?
    - Robuste monotone Zusammenhänge zwischen zwei Kennwerten messen
    - Schneller als über wiederholte High-Level-Aufrufe rechnen
    - Grundlage für Permutationstests schaffen

    Parameter:
    - x, y:
      Zwei gleich lange Datenvektoren

    Rückgabe:
    - Spearman-Korrelationskoeffizient ρ
    """
    # Ränge bilden
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()

    # Ränge standardisieren
    zx = zscore_1d(rx)
    zy = zscore_1d(ry)

    # Korrelation als normiertes Skalarprodukt
    return float((zx @ zy) / (len(zx) - 1))


# ============================================================
# PERMUTATIONSTEST
# Exakter Test für kleine n, Monte Carlo für größere n
# ============================================================

_perm_idx_cache = {}


def get_perm_indices(n: int):
    """
    Liefert alle Permutationen der Indizes 0..n-1 und cached das Ergebnis.

    Was kann man damit machen?
    - Exakte Permutationstests bei kleinen Stichproben effizient ausführen
    - Wiederholte Berechnungen beschleunigen

    Rückgabe:
    - Array aller Permutationen
    """
    if n not in _perm_idx_cache:
        perms = np.array(list(itertools.permutations(range(n))), dtype=np.int16)
        _perm_idx_cache[n] = perms
    return _perm_idx_cache[n]


def spearman_perm_pvalue_fast(x: np.ndarray, y: np.ndarray, rng=None) -> float:
    """
    Berechnet einen Permutations-p-Wert für Spearman-ρ.

    Was kann man damit machen?
    - Die Signifikanz einer beobachteten Rangkorrelation testen
    - Exakte Tests bei kleinen n und Monte-Carlo-Tests bei größeren n kombinieren
    - Robuste p-Werte ohne Normalverteilungsannahme erhalten

    Rückgabe:
    - Zweiseitiger p-Wert bezogen auf |ρ|
    """
    n = len(x)

    # Beobachtete absolute Korrelation
    rho_obs = abs(spearman_rho_fast(x, y))

    # Ränge einmalig vorbereiten
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    zx = zscore_1d(rx)
    zy = zscore_1d(ry)

    # Exakter Test für kleine n
    if n <= exact_perm_max_n:
        perm_idx = get_perm_indices(n)
        zy_perm = zy[perm_idx]
        rho_perm = np.abs((zy_perm @ zx) / (n - 1))
        return float(np.mean(rho_perm >= rho_obs - 1e-12))

    # Monte-Carlo-Permutation für größere n
    if rng is None:
        rng = np.random.default_rng(0)

    cnt = 0
    for _ in range(mc_B):
        perm = rng.permutation(n)
        rho = abs(float((zy[perm] @ zx) / (n - 1)))
        if rho >= rho_obs - 1e-12:
            cnt += 1

    return float((cnt + 1) / (mc_B + 1))


# ============================================================
# PARAMETER-REIHENFOLGE NACH ISO-GRUPPEN
# ============================================================

def iso_param_list(existing_cols) -> list[str]:
    """
    Erzeugt eine geordnete Liste vorhandener ISO-Parameter entsprechend GROUPS.

    Was kann man damit machen?
    - Nur fachlich relevante Spalten auswählen
    - Eine konsistente Reihenfolge für Tabellen und Heatmaps erzeugen
    - Vorhandene Parameter sauber in Gruppenstruktur bringen

    Rückgabe:
    - Geordnete Liste der in den Daten vorhandenen Parameter
    """
    out = []
    for _, params in GROUPS:
        for p in params:
            if p in existing_cols:
                out.append(p)
    return out


# ============================================================
# HIERARCHISCHES CLUSTERING AUF |ρ|
# ============================================================

def hclust_from_corr(C: pd.DataFrame, threshold: float, method="average"):
    """
    Führt hierarchisches Clustering auf Basis von d = 1 - |ρ| durch.

    Was kann man damit machen?
    - Stark ähnliche Kennwerte zu Clustern zusammenfassen
    - Positive und negative starke Zusammenhänge gleich behandeln
    - Signifikante Korrelationsblöcke in der Heatmap strukturieren

    Parameter:
    - C:
      Korrelationsmatrix
    - threshold:
      Mindestwert für |ρ|, ab dem Punkte im selben Cluster landen können
    - method:
      Linkage-Methode, Standard: average

    Rückgabe:
    - Dictionary {Parametername: Cluster-ID}
    """
    cols = list(C.columns)
    n = len(cols)

    if n <= 1:
        return {cols[0]: 1} if n == 1 else {}

    # Absolute Korrelationen als Ähnlichkeit
    A = np.abs(C.to_numpy(dtype=float))
    A = np.nan_to_num(A, nan=0.0)

    # Distanzmatrix
    D = 1.0 - A
    np.fill_diagonal(D, 0.0)

    # Kompakte Distanzdarstellung für linkage
    dvec = squareform(D, checks=False)
    Z = linkage(dvec, method=method)

    # Clusterschnitt auf Distanzniveau 1-threshold
    t = 1.0 - float(threshold)
    labels = fcluster(Z, t=t, criterion="distance")

    return {cols[i]: int(labels[i]) for i in range(n)}


def clusters_to_list(cluster_map: dict):
    """
    Wandelt ein Parameter->Cluster-Mapping in eine sortierte Clusterliste um.

    Was kann man damit machen?
    - Cluster besser lesbar ausgeben
    - Clustergrößen vergleichen
    - Cluster später als JSON speichern

    Rückgabe:
    - Liste von Tupeln (Cluster-ID, sortierte Parameterliste)
    """
    inv = {}
    for p, cid in cluster_map.items():
        inv.setdefault(cid, []).append(p)

    clusters = sorted(inv.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return [(cid, sorted(params)) for cid, params in clusters]


def reorder_by_cluster_then_iso(params: list[str], cluster_map: dict):
    """
    Sortiert Parameter zuerst nach Cluster, dann nach fachlicher ISO-Reihenfolge.

    Was kann man damit machen?
    - Heatmaps mit zusammenhängenden Clusterblöcken aufbauen
    - Innerhalb eines Clusters trotzdem eine nachvollziehbare Reihenfolge behalten

    Rückgabe:
    - Sortierte Parameterliste
    """
    iso_idx = {}
    k = 0
    for _, plist in GROUPS:
        for p in plist:
            iso_idx[p] = k
            k += 1

    def key(p):
        return (cluster_map.get(p, 999999), iso_idx.get(p, 999999), p)

    return sorted(params, key=key)


def cluster_boxes_from_order(order_cols: list[str], cluster_map: dict):
    """
    Bestimmt Boxen für zusammenhängende Clusterblöcke in der sortierten Reihenfolge.

    Was kann man damit machen?
    - In der Heatmap rote Rahmen um Clusterblöcke zeichnen
    - Visualisierung mit Start- und Endindizes vorbereiten

    Rückgabe:
    - Liste von Tupeln (Cluster-ID, Startindex, Endindex)
    """
    boxes = []
    if not order_cols:
        return boxes

    current = cluster_map.get(order_cols[0], None)
    start = 0

    for i, p in enumerate(order_cols[1:], start=1):
        cid = cluster_map.get(p, None)
        if cid != current:
            boxes.append((current, start, i - 1))
            current = cid
            start = i

    boxes.append((current, start, len(order_cols) - 1))
    return boxes


# ============================================================
# PLOTTING
# Heatmap mit Ausgrauen nicht-signifikanter Zellen und
# roten Cluster-Boxen
# ============================================================

def cluster_color(_cid: int):
    """
    Liefert die Farbe für Clusterrahmen.

    Was kann man damit machen?
    - Clusterblöcke einheitlich einfärben
    - Darstellung leicht zentral anpassen
    """
    return (1.0, 0.0, 0.0, 1.0)  # rot


def plot_heatmap_with_overlay_and_boxes(
    C: pd.DataFrame,
    mask_not_sig: np.ndarray,
    title: str,
    save_path: Path,
    boxes=None,
    note=None,
    alpha_fdr_for_legend: float = 0.10,
    rho_thr_for_legend: float = 0.70,
):
    """
    Zeichnet eine Korrelations-Heatmap mit Signifikanz-Overlay und Cluster-Boxen.

    Was kann man damit machen?
    - Eine vollständige Korrelationsmatrix visualisieren
    - Nicht-signifikante Zellen ausgrauen
    - Clusterblöcke visuell hervorheben
    - Eine präsentationsfähige Abbildung abspeichern

    Parameter:
    - C:
      Korrelationsmatrix
    - mask_not_sig:
      Bool-Maske für nicht-signifikante Zellen
    - title:
      Titel der Grafik
    - save_path:
      Dateipfad zum Speichern
    - boxes:
      Optionale Clusterrahmen
    - note:
      Zusatztext unter der Heatmap
    """
    arr = C.to_numpy(dtype=float)
    n = arr.shape[0]

    # Dynamische Figurgröße abhängig von der Parameterzahl
    fig_w = max(9, 0.45 * n + 4)
    fig_h = max(8, 0.45 * n + 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)

    # Korrelationsmatrix darstellen
    im = ax.imshow(arr, vmin=-1, vmax=1)

    # Nicht-signifikante Zellen halbtransparent weiß überlagern
    overlay = np.zeros((n, n, 4), dtype=float)
    overlay[..., :3] = 1.0
    overlay[..., 3] = mask_not_sig.astype(float) * 0.65
    ax.imshow(overlay)

    # Achsenbeschriftung
    labels = [TICK_LABELS.get(c, c) for c in C.columns]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=14)
    ax.set_yticklabels(labels, fontsize=14)

    ax.set_title(title, fontsize=14)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ")

    # Zellengitter
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Clusterrahmen und Clusterlabels
    if boxes is not None:
        for cid, i0, i1 in boxes:
            col = cluster_color(int(cid) if cid is not None else 0)

            ax.add_patch(
                plt.Rectangle(
                    (i0 - 0.5, i0 - 0.5),
                    (i1 - i0 + 1), (i1 - i0 + 1),
                    fill=False,
                    linewidth=3.0,
                    edgecolor=col
                )
            )

            ax.text(
                i0 - 0.45, i0 - 0.45, f"C{cid}",
                fontsize=10, fontweight="bold",
                ha="left", va="bottom",
                color=col,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor=col,
                    linewidth=1.5,
                    alpha=0.95
                )
            )

    # Legende bewusst unten platzieren
    from matplotlib.patches import Patch
    legend_items = [
        Patch(
            facecolor=(1, 1, 1, 0.65),
            edgecolor="none",
            label=f"Ausgegraut: nicht signifikant (q > {alpha_fdr_for_legend})"
        ),
        Patch(
            facecolor="none",
            edgecolor=cluster_color(1),
            linewidth=3,
            label=f"Cluster-Block: q ≤ {alpha_fdr_for_legend} & |ρ| ≥ {rho_thr_for_legend}"
        ),
    ]

    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=1,
        frameon=True
    )

    # Optionaler Hinweistext
    if note:
        ax.text(0.0, -3.0, note, fontsize=10, transform=ax.transData)

    # Platz für Legende unten freihalten
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(save_path)
    plt.close(fig)

    print(f"[OK] wrote {save_path}")


# ============================================================
# KERNANALYSE FÜR EIN MESSSYSTEM
# ============================================================

def analyze_system(df: pd.DataFrame, system_name: str):
    """
    Führt die komplette Korrelationsanalyse für ein Messsystem durch.

    Was kann man damit machen?
    - Daten eines Systems filtern
    - Korrelations-, p-, q- und n-Matrizen erzeugen
    - Cluster aus signifikanten und starken Korrelationen bestimmen
    - Heatmaps und Exportdateien automatisch erstellen

    Ablauf:
    1. Daten auf Messsystem und optional Filtercase filtern
    2. ISO-Parameter auswählen
    3. Paarweise Spearman-ρ, Permutations-p und Paaranzahl berechnen
    4. FDR-Korrektur anwenden
    5. Nur signifikante und starke Kanten für Cluster verwenden
    6. Tabellen, JSON und Heatmap exportieren
    """
    # Nur gewünschtes Messsystem auswählen
    sub = df[df["messsystem"].astype(str).str.lower() == system_name.lower()].copy()
    if sub.empty:
        print(f"[SKIP] no rows for {system_name}")
        return

    # Optional zusätzlich auf Filtercase einschränken
    if FILTERCASE is not None and "filtercase" in sub.columns:
        sub = sub[sub["filtercase"].astype(str).str.lower() == str(FILTERCASE).lower()].copy()
        if sub.empty:
            print(f"[SKIP] {system_name}: no rows after filtercase='{FILTERCASE}'")
            return

    # Identifikationsspalten von echten Kennwerten trennen
    id_cols = {"messsystem", "filtercase", "datei"}
    metric_cols = [c for c in sub.columns if c not in id_cols]

    # Alle Kennwertspalten numerisch erzwingen
    for c in metric_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    # Nur geordnete ISO-Parameter auswählen
    params = [p for p in iso_param_list(sub.columns) if p in metric_cols]
    if not params:
        print(f"[SKIP] {system_name}: no ISO params found")
        return

    X = sub[params]
    cols = list(X.columns)
    ncol = len(cols)

    # Ergebnis-Matrizen anlegen
    C = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)  # rho
    P = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)  # p
    N = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)  # n

    rng = np.random.default_rng(0)

    # Paarweise Korrelationen berechnen
    for i in range(ncol):
        for j in range(ncol):
            a = X.iloc[:, i]
            b = X.iloc[:, j]

            # Nur gemeinsame gültige Werte verwenden
            m = a.notna() & b.notna()
            nij = int(m.sum())
            N.iat[i, j] = nij

            if nij < min_n:
                continue

            xa = a[m].to_numpy(dtype=float)
            xb = b[m].to_numpy(dtype=float)

            # Spearman ρ
            C.iat[i, j] = spearman_rho_fast(xa, xb)

            # Permutations-p-Wert
            P.iat[i, j] = 0.0 if i == j else spearman_perm_pvalue_fast(xa, xb, rng=rng)

    # --------------------------------------------------------
    # FDR nur auf oberem Dreieck anwenden
    # --------------------------------------------------------
    p_list, ij_list = [], []
    for i in range(ncol):
        for j in range(i + 1, ncol):
            p = P.iat[i, j]
            if np.isfinite(p):
                p_list.append(p)
                ij_list.append((i, j))

    p_arr = np.array(p_list, dtype=float)
    q_arr = bh_fdr(p_arr) if p_arr.size else np.array([])

    # q-Matrix aufbauen
    Q = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    for (i, j), q in zip(ij_list, q_arr):
        Q.iat[i, j] = q
        Q.iat[j, i] = q
    np.fill_diagonal(Q.values, 0.0)

    # Signifikanzmaske
    sig = (Q.to_numpy(dtype=float) <= alpha_fdr) & np.isfinite(C.to_numpy(dtype=float))
    sig |= np.eye(ncol, dtype=bool)
    mask_not_sig = ~sig

    # --------------------------------------------------------
    # Cluster-Matrix aufbauen:
    # Nur signifikante und starke Korrelationen bleiben erhalten,
    # alles andere wird auf 0 gesetzt
    # --------------------------------------------------------
    C_sig = C.copy()
    for i in range(ncol):
        for j in range(ncol):
            if i == j:
                continue

            ok = (
                np.isfinite(C_sig.iat[i, j])
                and np.isfinite(Q.iat[i, j])
                and np.isfinite(N.iat[i, j])
                and (N.iat[i, j] >= min_n)
                and (Q.iat[i, j] <= alpha_fdr)
                and (abs(C_sig.iat[i, j]) >= rho_thr_cluster)
            )

            if not ok:
                C_sig.iat[i, j] = 0.0

    # Cluster bestimmen
    cluster_map = hclust_from_corr(C_sig, threshold=rho_thr_cluster, method="average")
    clusters = clusters_to_list(cluster_map)

    # Konsolenausgabe der Cluster
    print("\n" + "=" * 70)
    print(
        f"{system_name.upper()} – Cluster auf signifikanten Kanten: "
        f"q≤{alpha_fdr}, |ρ|≥{rho_thr_cluster:.2f}, min_n={min_n}"
        + (f", filtercase={FILTERCASE}" if FILTERCASE is not None else "")
    )
    print("=" * 70)
    for cid, cl in clusters:
        print(f"Cluster C{cid} (n={len(cl)}): " + ", ".join(cl))
    print("=" * 70)

    # Parameter für Heatmap nach Cluster und ISO-Reihenfolge sortieren
    params_sorted = reorder_by_cluster_then_iso(cols, cluster_map)

    C_sorted = C.loc[params_sorted, params_sorted]
    Q_sorted = Q.loc[params_sorted, params_sorted]
    N_sorted = N.loc[params_sorted, params_sorted]

    sig_sorted = (Q_sorted.to_numpy(dtype=float) <= alpha_fdr) & np.isfinite(C_sorted.to_numpy(dtype=float))
    sig_sorted |= np.eye(ncol, dtype=bool)
    mask_not_sig_sorted = ~sig_sorted

    boxes = cluster_boxes_from_order(params_sorted, cluster_map)

    # --------------------------------------------------------
    # Ergebnisdateien speichern
    # --------------------------------------------------------
    C_sorted.to_csv(out_dir / f"rho_{system_name.lower()}.csv", sep=";")
    P.to_csv(out_dir / f"p_perm_{system_name.lower()}.csv", sep=";")
    Q_sorted.to_csv(out_dir / f"q_fdr_{system_name.lower()}.csv", sep=";")
    N_sorted.to_csv(out_dir / f"n_pairs_{system_name.lower()}.csv", sep=";")

    cluster_json = out_dir / f"clusters_{system_name.lower()}_permfdr_absrho{int(rho_thr_cluster*100)}.json"
    cluster_json.write_text(json.dumps({
        "filtercase": FILTERCASE,
        "alpha_fdr": alpha_fdr,
        "rho_threshold": rho_thr_cluster,
        "min_n": min_n,
        "clusters": [{"id": int(cid), "params": cl} for cid, cl in clusters]
    }, indent=2), encoding="utf-8")

    print(f"[OK] wrote rho/p/q/n + {cluster_json.name}")

    # Hinweistext unter der Heatmap
    note = (
        f"Permutationstest (Spearman), FDR: q≤{alpha_fdr}; "
        f"Cluster-Kanten: |ρ|≥{rho_thr_cluster:.2f}; min_n={min_n}"
        + (f"; filtercase={FILTERCASE}" if FILTERCASE is not None else "")
    )

    # Heatmap speichern
    plot_heatmap_with_overlay_and_boxes(
        C=C_sorted,
        mask_not_sig=mask_not_sig_sorted,
        title=f"{system_name}: Spearman-ρ (nicht signifikant ausgegraut) + Cluster-Blöcke",
        save_path=out_dir / f"heatmap_{system_name.lower()}_permfdr_clusters.png",
        boxes=boxes,
        note=note,
        alpha_fdr_for_legend=alpha_fdr,
        rho_thr_for_legend=rho_thr_cluster,
    )


# ============================================================
# HAUPTFUNKTION
# CSV laden, Spalten harmonisieren und Analyse für beide
# Messsysteme starten
# ============================================================

def main():
    """
    Startet die gesamte Korrelationsanalyse.

    Was kann man damit machen?
    - Gesamtdaten aus der CSV laden
    - Spaltennamen angleichen
    - Nanofocus und Keyence separat auswerten
    """
    df = pd.read_csv(csv_path, sep=sep)

    # Smr-Spalten auf Smrk harmonisieren
    rename_map = {}
    if "Smr1" in df.columns and "Smrk1" not in df.columns:
        rename_map["Smr1"] = "Smrk1"
    if "Smr2" in df.columns and "Smrk2" not in df.columns:
        rename_map["Smr2"] = "Smrk2"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Analyse getrennt nach Messsystem durchführen
    analyze_system(df, "Nanofocus")
    analyze_system(df, "Keyence")


if __name__ == "__main__":
    main()