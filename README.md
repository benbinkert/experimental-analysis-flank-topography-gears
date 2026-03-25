# Bachelorarbeit – Experimentelle Analyse von Messmethoden und Messgrößen zur Bewertung der Flankentopographie von wälzgeschälten Zahnrädern

**Autor:** Ben Robin Binkert  
**Institut:** wbk Institut für Produktionstechnik, Karlsruher Institut für Technologie (KIT)  
**Abgabedatum:** 31.03.2026

---

## Thema

Diese Bachelorarbeit untersucht experimentell verschiedene Messmethoden und Messgrößen zur Charakterisierung der Flankentopographie von wälzgeschälten Zahnrädern (Power Skiving). Im Mittelpunkt steht der Vergleich zweier optischer Messsysteme – **NanoFocus** und **Keyence** – sowie deren Gegenüberstellung mit einem taktilen Perthometer (Mahr XCR 20) und einer Koordinatenmessmaschine (Zeiss KMG). Außerdem werden die Messdaten mit Simulationsdaten verglichen.

---

## Repository-Struktur

```
Bachelorarbeit/
├── Code/                          # Python-Auswerteskripte
│   ├── Main.py                    # Hauptskript / Einstiegspunkt
│   ├── kenngroesenBerechnung.py   # Batch-Berechnung von ISO-Flächenkennwerten
│   ├── messystemvergleich.py      # Vergleich NanoFocus vs. Keyence
│   ├── PerthometerNanofocus.py    # Vergleich Perthometer vs. NanoFocus
│   ├── statistikPerthoKmg.py      # Statistik Perthometer & KMG
│   ├── zeissKMG.py                # Auswertung Zeiss-KMG-Daten
│   ├── zeissKmg-perthometer-kenngroessen.py
│   ├── korrelationsAnalyse.py     # Spearman-Korrelation + FDR + Heatmaps
│   ├── Statistik.py               # Deskriptive Statistik der Kennwerte
│   ├── Plots.py                   # Plot-Funktionen
│   ├── Filter.py                  # 1D-Profilfilter (Gauss, Butterworth)
│   ├── Unterprogramme.py          # Hilfsfunktionen
│   ├── KeyenceImportZON.py        # Einlesen von Keyence-.zon-Dateien
│   ├── keyenceImportASCI.py       # Einlesen von Keyence-ASCII-Dateien
│   ├── keyenceFFT.py              # FFT-Analyse Keyence-Daten
│   ├── mat_in_surfalize.py        # MATLAB-.mat-Dateien in surfalize laden
│   ├── templateMatching.py        # Template-Matching zur Flächenausrichtung
│   └── GeometrischHilfen/         # Geometrische Hilfsberechnungen
│       ├── Hilfslinien.py
│       ├── OrthogonaleLinie.py
│       └── tool.py
├── Data/                          # Messdaten (nicht versioniert / groß)
│   ├── Keyence/
│   │   ├── WSP00/
│   │   └── WSP03/
│   ├── Nanofocus/
│   │   ├── WSP00/
│   │   └── WSP03/
│   ├── Simulation/
│   │   └── WSP00/
│   └── Zeiss KMG/
├── Ergebnisse/                    # Berechnete Kennwerte (CSV / Excel)
│   ├── Gesamt/
│   ├── KMGvsPertho/
│   ├── Messystemvergleich/
│   ├── NanoPertho/
│   └── Prozesfrei/
├── Bilder/                        # Abbildungen und Plots
│   ├── Plots/
│   └── Ueberprueft/
├── Notizen/                       # Arbeitsnotizen
├── Vorlage-wbk.tex                # LaTeX-Quellcode der Arbeit
├── Quellen.bib                    # Bibliografie (BibTeX)
├── wbk.cls / wbk.bst              # LaTeX-Formatvorlage (wbk-Template)
└── out/                           # Kompiliertes PDF der Arbeit
```

---

## Messsysteme und Datensätze

| Messsystem | Dateityp | Beschreibung |
|---|---|---|
| **NanoFocus µsurf** | `.nms` | Konfokales Messsystem (höhere laterale Auflösung) |
| **Keyence VR-6000** | `.zon`, `.sdf` | Optisches Profilometer, Triangulationsverfahren |
| **Mahr Perthometer XCR 20** | `.txt` (ASCII) | Taktiles Profilometer |
| **Zeiss KMG** | CSV / Tabellen | Koordinatenmessmaschine |
| **Simulation** | `.sdf`, `.mat` | Simulierte Flankentopographie |

**Proben:**
- `WSP00` – Wendeschneidplatte 0 (Referenzzustand)
- `WSP03` – Wendeschneidplatte 03 (verschlissener Zustand)

---

## Vorverarbeitungspipeline

Alle Oberflächen werden standardisiert mit folgenden Schritten aufbereitet:

1. **Leveln** – Neigung der Oberfläche entfernen
2. **Polynomiales Detrending** (Grad 2) – Formabweichung kompensieren
3. **Thresholding** – Ausreißer entfernen
4. **Auffüllen nicht gemessener Werte** – zeilenweise lineare Interpolation
5. **Filterung** – je nach Auswertefall:
   - *Prozessfrei:* Bandpass S=1,6 µm / L=100 µm (NanoFocus) bzw. S=8 µm / L=100 µm (Keyence)
   - *Vergleich:* Bandpass S=8 µm / L=250 µm (einheitlich für beide Systeme)
   - *Gesamt:* Lowpass mit S-Filter

---

## Berechnete ISO-Kennwerte

Die folgenden ISO 25178 Flächenkennwerte werden automatisiert für jede Messung berechnet:

| Kategorie | Kennwerte |
|---|---|
| Höhenparameter | Sa, Sq, Sp, Sv, Sz, Ssk, Sku |
| Hybridparameter | Sdr, Sdq |
| Funktionale Parameter | Sk, Spk, Svk, Smr1, Smr2, Sxp |
| Funktionale Volumen | Vmp, Vmc, Vvv, Vvc |
| Isotropie / Textur | Sal, Str |

---

## Auswertungen

| Skript | Inhalt |
|---|---|
| `kenngroesenBerechnung.py` | Batch-Berechnung der ISO-Kennwerte für alle Dateien |
| `messystemvergleich.py` | Direkter Vergleich NanoFocus ↔ Keyence (gleiche Auswertefläche) |
| `PerthometerNanofocus.py` | Vergleich taktiles Profil ↔ konfokale Fläche (R- und W-Parameter) |
| `zeissKMG.py` | Auswertung der KMG-Welligkeit (Flankenlinien) |
| `korrelationsAnalyse.py` | Spearman-Korrelation, Permutationstest, FDR-Korrektur, Heatmaps |
| `Statistik.py` | Mittelwert, Standardabweichung, Systemvergleich aus CSV |
| `templateMatching.py` | Template-Matching zur geometrischen Ausrichtung von Flächen |
| `keyenceFFT.py` | FFT-Analyse der Welligkeit aus Keyence-Daten |

---

## LaTeX-Kompilierung

Die Bachelorarbeit wird mit `latexmk` und der wbk-Formatvorlage kompiliert:

```bash
# Alles löschen und neu bauen
/Library/TeX/texbin/latexmk -C -outdir=out Vorlage-wbk.tex

# PDF erzeugen
/Library/TeX/texbin/latexmk -pdf -interaction=nonstopmode -synctex=1 -outdir=out Vorlage-wbk.tex
```

Das kompilierte PDF liegt unter `out/Vorlage-wbk.pdf`.

---

## Abhängigkeiten (Python)

Das Projekt verwendet folgende Python-Bibliotheken:

- [`surfalize`](https://github.com/fredericjs/surfalize) – Laden und Auswerten von Oberflächenmessdaten (`.nms`, `.sdf`, `.zon`)
- `numpy`, `scipy` – Numerische Berechnungen und Filterung
- `pandas` – Datenverarbeitung und CSV-Export
- `matplotlib` – Visualisierung und Plots
- `pathlib`, `re` – Datei- und Pfadverwaltung

---

## Lizenz

Dieses Repository enthält den Quellcode einer akademischen Abschlussarbeit. Alle Rechte vorbehalten.
