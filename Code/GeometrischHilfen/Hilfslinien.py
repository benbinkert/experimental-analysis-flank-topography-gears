import numpy as np
import matplotlib.pyplot as plt


def line_distance_um(line1, line2, step_x, step_y):
    """
    Berechnet den senkrechten Abstand zwischen zwei Linien in µm.

    Was kann man damit machen?
    - Den orthogonalen Abstand zwischen zwei eingezeichneten Linien bestimmen
    - Abstände direkt in physikalischen Einheiten ausgeben
    - Grundlage für die spätere Mittelwertbildung mehrerer Linienabstände

    Parameter:
    - line1, line2:
      Jeweils zwei Punkte in Pixelkoordinaten:
      ((x1, y1), (x2, y2))
    - step_x, step_y:
      Umrechnung von Pixel in µm pro Pixel in x- bzw. y-Richtung

    Rückgabe:
    - Senkrechter Abstand zwischen den Linien in µm
    """
    # Zwei Punkte der ersten Linie auslesen
    (x1, y1), (x2, y2) = line1

    # Von der zweiten Linie reicht ein Punkt,
    # da der Abstand zur unendlichen Geraden berechnet wird
    (x3, y3), _ = line2

    # Pixelkoordinaten in physikalische Koordinaten [µm] umrechnen
    x1u, y1u = x1 * step_x, y1 * step_y
    x2u, y2u = x2 * step_x, y2 * step_y
    x3u, y3u = x3 * step_x, y3 * step_y

    # Richtungsvektor der ersten Linie
    dx = x2u - x1u
    dy = y2u - y1u

    # Länge des Richtungsvektors
    n = np.hypot(dx, dy)
    if n == 0:
        raise ValueError("Eine Linie ist degeneriert.")

    # Einheitsnormale auf Linie 1
    # Diese steht senkrecht auf dem Richtungsvektor
    nx = -dy / n
    ny = dx / n

    # Punkt-Linien-Abstand über Projektion auf die Normale
    d = abs((x3u - x1u) * nx + (y3u - y1u) * ny)
    return d


def _clip_infinite_line_to_axes(p1, p2, width, height):
    """
    Schneidet eine gedachte unendliche Gerade auf den Bildrand zurück.

    Was kann man damit machen?
    - Eine nur durch zwei Klickpunkte definierte Linie über das ganze Bild ziehen
    - Die Gerade sauber bis an die Bildgrenzen darstellen
    - Für eine bessere Visualisierung die volle Linie statt nur des Liniensegments zeigen

    Parameter:
    - p1, p2:
      Zwei Punkte der Linie in Pixelkoordinaten
    - width, height:
      Bildbreite und Bildhöhe in Pixel

    Rückgabe:
    - Zwei Punkte ((xa, ya), (xb, yb)), die auf dem Bildrand liegen
      und die sichtbare Schnittkante der unendlichen Linie bilden
    """
    x0, y0 = p1
    x1, y1 = p2
    dx = x1 - x0
    dy = y1 - y0

    # Liste möglicher Schnittpunkte mit dem Bildrand
    pts = []

    # Schnitt mit linker und rechter Bildkante
    if abs(dx) > 1e-12:
        t = (0 - x0) / dx
        y = y0 + t * dy
        if 0 <= y <= height - 1:
            pts.append((0, y))

        t = ((width - 1) - x0) / dx
        y = y0 + t * dy
        if 0 <= y <= height - 1:
            pts.append((width - 1, y))

    # Schnitt mit oberer und unterer Bildkante
    if abs(dy) > 1e-12:
        t = (0 - y0) / dy
        x = x0 + t * dx
        if 0 <= x <= width - 1:
            pts.append((x, 0))

        t = ((height - 1) - y0) / dy
        x = x0 + t * dx
        if 0 <= x <= width - 1:
            pts.append((x, height - 1))

    # Doppelte Schnittpunkte entfernen
    unique = []
    for p in pts:
        if not any(np.hypot(p[0] - q[0], p[1] - q[1]) < 1e-9 for q in unique):
            unique.append(p)

    # Falls weniger als zwei gültige Randpunkte gefunden wurden,
    # werden die Originalpunkte zurückgegeben
    if len(unique) < 2:
        return (p1, p2)

    # Von allen möglichen Randpunkten die zwei am weitesten
    # auseinanderliegenden Punkte wählen
    best = None
    best_d = -1
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            d = np.hypot(unique[j][0] - unique[i][0], unique[j][1] - unique[i][1])
            if d > best_d:
                best_d = d
                best = (unique[i], unique[j])

    return best


def draw_multiple_lines_and_measure(surface):
    """
    Zeigt eine Oberfläche an, lässt beliebig viele Linien anklicken
    und berechnet die senkrechten Abstände zwischen benachbarten Linien.

    Was kann man damit machen?
    - Mehrere Linien manuell auf einer Oberfläche einzeichnen
    - Die Linien im Bild visualisieren
    - Den senkrechten Abstand zwischen aufeinanderfolgenden Linien bestimmen
    - Einen Mittelwert der Abstände direkt im Plot anzeigen

    Ablauf:
    1. Oberfläche anzeigen
    2. Pro Linie zwei Punkte anklicken
    3. Mit Enter beenden
    4. Linien werden eingezeichnet und die Abstände berechnet

    Rückgabe:
    - Liste der berechneten Linienabstände in µm
    """
    z = surface.data
    h, w = z.shape

    # Erster Plot:
    # Benutzer klickt hier die Linienpunkte an
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        z,
        cmap="turbo",
        extent=(0, surface.width_um, 0, surface.height_um),
        origin="upper",   # gleiches Koordinatensystem wie surface.show()
        aspect="auto"
    )
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("z [µm]")

    ax.set_title("Je Linie 2 Punkte klicken, Enter zum Beenden")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")

    print("Je Linie 2 Punkte klicken. Enter beendet.")
    pts = plt.ginput(n=-1, timeout=0)
    plt.close(fig)

    # Mindestens zwei Linien = vier Punkte
    if len(pts) < 4:
        print("Mindestens 2 Linien (= 4 Klicks) nötig.")
        return []

    # Bei ungerader Punktzahl letzten Punkt ignorieren
    if len(pts) % 2 != 0:
        print("Ungerade Zahl an Klicks -> letzter Punkt wird ignoriert.")
        pts = pts[:-1]

    # Umrechnung von Plotkoordinaten [µm] zu Pixelkoordinaten
    # Wegen origin='upper' muss y gespiegelt werden
    pixel_points = []
    for xu, yu in pts:
        xp = xu / surface.step_x
        yp = (surface.height_um - yu) / surface.step_y
        pixel_points.append((xp, yp))

    # Aus je zwei Punkten eine Linie erzeugen
    lines = []
    for i in range(0, len(pixel_points), 2):
        lines.append((pixel_points[i], pixel_points[i + 1]))

    # Zweiter Plot:
    # Ergebnisdarstellung mit Linien und Mittelwert
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        z,
        cmap="turbo",
        extent=(0, surface.width_um, 0, surface.height_um),
        origin="upper",
        aspect="auto"
    )
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("z [µm]")

    ax.set_title("Eingezeichnete Linien + Mittelwert Abstand")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")

    # Jede Linie bis zum Bildrand verlängert einzeichnen
    for idx, (p1, p2) in enumerate(lines):
        a, b = _clip_infinite_line_to_axes(p1, p2, w, h)

        ax.plot(
            [a[0] * surface.step_x, b[0] * surface.step_x],
            [surface.height_um - a[1] * surface.step_y,
             surface.height_um - b[1] * surface.step_y],
            linewidth=2,
            label=f"Linie {idx + 1}"
        )

    # Senkrechte Abstände zwischen benachbarten Linien berechnen
    distances = []
    for i in range(len(lines) - 1):
        d = line_distance_um(lines[i], lines[i + 1], surface.step_x, surface.step_y)
        distances.append(d)
        print(f"Abstand Linie {i+1} -> Linie {i+2}: {d:.3f} µm")

    distances = np.asarray(distances, dtype=float)

    # Mittelwert der Abstände berechnen und im Plot anzeigen
    if distances.size:
        d_mean = float(np.mean(distances))
        print(f"\nMittelwert: {d_mean:.3f} µm")

        ax.text(
            0.01, 0.99,
            f"Ø Abstand: {d_mean:.1f} µm",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11,
            bbox=dict(
                facecolor="white",
                alpha=0.85,
                edgecolor="black",
                boxstyle="round,pad=0.4"
            )
        )

    ax.legend()
    plt.tight_layout()
    plt.show()

    return distances


def measure_horizontal_distances(surface, mode="sequential"):
    """
    Misst horizontale Abstände Δx zwischen angeklickten Punkten auf einer Oberfläche.

    Was kann man damit machen?
    - Beliebig viele Punkte auf der Oberfläche anklicken
    - Reine horizontale Abstände in µm auswerten
    - Abstände zwischen aufeinanderfolgenden Punkten oder relativ zum ersten Punkt bestimmen
    - Einzelwerte sowie Mittelwert, Minimum und Maximum direkt anzeigen

    Modi:
    - mode="sequential":
      Abstand zwischen Punkt 1->2, 2->3, 3->4, ...
    - mode="from_first":
      Abstand zwischen Punkt 1->2, 1->3, 1->4, ...

    Rückgabe:
    - NumPy-Array mit allen gemessenen horizontalen Abständen in µm
    """
    z = surface.data

    # Oberfläche darstellen
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        z,
        origin="upper",
        cmap="turbo",
        extent=[0, surface.width_um, surface.height_um, 0],
        aspect="auto"
    )
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("z [µm]")

    ax.set_title("Punkte anklicken (Enter beendet) – horizontale Abstände Δx")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")

    print("Punkte anklicken. Enter beendet.")
    pts = plt.ginput(n=-1, timeout=0)

    # Mindestens zwei Punkte nötig
    if len(pts) < 2:
        plt.close(fig)
        print("Mindestens 2 Punkte nötig.")
        return None

    # x- und y-Koordinaten extrahieren
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)

    # Angeclickte Punkte im Plot markieren
    ax.plot(xs, ys, "ro", markersize=5)

    distances = []

    if mode == "sequential":
        # Horizontale Abstände zwischen aufeinanderfolgenden Punkten
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]

            # Nur horizontale Distanz Δx
            dx = abs(x2 - x1)
            distances.append(dx)

            # Horizontale Hilfslinie einzeichnen
            ax.plot([x1, x2], [y1, y1], "w--", linewidth=1.5)

            # Distanzwert mittig beschriften
            xm = 0.5 * (x1 + x2)
            ax.text(
                xm, y1,
                f"{dx:.1f}",
                color="white",
                fontsize=9,
                ha="center",
                va="bottom",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.5)
            )

        label = "Δx (sequential)"

    elif mode == "from_first":
        # Alle horizontalen Abstände vom ersten Punkt aus
        x0, y0 = pts[0]
        for i in range(1, len(pts)):
            xi, yi = pts[i]
            dx = abs(xi - x0)
            distances.append(dx)

            # Horizontale Hilfslinie vom ersten Punkt
            ax.plot([x0, xi], [y0, y0], "w--", linewidth=1.5)

            # Distanzwert mittig beschriften
            xm = 0.5 * (x0 + xi)
            ax.text(
                xm, y0,
                f"{dx:.1f}",
                color="white",
                fontsize=9,
                ha="center",
                va="bottom",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.5)
            )

        label = "Δx (from first)"

    else:
        plt.close(fig)
        raise ValueError("mode muss 'sequential' oder 'from_first' sein.")

    distances = np.array(distances, dtype=float)

    # Statistische Zusammenfassung erzeugen
    txt = (
        f"{label}\n"
        f"n = {len(distances)}\n"
        f"Ø = {np.mean(distances):.3f} µm\n"
        f"min/max = {np.min(distances):.3f} / {np.max(distances):.3f} µm"
    )

    # Zusammenfassung im Plot anzeigen
    ax.text(
        0.01, 0.99, txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", boxstyle="round,pad=0.4")
    )

    plt.tight_layout()
    plt.show()

    # Ausgabe in der Konsole
    print(txt.replace("\n", " | "))
    print("Einzelwerte [µm]:", distances)

    return distances