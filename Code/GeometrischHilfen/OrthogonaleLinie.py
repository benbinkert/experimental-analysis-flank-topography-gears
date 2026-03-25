import numpy as np
import matplotlib.pyplot as plt


def _normalize(vx, vy):
    """
    Normiert einen 2D-Vektor.

    Was kann man damit machen?
    - Eine Richtungsinformation auf Länge 1 bringen
    - Aus einer beliebigen Linienrichtung einen Einheitsvektor erzeugen
    - Grundlage für Richtungs- und Normalenvektoren schaffen

    Parameter:
    - vx, vy:
      Komponenten des Vektors

    Rückgabe:
    - Normierter Vektor (vx, vy) mit Länge 1
    """
    n = np.hypot(vx, vy)
    if n == 0:
        raise ValueError("Nullvektor.")
    return vx / n, vy / n


def _clip_line_to_rect(point, direction, width_px, height_px):
    """
    Schneidet eine unendliche Gerade auf den Bildrand zurück.

    Was kann man damit machen?
    - Eine Gerade durch einen Punkt mit gegebener Richtung erzeugen
    - Diese Gerade genau auf das sichtbare Bild beschränken
    - Randpunkte bestimmen, um die Linie im Plot vollständig darzustellen

    Alles erfolgt in Pixelkoordinaten:
    - x nach rechts
    - y nach unten

    Parameter:
    - point:
      Punkt auf der Geraden in Pixelkoordinaten
    - direction:
      Richtungsvektor der Geraden in Pixelkoordinaten
    - width_px, height_px:
      Bildbreite und Bildhöhe in Pixel

    Rückgabe:
    - Zwei Randpunkte ((xA, yA), (xB, yB)) der sichtbaren Linie
    """
    x0, y0 = point
    dx, dy = direction
    pts = []

    # Schnitt mit x = 0 und x = width_px - 1
    if abs(dx) > 1e-12:
        t = (0 - x0) / dx
        y = y0 + t * dy
        if 0 <= y <= height_px - 1:
            pts.append((0.0, y))

        t = ((width_px - 1) - x0) / dx
        y = y0 + t * dy
        if 0 <= y <= height_px - 1:
            pts.append((width_px - 1.0, y))

    # Schnitt mit y = 0 und y = height_px - 1
    if abs(dy) > 1e-12:
        t = (0 - y0) / dy
        x = x0 + t * dx
        if 0 <= x <= width_px - 1:
            pts.append((x, 0.0))

        t = ((height_px - 1) - y0) / dy
        x = x0 + t * dx
        if 0 <= x <= width_px - 1:
            pts.append((x, height_px - 1.0))

    # Doppelte Schnittpunkte entfernen
    unique = []
    for p in pts:
        if not any(np.hypot(p[0] - q[0], p[1] - q[1]) < 1e-9 for q in unique):
            unique.append(p)

    if len(unique) < 2:
        raise ValueError("Gerade schneidet den Rand nicht sauber.")

    # Zwei am weitesten entfernte Schnittpunkte wählen
    best = None
    best_d = -1
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            d = np.hypot(unique[j][0] - unique[i][0], unique[j][1] - unique[i][1])
            if d > best_d:
                best_d = d
                best = (unique[i], unique[j])

    a, b = best

    # Punkte entlang der Geradenrichtung konsistent sortieren
    ta = a[0] * dx + a[1] * dy
    tb = b[0] * dx + b[1] * dy
    if ta <= tb:
        return a, b
    else:
        return b, a


def _project_point_to_parallel_family(point, ref_point, normal):
    """
    Berechnet die Projektion eines Punkts auf die Normalenrichtung einer Parallelschar.

    Was kann man damit machen?
    - Den relativen Abstand eines Punkts zu einer Referenzlinie bestimmen
    - Mehrere geklickte Linien nach ihrer Lage entlang der Normalen sortieren
    - Eine Parallelschar geometrisch beschreiben

    Alles erfolgt in Pixelkoordinaten.

    Parameter:
    - point:
      Zu projizierender Punkt
    - ref_point:
      Referenzpunkt auf der ersten Linie
    - normal:
      Normalenvektor zur Linienrichtung

    Rückgabe:
    - Signierter Abstand entlang der Normalenrichtung
    """
    x, y = point
    x0, y0 = ref_point
    nx, ny = normal
    return (x - x0) * nx + (y - y0) * ny


def _um_to_px(surface, x_um, y_um):
    """
    Wandelt Plot-/surfalize-Koordinaten in Pixelkoordinaten um.

    Was kann man damit machen?
    - Geklickte Punkte aus dem Plot in Arraykoordinaten überführen
    - Zwischen physikalischem Koordinatensystem und Bildlogik wechseln
    - Interaktive Eingaben für interne Berechnungen nutzbar machen

    Annahme:
    - Plot: Ursprung unten links
    - Array: Ursprung oben links
    """
    x_px = x_um / surface.step_x
    y_px = (surface.height_um - y_um) / surface.step_y
    return x_px, y_px


def _px_to_um(surface, x_px, y_px):
    """
    Wandelt Pixelkoordinaten in Plot-/surfalize-Koordinaten um.

    Was kann man damit machen?
    - Interne Pixelpunkte wieder in physikalische Koordinaten zurückführen
    - Randpunkte und Linien im Plot korrekt darstellen
    - Ergebnisse in µm ausgeben

    Annahme:
    - Array: Ursprung oben links
    - Plot: Ursprung unten links
    """
    x_um = x_px * surface.step_x
    y_um = surface.height_um - y_px * surface.step_y
    return x_um, y_um


def _sort_points_left_to_right_um(p1_um, p2_um):
    """
    Sortiert zwei Punkte konsistent von links nach rechts.

    Was kann man damit machen?
    - Profileingaben in stabiler Reihenfolge übergeben
    - Unabhängig von der Klickreihenfolge eine einheitliche Orientierung erzwingen
    - Darstellung und Weiterverarbeitung robuster machen

    Bei gleichem x wird nach y sortiert.
    """
    if (p1_um[0], p1_um[1]) <= (p2_um[0], p2_um[1]):
        return p1_um, p2_um
    return p2_um, p1_um


def draw_parallelized_lines_and_user_normal(surface, show_profile=True):
    """
    Klickt mehrere Linien an, parallelisiert sie zur ersten Linie
    und erzeugt anschließend eine Orthogonale durch einen frei gewählten Punkt.

    Was kann man damit machen?
    - Mehrere manuell gesetzte Linien auf dieselbe Referenzrichtung bringen
    - Eine Parallelschar auf der Oberfläche visualisieren
    - Eine Orthogonale zur Parallelschar durch einen Benutzerpunkt legen
    - Optional direkt ein Profil entlang dieser Orthogonalen extrahieren

    Ablauf:
    1. Im ersten Plot je Linie zwei Punkte anklicken
    2. Alle Linien werden parallel zur ersten Linie ausgerichtet
    3. Im zweiten Plot einen Punkt für die Orthogonale anklicken
    4. Die Orthogonale wird eingezeichnet
    5. Optional wird entlang dieser Orthogonalen ein Profil erzeugt

    Rückgabe:
    - Dictionary mit Linien, Richtungen, Orthogonale und optionalem Profil
    """
    z = surface.data
    h, w = z.shape

    # ---------------- Plot 1: Linien anklicken ----------------
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    im1 = ax1.imshow(
        z,
        cmap="turbo",
        extent=(0, surface.width_um, 0, surface.height_um),
        aspect="auto"
    )
    plt.colorbar(im1, ax=ax1, label="z [µm]")
    ax1.set_title("Je Linie 2 Punkte klicken, Enter beendet")
    ax1.set_xlabel("x [µm]")
    ax1.set_ylabel("y [µm]")

    print("Je Linie 2 Punkte klicken. Enter beendet.")
    pts_um = plt.ginput(n=-1, timeout=0)
    plt.close(fig1)

    # Mindestens zwei Linien = vier Klicks
    if len(pts_um) < 4:
        print("Mindestens 2 Linien (= 4 Klicks) nötig.")
        return None

    # Bei ungerader Punktzahl den letzten Punkt ignorieren
    if len(pts_um) % 2 != 0:
        print("Ungerade Zahl an Klicks -> letzter Punkt wird ignoriert.")
        pts_um = pts_um[:-1]

    # Geklickte Punkte in Pixelkoordinaten umrechnen
    pts_px = [_um_to_px(surface, xu, yu) for xu, yu in pts_um]

    # Aus je zwei Punkten eine Linie bilden
    clicked_lines = []
    for i in range(0, len(pts_px), 2):
        clicked_lines.append((pts_px[i], pts_px[i + 1]))

    # Referenzrichtung aus der ersten Linie bestimmen
    (p1, p2) = clicked_lines[0]
    ref_dx = p2[0] - p1[0]
    ref_dy = p2[1] - p1[1]

    # Einheitsvektor in Linienrichtung
    ux, uy = _normalize(ref_dx, ref_dy)

    # Zugehörige Normale
    nx, ny = -uy, ux

    # Alle geklickten Linien zur Referenzrichtung parallelisieren
    parallel_lines = []
    for line in clicked_lines:
        (a, b) = line

        # Mittelpunkt der geklickten Linie
        mx = 0.5 * (a[0] + b[0])
        my = 0.5 * (a[1] + b[1])

        # Relativer Versatz entlang der Normalen
        offset = _project_point_to_parallel_family((mx, my), p1, (nx, ny))

        # Sichtbare Randpunkte der parallelisierten Linie
        edge_pts = _clip_line_to_rect((mx, my), (ux, uy), w, h)

        parallel_lines.append({
            "mid_px": (mx, my),
            "offset_px": offset,
            "edge_points_px": edge_pts,
        })

    # Linien entlang der Normalen sortieren
    parallel_lines.sort(key=lambda item: item["offset_px"])

    # ---------------- Plot 2: Parallelisierte Linien + Orthogonale ----------------
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    im2 = ax2.imshow(
        z,
        cmap="turbo",
        extent=(0, surface.width_um, 0, surface.height_um),
        aspect="equal"
    )
    plt.colorbar(im2, ax=ax2, label="z [µm]")
    ax2.set_title("Parallelisierte Linien; klicke 1 Punkt für die Orthogonale")
    ax2.set_xlabel("x [µm]")
    ax2.set_ylabel("y [µm]")

    # Parallelisierte Linien einzeichnen
    for i, item in enumerate(parallel_lines):
        (a_px, b_px) = item["edge_points_px"]
        a_um = _px_to_um(surface, a_px[0], a_px[1])
        b_um = _px_to_um(surface, b_px[0], b_px[1])

        a_um, b_um = _sort_points_left_to_right_um(a_um, b_um)

        ax2.plot(
            [a_um[0], b_um[0]],
            [a_um[1], b_um[1]],
            linewidth=2,
            label=f"Linie {i+1}"
        )

    ax2.legend()
    plt.draw()

    print("Klicke im zweiten Plot 1 Punkt für die Orthogonale.")
    clicked = plt.ginput(n=1, timeout=0)

    if len(clicked) != 1:
        plt.close(fig2)
        print("Kein Punkt gewählt.")
        return None

    # Gewählten Punkt in µm und Pixel speichern
    x_um, y_um = clicked[0]
    x_px, y_px = _um_to_px(surface, x_um, y_um)

    # Orthogonale durch den gewählten Punkt bestimmen
    orth_edge_pts_px = _clip_line_to_rect((x_px, y_px), (nx, ny), w, h)
    oa_px, ob_px = orth_edge_pts_px

    # In µm umrechnen
    oa_um = _px_to_um(surface, oa_px[0], oa_px[1])
    ob_um = _px_to_um(surface, ob_px[0], ob_px[1])

    # Für Profilberechnung konsistent sortieren
    oa_um_sorted, ob_um_sorted = _sort_points_left_to_right_um(oa_um, ob_um)

    # Orthogonale einzeichnen
    ax2.plot(
        [oa_um_sorted[0], ob_um_sorted[0]],
        [oa_um_sorted[1], ob_um_sorted[1]],
        linestyle="--",
        linewidth=2
    )

    # Benutzerpunkt und Randpunkte markieren
    ax2.plot(x_um, y_um, marker="o")
    ax2.plot(oa_um_sorted[0], oa_um_sorted[1], marker="s")
    ax2.plot(ob_um_sorted[0], ob_um_sorted[1], marker="s")

    plt.tight_layout()
    plt.show()

    # Optional Profil entlang der Orthogonalen erzeugen
    profile = None
    if show_profile:
        try:
            profile = surface.get_oblique_profile_fixed(
                oa_um_sorted[0], oa_um_sorted[1],
                ob_um_sorted[0], ob_um_sorted[1]
            )
            profile.show()
        except Exception as e:
            print(f"Profil konnte nicht erzeugt werden: {e}")

    # Ergebnisse ausgeben
    print("\nParallele Linien (Randpunkte):")
    for i, item in enumerate(parallel_lines):
        a_px, b_px = item["edge_points_px"]
        a_um = _px_to_um(surface, a_px[0], a_px[1])
        b_um = _px_to_um(surface, b_px[0], b_px[1])
        a_um, b_um = _sort_points_left_to_right_um(a_um, b_um)

        print(f"Linie {i+1}:")
        print(f"  Pixel: {a_px} -> {b_px}")
        print(f"  µm: ({a_um[0]:.3f}, {a_um[1]:.3f}) -> ({b_um[0]:.3f}, {b_um[1]:.3f})")

    print("\nOrthogonale durch den gewählten Punkt:")
    print(f"  Gewählter Punkt Pixel: ({x_px:.3f}, {y_px:.3f})")
    print(f"  Gewählter Punkt µm: ({x_um:.3f}, {y_um:.3f})")
    print(f"  Schnittpunkte Pixel: {oa_px} -> {ob_px}")
    print(
        f"  Schnittpunkte µm: ({oa_um_sorted[0]:.3f}, {oa_um_sorted[1]:.3f})"
        f" -> ({ob_um_sorted[0]:.3f}, {ob_um_sorted[1]:.3f})"
        f"    ,COPY: {oa_um_sorted[0]:.0f}, {oa_um_sorted[1]:.0f},{ob_um_sorted[0]:.0f}, {ob_um_sorted[1]:.0f}"
    )

    return {
        "parallel_lines": parallel_lines,
        "reference_direction_px": (ux, uy),
        "reference_normal_px": (nx, ny),
        "selected_point_px": (x_px, y_px),
        "selected_point_um": (x_um, y_um),
        "orthogonal_line_edge_points_px": orth_edge_pts_px,
        "orthogonal_line_edge_points_um": (oa_um_sorted, ob_um_sorted),
        "profile": profile,
    }


# ============================================================
# Für Simulationen:
# Alles direkt in µm rechnen, da rechteckige Felder
# wie z. B. 25 x 100 µm verwendet werden
# ============================================================

def _normalize_um(vx, vy):
    """
    Normiert einen 2D-Vektor in physikalischen Koordinaten.

    Was kann man damit machen?
    - Eine Richtung direkt in µm beschreiben
    - Einen physikalisch korrekten Einheitsvektor erzeugen
    - Grundlage für echte Orthogonalität in Simulationen schaffen
    """
    n = np.hypot(vx, vy)
    if n == 0:
        raise ValueError("Nullvektor.")
    return vx / n, vy / n


def _clip_line_to_rect_um(point, direction, width_um, height_um):
    """
    Schneidet eine unendliche Gerade auf ein Rechteck in µm zurück.

    Was kann man damit machen?
    - Linien direkt im physikalischen Koordinatensystem beschreiben
    - Randpunkte einer Linie auf einem Simulationsfeld bestimmen
    - Rechteckige Felder korrekt behandeln, ohne Pixelverzerrung

    Parameter:
    - point:
      Punkt auf der Geraden in µm
    - direction:
      Richtungsvektor der Geraden in µm
    - width_um, height_um:
      Breite und Höhe des Rechtecks in µm

    Rückgabe:
    - Zwei Randpunkte ((xA, yA), (xB, yB)) in µm
    """
    x0, y0 = point
    dx, dy = direction
    pts = []

    # Schnitt mit x = 0 und x = width_um
    if abs(dx) > 1e-12:
        t = (0 - x0) / dx
        y = y0 + t * dy
        if 0 <= y <= height_um:
            pts.append((0.0, y))

        t = (width_um - x0) / dx
        y = y0 + t * dy
        if 0 <= y <= height_um:
            pts.append((width_um, y))

    # Schnitt mit y = 0 und y = height_um
    if abs(dy) > 1e-12:
        t = (0 - y0) / dy
        x = x0 + t * dx
        if 0 <= x <= width_um:
            pts.append((x, 0.0))

        t = (height_um - y0) / dy
        x = x0 + t * dx
        if 0 <= x <= width_um:
            pts.append((x, height_um))

    # Doppelte entfernen
    unique = []
    for p in pts:
        if not any(np.hypot(p[0] - q[0], p[1] - q[1]) < 1e-9 for q in unique):
            unique.append(p)

    if len(unique) < 2:
        raise ValueError("Gerade schneidet den Rand nicht sauber.")

    # Zwei am weitesten entfernte Punkte wählen
    best = None
    best_d = -1
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            d = np.hypot(unique[j][0] - unique[i][0], unique[j][1] - unique[i][1])
            if d > best_d:
                best_d = d
                best = (unique[i], unique[j])

    return best


def draw_parallelized_lines_and_user_normal_sim(surface, show_profile=True):
    """
    Simulationvariante der Linien- und Orthogonalenkonstruktion in µm.

    Was kann man damit machen?
    - Linien direkt in physikalischen Koordinaten anklicken
    - Alle Linien parallel zur ersten Linie ausrichten
    - Eine wirklich orthogonale Gerade im µm-System erzeugen
    - Rechteckige Simulationsfelder korrekt behandeln
    - Optional ein Profil entlang der Orthogonalen extrahieren

    Unterschied zur Pixelvariante:
    - Alle geometrischen Rechnungen laufen direkt in µm
    - Dadurch bleibt die Orthogonale auch bei nichtquadratischen Feldern korrekt
    """
    z = surface.data

    # ---------- Plot 1: Linien anklicken ----------
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    im1 = ax1.imshow(
        z,
        cmap="turbo",
        extent=(0, surface.width_um, 0, surface.height_um),
        aspect="auto"
    )
    plt.colorbar(im1, ax=ax1, label="z [µm]")
    ax1.set_title("Je Linie 2 Punkte klicken, Enter beendet")
    ax1.set_xlabel("x [µm]")
    ax1.set_ylabel("y [µm]")
    ax1.set_xlim(0, surface.width_um)
    ax1.set_ylim(0, surface.height_um)

    print("Je Linie 2 Punkte klicken. Enter beendet.")
    pts_um = plt.ginput(n=-1, timeout=0)
    plt.close(fig1)

    # Mindestens zwei Linien = vier Punkte
    if len(pts_um) < 4:
        print("Mindestens 2 Linien (= 4 Klicks) nötig.")
        return None

    # Ungerade Zahl an Klicks korrigieren
    if len(pts_um) % 2 != 0:
        print("Ungerade Zahl an Klicks -> letzter Punkt wird ignoriert.")
        pts_um = pts_um[:-1]

    # Aus je zwei Punkten eine Linie bilden
    clicked_lines = []
    for i in range(0, len(pts_um), 2):
        clicked_lines.append((pts_um[i], pts_um[i + 1]))

    # Referenzrichtung aus der ersten Linie in µm bestimmen
    (p1, p2) = clicked_lines[0]
    ref_dx = p2[0] - p1[0]
    ref_dy = p2[1] - p1[1]
    ux, uy = _normalize_um(ref_dx, ref_dy)

    # Physikalisch korrekte Normale
    nx, ny = -uy, ux

    # Parallele Linien erzeugen
    parallel_lines = []
    for (a, b) in clicked_lines:
        mx = 0.5 * (a[0] + b[0])
        my = 0.5 * (a[1] + b[1])

        # Signierter Offset entlang der Normalen
        offset = (mx - p1[0]) * nx + (my - p1[1]) * ny

        edge_pts = _clip_line_to_rect_um(
            (mx, my), (ux, uy), surface.width_um, surface.height_um
        )

        parallel_lines.append({
            "mid_um": (mx, my),
            "offset_um": offset,
            "edge_points_um": edge_pts
        })

    # Linien entlang der Normalen sortieren
    parallel_lines.sort(key=lambda item: item["offset_um"])

    # ---------- Plot 2: Parallelisierte Linien + Orthogonale ----------
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    im2 = ax2.imshow(
        z,
        cmap="turbo",
        extent=(0, surface.width_um, 0, surface.height_um),
        aspect="equal"
    )
    plt.colorbar(im2, ax=ax2, label="z [µm]")
    ax2.set_title("Parallelisierte Linien; klicke 1 Punkt für die Orthogonale")
    ax2.set_xlabel("x [µm]")
    ax2.set_ylabel("y [µm]")
    ax2.set_xlim(0, surface.width_um)
    ax2.set_ylim(0, surface.height_um)

    # Parallelisierte Linien darstellen
    for i, item in enumerate(parallel_lines):
        (a, b) = item["edge_points_um"]
        ax2.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            linewidth=2,
            label=f"Linie {i+1}"
        )

    ax2.legend()
    plt.draw()

    print("Klicke im zweiten Plot 1 Punkt für die Orthogonale.")
    clicked = plt.ginput(n=1, timeout=0)
    if len(clicked) != 1:
        plt.close(fig2)
        print("Kein Punkt gewählt.")
        return None

    # Gewählten Punkt speichern
    cx, cy = clicked[0]

    # Orthogonale durch den Benutzerpunkt bestimmen
    orth_edge_pts_um = _clip_line_to_rect_um(
        (cx, cy), (nx, ny), surface.width_um, surface.height_um
    )
    oa, ob = orth_edge_pts_um

    # Orthogonale einzeichnen
    ax2.plot(
        [oa[0], ob[0]],
        [oa[1], ob[1]],
        linestyle="--",
        linewidth=2,
        color="green"
    )

    # Punkt und Randpunkte markieren
    ax2.plot(cx, cy, marker="o", color="red")
    ax2.plot(oa[0], oa[1], marker="s", color="black")
    ax2.plot(ob[0], ob[1], marker="s", color="black")

    plt.tight_layout()
    plt.show()

    # Optional Profil entlang der Orthogonalen erzeugen
    profile = None
    if show_profile:
        try:
            profile = surface.get_oblique_profile_fixed(
                oa[0], oa[1], ob[0], ob[1]
            )
            profile.show()
        except Exception as e:
            print(f"Profil konnte nicht erzeugt werden: {e}")

    # Ergebnisse ausgeben
    print("\nOrthogonale:")
    print(f"  Punkt: ({cx:.3f}, {cy:.3f}) µm")
    print(
        f"  Randpunkte: ({oa[0]:.3f}, {oa[1]:.3f}) -> ({ob[0]:.3f}, {ob[1]:.3f}) µm"
        f"    ,COPY: {oa[0]:.0f}, {oa[1]:.0f},{ob[0]:.0f}, {ob[1]:.0f}"
    )

    return {
        "parallel_lines": parallel_lines,
        "reference_direction_um": (ux, uy),
        "reference_normal_um": (nx, ny),
        "selected_point_um": (cx, cy),
        "orthogonal_line_edge_points_um": (oa, ob),
        "profile": profile,
    }