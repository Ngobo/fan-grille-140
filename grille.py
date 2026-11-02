"""Решётка для вентилятора ARCTIC P14 в отверстие Ø140 мм (ДСП 16 мм).

Запуск:  .venv/bin/python grille.py [A] [B]   (без аргументов — оба варианта)
  A — «Кольца»: концентрические щели с перемычками в шахматном порядке
  B — «Вихрь»: изогнутые лучи и два кольца жёсткости
Результат: grille_140_<вариант>.3mf и .stl рядом со скриптом.

Модель стоит лицевой стороной на z = 0 (так её и печатать), втулка смотрит вверх.
"""
import ctypes
import math
import sys
from pathlib import Path

import lib3mf
import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold, Mesh, triangulate

# --- Параметры, мм ---------------------------------------------------------
HOLE_D = 140.0          # диаметр отверстия в ДСП
GLUE_GAP = 0.2          # зазор под клей на сторону
SLEEVE_WALL = 2.0       # толщина стенки втулки
SLEEVE_DEPTH = 13.0     # насколько втулка заходит в отверстие (ДСП 16)
FLANGE_D = 164.0        # наружный диаметр фланца
FLANGE_T = 3.0          # толщина фланца и решётки
CHAMFER = 0.6           # фаска на лицевой кромке фланца

HUB_D = 40.0            # центральный диск (мотор P14 — Ø50)
RIB = 1.6               # толщина рёбер и перемычек
RIB_CHAMFER = 0.4       # фаска по кромкам щелей с обеих сторон (0 — без фаски)

# Вариант A
RINGS = 6               # число кольцевых щелей
BRIDGE_STEP = 30.0      # примерный шаг перемычек по окружности

# Вариант B: зоны между кольцами жёсткости и число лучей в каждой
SWIRL_RINGS = (34.0, 51.0)      # радиусы осей колец жёсткости
SWIRL_SPOKES = (18, 27, 36)     # лучей во внутренней, средней и внешней зоне
SWIRL_DEG = 31.5                # закрутка луча от диска до втулки; знак — направление

SEGMENTS = 360          # гладкость окружностей
# ---------------------------------------------------------------------------

SLEEVE_OD = HOLE_D - 2 * GLUE_GAP
R_IN = SLEEVE_OD / 2 - SLEEVE_WALL
R_OUT = SLEEVE_OD / 2
R_FL = FLANGE_D / 2
R_HUB = HUB_D / 2


# Внешняя щель заходит на SLOT_UNDERCUT под стенку втулки: если её край совпадёт
# со стенкой точно, булевы операции дают касающиеся грани и сетка не watertight.
SLOT_UNDERCUT = 0.4
SLOT_R_MAX = R_IN + SLOT_UNDERCUT


def slot_width() -> float:
    return (SLOT_R_MAX - R_HUB - (RINGS - 1) * RIB) / RINGS


def disk() -> Manifold:
    """Сплошной диск фланца с фаской по лицевой кромке."""
    profile = [(0, 0), (R_FL - CHAMFER, 0), (R_FL, CHAMFER), (R_FL, FLANGE_T), (0, FLANGE_T)]
    return Manifold.revolve(CrossSection([profile]), circular_segments=SEGMENTS)


def sleeve() -> Manifold:
    """Втулка; начинается чуть ниже верха решётки, чтобы тела перекрывались, а не касались."""
    z0, top = FLANGE_T - 0.1, FLANGE_T + SLEEVE_DEPTH
    profile = [(R_IN, z0), (R_OUT, z0), (R_OUT, top), (R_IN, top)]
    return Manifold.revolve(CrossSection([profile]), circular_segments=SEGMENTS)


def plate(openings: CrossSection) -> Manifold:
    """Фланец и решётка одной пластиной; щели прорезаны насквозь с запасом по z."""
    return disk() - chamfered_cut(openings)


def chamfered_cut(openings: CrossSection) -> Manifold:
    """Тело выреза щелей с фаской 45° по кромкам на обеих сторонах решётки.

    Каждая щель — отдельная призма: на лицевых сторонах её контур раздвинут на
    RIB_CHAMFER по биссектрисам углов, в толще пластины он равен самой щели.
    """
    c = RIB_CHAMFER
    if c <= 0:
        return Manifold.extrude(openings, FLANGE_T + 2).translate((0, 0, -1))
    layers = [(-1, c), (0, c), (c, 0), (FLANGE_T - c, 0), (FLANGE_T, c), (FLANGE_T + 1, c)]
    prisms = [loft(np.asarray(part.to_polygons()[0]), layers) for part in openings.decompose()]
    return Manifold.compose(prisms)


def loft(poly: np.ndarray, layers) -> Manifold:
    """Призма по контуру poly (против часовой стрелки) через слои (z, отступ наружу)."""
    n = len(poly)
    edge = np.roll(poly, -1, axis=0) - poly
    normal = np.stack([edge[:, 1], -edge[:, 0]], axis=1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    prev = np.roll(normal, 1, axis=0)
    miter = (normal + prev) / (1 + np.sum(normal * prev, axis=1, keepdims=True))

    verts = np.vstack([np.column_stack([poly + miter * d, np.full(n, z)]) for z, d in layers])
    j = np.arange(n)
    tris = []
    for i in range(len(layers) - 1):
        a, b = i * n + j, i * n + (j + 1) % n
        top_a, top_b = a + n, b + n
        tris += [np.stack([a, b, top_b], 1), np.stack([a, top_b, top_a], 1)]
    cap = triangulate([poly])
    tris += [cap[:, ::-1], cap + (len(layers) - 1) * n]
    mesh = Mesh(vert_properties=verts.astype(np.float32), tri_verts=np.vstack(tris).astype(np.uint32))
    return Manifold(mesh)


def slots_rings() -> CrossSection:
    """Кольцевые щели, разрезанные перемычками в шахматном порядке."""
    w = slot_width()
    out = CrossSection()
    prev = np.array([])
    for i in range(RINGS):
        r0 = R_HUB + i * (w + RIB)
        r1 = r0 + w
        ring = (CrossSection.circle(r1, SEGMENTS) - CrossSection.circle(r0, SEGMENTS))
        n = max(3, round(2 * math.pi * (r0 + r1) / 2 / BRIDGE_STEP))
        angles = stagger(n, prev)
        bar = CrossSection.square((r1 + 2, RIB)).translate((0, -RIB / 2))
        for a in angles:
            ring = ring - bar.rotate(a)
        out = out + ring
        prev = angles
    return out


def stagger(n: int, prev: np.ndarray) -> np.ndarray:
    """Углы n перемычек, повёрнутые как можно дальше от перемычек соседнего кольца."""
    step = 360.0 / n
    if prev.size == 0:
        return np.arange(n) * step + 90.0

    def clearance(base):
        a = base + np.arange(n) * step
        diff = np.abs((a[:, None] - prev[None, :] + 180) % 360 - 180)
        return diff.min()

    best = max(np.linspace(0, step, 181, endpoint=False), key=clearance)
    return best + np.arange(n) * step


def slots_swirl() -> CrossSection:
    """Кольцевой проём, из которого вычтены изогнутые лучи и кольца жёсткости."""
    out = CrossSection.circle(SLOT_R_MAX, SEGMENTS) - CrossSection.circle(R_HUB, SEGMENTS)
    for r in SWIRL_RINGS:
        out = out - (CrossSection.circle(r + RIB / 2, SEGMENTS) - CrossSection.circle(r - RIB / 2, SEGMENTS))
    bounds = (R_HUB - 1, *SWIRL_RINGS, SLOT_R_MAX + 1)
    for (ra, rb), n in zip(zip(bounds, bounds[1:]), SWIRL_SPOKES):
        for j in range(n):
            out = out - spoke(ra, rb, 360.0 * j / n)
    return out


def spoke(ra: float, rb: float, a0: float) -> CrossSection:
    """Луч шириной RIB вдоль спирали от радиуса ra до rb."""
    r = np.linspace(ra, rb, 24)
    a = np.radians(a0 + SWIRL_DEG * (r - R_HUB) / (R_IN - R_HUB))
    pts = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
    tang = np.gradient(pts, axis=0)
    norm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    norm /= np.linalg.norm(norm, axis=1, keepdims=True)
    left, right = pts + norm * RIB / 2, pts - norm * RIB / 2
    outline = np.vstack([right, left[::-1]])   # против часовой стрелки, иначе это «дырка»
    return CrossSection([outline.tolist()])


STYLES = {"A": slots_rings, "B": slots_swirl}


def build(style: str) -> Manifold:
    return plate(STYLES[style]()) + sleeve()


def open_share(style: str) -> float:
    """Доля открытой площади в проёме втулки."""
    area = (STYLES[style]() ^ CrossSection.circle(R_IN, SEGMENTS)).area()
    return area / (math.pi * R_IN ** 2)


def to_trimesh(m: Manifold) -> trimesh.Trimesh:
    mesh = m.to_mesh()
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vert_properties)[:, :3],
        faces=np.asarray(mesh.tri_verts),
        process=True,
    )


def export_3mf(mesh: trimesh.Trimesh, path: Path, name: str):
    """3MF через lib3mf — эталонную библиотеку консорциума 3MF.
    Экспорт trimesh пишет атрибуты расширения Production, на которых
    lib3mf в строгом режиме падает с «Invalid Attribute in namespace»."""
    w = lib3mf.get_wrapper()
    model = w.CreateModel()
    model.SetUnit(lib3mf.ModelUnit.MilliMeter)
    obj = model.AddMeshObject()
    obj.SetName(name)
    verts = [lib3mf.Position((ctypes.c_float * 3)(*v)) for v in mesh.vertices.tolist()]
    tris = [lib3mf.Triangle((ctypes.c_uint32 * 3)(*f)) for f in mesh.faces.tolist()]
    obj.SetGeometry(verts, tris)
    model.AddBuildItem(obj, w.GetIdentityTransform())
    model.QueryWriter("3mf").WriteToFile(str(path))


def main():
    here = Path(__file__).parent
    for style in sys.argv[1:] or STYLES:
        mesh = to_trimesh(build(style))
        mesh.export(here / f"grille_140_{style}.stl")
        export_3mf(mesh, here / f"grille_140_{style}.3mf", f"grille_140_{style}")
        print(f"{style}: watertight={mesh.is_watertight}  volume={mesh.volume / 1000:.1f} см³  "
              f"bounds={mesh.extents.round(2)}  open={open_share(style):.0%}")


if __name__ == "__main__":
    main()
