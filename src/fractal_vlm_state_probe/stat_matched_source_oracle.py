"""Independent scalar payload/rank and fixed-reference numerical checks."""

import math

import numpy as np


def quartet(a, b, spec, geometry, shift):
    c = geometry["construction"]
    source_pixels = []
    for image in (a, b):
        pixels = []
        for r in range(c["active_grid_rows"][1] - c["active_grid_rows"][0]):
            for col in range(c["active_grid_cols"][1] - c["active_grid_cols"][0]):
                for dy, dx in c["tile_offsets_yx"]:
                    for u in range(c["tile_shape_hw"][0]):
                        for v in range(c["tile_shape_hw"][1]):
                            pixels.append(
                                tuple(
                                    int(x)
                                    for x in image[
                                        r * c["cell_shape_hw"][0] + dy + u,
                                        col * c["cell_shape_hw"][1] + dx + v,
                                    ]
                                )
                            )
        source_pixels.append(pixels)

    def rank(pixels):
        return sorted(
            range(len(pixels)),
            key=lambda i: (
                (pixels[i][0] * 0.2126 + pixels[i][1] * 0.7152) + pixels[i][2] * 0.0722
            ),
        )

    def cross(target, palette):
        out = [None] * len(target)
        for t, p in zip(rank(target), rank(palette), strict=True):
            out[t] = palette[p]
        return out

    pa, pb = source_pixels
    lists = {"mm": pa, "mj": cross(pa, pb), "jm": cross(pb, pa), "jj": pb}
    result = {}
    th, tw = c["tile_shape_hw"]
    for cell, pixels in lists.items():
        out = np.empty(spec["construction"]["raw_shape_hwc"], dtype=np.uint8)
        out[:] = spec["construction"]["background_rgb"]
        nr = c["active_grid_rows"][1] - c["active_grid_rows"][0]
        nc = c["active_grid_cols"][1] - c["active_grid_cols"][0]
        for r in range(nr):
            for col in range(nc):
                for target, (dy, dx) in enumerate(c["tile_offsets_yx"]):
                    donor = (target + shift) % 4
                    for u in range(th):
                        for v in range(tw):
                            i = (((r * nc + col) * 4 + donor) * th + u) * tw + v
                            y = (
                                (r + c["active_grid_rows"][0]) * c["cell_shape_hw"][0]
                                + dy
                                + u
                            )
                            x = (
                                (col + c["active_grid_cols"][0]) * c["cell_shape_hw"][1]
                                + dx
                                + v
                            )
                            out[y, x] = pixels[i]
        result[cell] = out
    return result


def direction(cells, learned, floor):
    values = {k: np.asarray(v, dtype=float).ravel().tolist() for k, v in cells.items()}
    z = [
        (j - m) - (n - a)
        for j, m, n, a in zip(
            values["jj"], values["mj"], values["jm"], values["mm"], strict=True
        )
    ]
    norm = math.hypot(*z)
    threshold = (
        floor * sum(math.hypot(*v) for v in values.values())
        if learned
        else 100 * 4e-10 * math.sqrt(len(z))
    )
    return [x / norm for x in z] if norm > threshold else None


def profile(load_cells, spec, view, floor=1e-6):
    pairs = spec["sources"]["pairs"]
    families = list(dict.fromkeys(p["pairing_family"] for p in pairs))
    classes = {p["pairing_family"]: p["broad_class"] for p in pairs}
    refs = {}
    for family in families:
        units = [
            direction(
                load_cells(p["pair_id"], 0, view),
                view in ("encoder", "projector"),
                floor,
            )
            for p in pairs
            if p["split"] == "reference" and p["pairing_family"] == family
        ]
        if any(x is None for x in units):
            refs[family] = None
            continue
        combined = [x + y for x, y in zip(*units, strict=True)]
        norm = math.hypot(*combined)
        refs[family] = [v / norm for v in combined] if norm > 1e-12 else None
    tests = [p for p in pairs if p["split"] == "test"]
    scores, margins = [], []
    for shift in range(4):
        scores.append([])
        margins.append([])
        for pair in tests:
            unit = direction(
                load_cells(pair["pair_id"], shift, view),
                view in ("encoder", "projector"),
                floor,
            )
            row = [
                math.fsum(x * y for x, y in zip(unit, refs[f], strict=True))
                if unit is not None and refs[f] is not None
                else None
                for f in families
            ]
            scores[-1].append(row)
            own = families.index(pair["pairing_family"])
            others = [
                i
                for i, f in enumerate(families)
                if i != own and classes[f] == pair["broad_class"]
            ]
            margin = (
                None
                if any(row[i] is None for i in (own, *others))
                else row[own] - math.fsum(row[i] for i in others) / 3
            )
            margins[-1].append(margin)
    differences = []
    for i in range(16):
        values = [margins[s][i] for s in range(4)]
        differences.append(
            None
            if any(v is None for v in values)
            else math.fsum(values[1:]) / 3 - values[0]
        )
    result = (
        None if any(x is None for x in differences) else math.fsum(differences) / 16
    )
    return {"scores": scores, "margins": margins, "paired_margin_change": result}
