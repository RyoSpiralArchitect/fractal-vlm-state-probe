"""Independent explicit-site map and scalar fixture reconstruction."""

import numpy as np


def coordinate_map(spec, shift):
    c = spec["construction"]
    height, width, _ = c["raw_shape_hwc"]
    successors = dict.fromkeys(range(height * width))
    for site in successors:
        successors[site] = site
    for row in range(c["active_grid_rows"][0], c["active_grid_rows"][1]):
        for col in range(c["active_grid_cols"][0], c["active_grid_cols"][1]):
            cy = row * c["cell_shape_hw"][0]
            cx = col * c["cell_shape_hw"][1]
            for target in range(4):
                donor = (target + shift) % 4
                dst, src = c["tile_offsets_yx"][target], c["tile_offsets_yx"][donor]
                for y in range(c["tile_shape_hw"][0]):
                    for x in range(c["tile_shape_hw"][1]):
                        successors[(cy + dst[0] + y) * width + cx + dst[1] + x] = (
                            (cy + src[0] + y) * width + cx + src[1] + x
                        )
    return np.fromiter(
        (successors[i] for i in range(height * width)), dtype="<u4"
    ).reshape(height, width)


def fixture(spec, name, shift, sentinel=False):
    c, f = spec["construction"], spec["fixtures"]
    h, w, _ = c["raw_shape_hwc"]
    out = np.empty((h, w, 3), dtype=np.uint8)
    out[:] = f["background_rgb"][name]
    for row in range(*c["active_grid_rows"]):
        for col in range(*c["active_grid_cols"]):
            r = row - c["active_grid_rows"][0]
            cc = col - c["active_grid_cols"][0]
            for target in range(4):
                k = (target + shift) % 4
                noise = None
                if name == "noise":
                    noise = np.random.Generator(
                        np.random.PCG64(
                            np.random.SeedSequence([f["noise_seed"], r, cc, k])
                        )
                    ).integers(0, 256, size=(*c["tile_shape_hw"], 3), dtype=np.uint8)
                y0 = row * c["cell_shape_hw"][0] + c["tile_offsets_yx"][target][0]
                x0 = col * c["cell_shape_hw"][1] + c["tile_offsets_yx"][target][1]
                for u in range(c["tile_shape_hw"][0]):
                    for v in range(c["tile_shape_hw"][1]):
                        rgb = [0, 0, 0]
                        if name == "constant":
                            rgb = [51, 128, 204]
                        elif name == "ramp":
                            rgb = [
                                (41 * k + 11 * r + 3 * cc + 5 * u + v) % 256,
                                (17 * k + 7 * r + 13 * cc + 3 * v) % 256,
                                (29 * k + 5 * r + 19 * cc + 7 * u) % 256,
                            ]
                        elif name == "checkerboard":
                            a = 15 + 32 * k + 128 * ((u + v + r + cc) % 2)
                            rgb = [a, 255 - a, a]
                        elif name == "impulse" and (u, v) == (k, (2 * k) % 6):
                            rgb = [255, 128, 64]
                        elif name == "noise":
                            rgb = noise[u, v]
                        elif name not in ("zero", "impulse"):
                            raise ValueError("unregistered oracle fixture")
                        out[y0 + u, x0 + v] = rgb
    if sentinel:
        s = f["sentinel"]
        for y in range(*s["raw_rows"]):
            for x in range(*s["raw_cols"]):
                out[y, x] = [255, 255, 255]
    return out
