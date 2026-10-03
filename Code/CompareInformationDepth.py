import cv2
import numpy as np
import matplotlib.pyplot as plt


sphe_path = r"C:\Github\TranslucentSpheroidCompare\TestImages\SpheroidPNG.png"
Bsphe_path = r"C:\Github\TranslucentSpheroidCompare\TestImages\BSpheroidPNG.png"

sphe = cv2.imread(sphe_path, cv2.IMREAD_UNCHANGED)
Bsphe = cv2.imread(Bsphe_path, cv2.IMREAD_UNCHANGED)

def information_depth(img, n_zones=10):

    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        mask = alpha > 0

    else:
        bgr = img
        gray_temp = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = gray_temp > 0

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    ys, xs = np.where(mask)

    cx = xs.mean()
    cy = ys.mean()

    y, x = np.indices(gray.shape)

    distance = np.sqrt((x - cx)**2 + (y - cy)**2)

    max_radius = distance[mask].max()
    normalized_radius = distance / max_radius

    edges = np.linspace(0, 1, n_zones + 1)

    mean_intensity = []
    intensity_sd = []
    coefficient_variation = []
    pixel_count = []

    for i in range(n_zones):
        inner = edges[i]
        outer = edges[i + 1]

        zone = (
            mask &
            (normalized_radius >= inner) &
            (normalized_radius < outer)
        )

        values = gray[zone].astype(float)

        if len(values) == 0:
            continue

        mean = np.mean(values)
        sd = np.std(values)

        mean_intensity.append(mean)
        intensity_sd.append(sd)

        if mean != 0:
            coefficient_variation.append(sd / mean)
        else:
            coefficient_variation.append(np.nan)

        pixel_count.append(len(values))

    depth = (edges[:-1] + edges[1:]) / 2


    return (
        depth,
        np.array(mean_intensity),
        np.array(intensity_sd),
        np.array(coefficient_variation),
        np.array(pixel_count)
    )

sphe_depth, sphe_mean, sphe_sd, sphe_cv, sphe_pixels = \
    information_depth(sphe)
Bsphe_depth, Bsphe_mean, Bsphe_sd, Bsphe_cv, Bsphe_pixels = \
    information_depth(Bsphe)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(
    sphe_depth,
    sphe_mean,
    marker="o",
    linewidth=2,
    label="Normal"
)

axes[0].plot(
    Bsphe_depth,
    Bsphe_mean,
    marker="o",
    linewidth=2,
    label="Translucent"
)

axes[0].set_xlabel(
    "Distance from center\n(normalized)"
)

axes[0].set_ylabel(
    "Mean grayscale intensity"
)

axes[0].set_title(
    "Mean intensity"
)

axes[0].legend()

axes[0].grid(
    alpha=0.25
)

axes[1].plot(
    sphe_depth,
    sphe_sd,
    marker="o",
    linewidth=2,
    label="Normal"
)

axes[1].plot(
    Bsphe_depth,
    Bsphe_sd,
    marker="o",
    linewidth=2,
    label="Translucent"
)

axes[1].set_xlabel(
    "Distance from center\n(normalized)"
)

axes[1].set_ylabel(
    "Intensity standard deviation"
)

axes[1].set_title(
    "Internal intensity variation"
)

axes[1].legend()

axes[1].grid(
    alpha=0.25
)

axes[2].plot(
    sphe_depth,
    sphe_cv,
    marker="o",
    linewidth=2,
    label="Normal"
)

axes[2].plot(
    Bsphe_depth,
    Bsphe_cv,
    marker="o",
    linewidth=2,
    label="Translucent"
)

axes[2].set_xlabel(
    "Distance from center\n(normalized)"
)

axes[2].set_ylabel(
    "Coefficient of variation"
)

axes[2].set_title(
    "Relative internal variation"
)

axes[2].legend()

axes[2].grid(
    alpha=0.25
)

plt.tight_layout()

plt.show()