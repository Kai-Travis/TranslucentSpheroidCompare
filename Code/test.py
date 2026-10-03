import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter


sphe_path = r"C:\Github\TranslucentSpheroidCompare\TestImages\SpheroidPNG.png"
Bsphe_path = r"C:\Github\TranslucentSpheroidCompare\TestImages\BSpheroidPNG.png"


sphe = cv2.imread(sphe_path, cv2.IMREAD_UNCHANGED)
Bsphe = cv2.imread(Bsphe_path, cv2.IMREAD_UNCHANGED)


def prepare_image(img, target_size=250):

    if img.shape[2] == 4:

        bgr = img[:, :, :3]
        alpha = img[:, :, 3]

        mask = alpha > 0

    else:

        bgr = img

        gray_temp = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2GRAY
        )

        mask = gray_temp > 0

    gray = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        (target_size, target_size),
        interpolation=cv2.INTER_CUBIC
    )

    mask = cv2.resize(
        mask.astype(np.uint8),
        (target_size, target_size),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    return gray.astype(float), mask


def local_contrast(image, mask, window_size):

    # ------------------------------------------------
    # Only use pixels inside the spheroid
    # ------------------------------------------------

    valid = mask.astype(float)

    image_inside = image * valid

    # ------------------------------------------------
    # Number of valid pixels in each local window
    # ------------------------------------------------

    local_count = uniform_filter(
        valid,
        size=window_size
    )

    # ------------------------------------------------
    # Local mean
    # ------------------------------------------------

    local_sum = uniform_filter(
        image_inside,
        size=window_size
    )

    local_mean = np.divide(
        local_sum,
        local_count,
        out=np.zeros_like(local_sum),
        where=local_count > 0
    )

    # ------------------------------------------------
    # Local mean of squared intensity
    # ------------------------------------------------

    local_sum_squared = uniform_filter(
        image_inside ** 2,
        size=window_size
    )

    local_mean_squared = np.divide(
        local_sum_squared,
        local_count,
        out=np.zeros_like(local_sum_squared),
        where=local_count > 0
    )

    # ------------------------------------------------
    # Variance
    # ------------------------------------------------

    variance = (
        local_mean_squared
        - local_mean ** 2
    )

    variance = np.maximum(
        variance,
        0
    )

    # ------------------------------------------------
    # Standard deviation = local contrast
    # ------------------------------------------------

    contrast = np.sqrt(variance)

    contrast[~mask] = 0

    return contrast


# =====================================================
# Prepare images
# =====================================================

normal, normal_mask = prepare_image(sphe)
translucent, translucent_mask = prepare_image(Bsphe)


# =====================================================
# Try different window sizes
# =====================================================

window_sizes = [5, 15, 30, 50]


fig, axes = plt.subplots(
    2,
    len(window_sizes),
    figsize=(16, 8)
)


# Use one common scale
all_contrasts = []

for window in window_sizes:

    all_contrasts.append(
        local_contrast(
            normal,
            normal_mask,
            window
        )
    )

    all_contrasts.append(
        local_contrast(
            translucent,
            translucent_mask,
            window
        )
    )


vmax = max(
    c.max()
    for c in all_contrasts
)


# =====================================================
# Plot
# =====================================================

for i, window in enumerate(window_sizes):

    normal_contrast = local_contrast(
        normal,
        normal_mask,
        window
    )

    translucent_contrast = local_contrast(
        translucent,
        translucent_mask,
        window
    )


    # Normal
    im1 = axes[0, i].imshow(
        normal_contrast,
        cmap="jet",
        vmin=0,
        vmax=vmax
    )

    axes[0, i].set_title(
        f"Normal\n{window} × {window}"
    )

    axes[0, i].axis("off")


    # Translucent
    im2 = axes[1, i].imshow(
        translucent_contrast,
        cmap="jet",
        vmin=0,
        vmax=vmax
    )

    axes[1, i].set_title(
        f"Translucent\n{window} × {window}"
    )

    axes[1, i].axis("off")


# =====================================================
# Labels
# =====================================================

axes[0, 0].set_ylabel(
    "Normal",
    fontsize=14
)

axes[1, 0].set_ylabel(
    "Translucent",
    fontsize=14
)


# Shared colour bar
fig.colorbar(
    im2,
    ax=axes,
    shrink=0.8,
    label="Local contrast (SD)"
)


plt.tight_layout()
plt.show()