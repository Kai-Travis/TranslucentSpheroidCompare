import cv2
import numpy as np
import matplotlib.pyplot as plt

sphe_path = r"C:\Github\TranslucentSpheroidCompare\TestImages\SpheroidPNG.png"
Bsphe_path = r"C:\Github\TranslucentSpheroidCompare\TestImages\BSpheroidPNG.png"

sphe = cv2.imread(sphe_path, cv2.IMREAD_UNCHANGED)
Bsphe = cv2.imread(Bsphe_path, cv2.IMREAD_UNCHANGED)

def CalcRadialIntensity(img):
    if img.shape[2] == 4:
        bgr = img[:,:,:3]
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

    distance = np.sqrt((x - cx)**2 * (y - cy)**2)

    distances = distance[mask]
    intensities = gray[mask]

    max_radius = distances.max()
    normalized_radius = distances / max_radius

    bins = np.linspace(0, 1, 101)

    radial_intensity = []
    radial_radius = []

    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i+1]

        in_bin = ((normalized_radius >= lower) & (normalized_radius < upper))

        if np.any(in_bin):
            mean_intensity = intensities[in_bin].mean()

            radial_intensity.append(mean_intensity)
            radial_radius.append((lower + upper) / 2)

    return np.array(radial_radius), np.array(radial_intensity), mask, cx, cy

sphe_radius, sphe_intensity, sphe_mask, sphe_cx, sphe_cy = \
    CalcRadialIntensity(sphe)

Bsphe_radius, Bsphe_intensity, Bsphe_mask, Bsphe_cx, Bsphe_cy = \
    CalcRadialIntensity(Bsphe)

plt.figure(figsize=(8,5))

plt.plot(sphe_radius,
         sphe_intensity,
         label = "Conventional Spheroids",
         linewidth = 2)

plt.plot(Bsphe_radius,
         Bsphe_intensity,
         label = "Translucent Spheroids",
         linewidth = 2)

plt.xlabel("Normalized radius")
plt.ylabel("Mean grayscale intensity")

plt.title("Radial Intensity Profile")

plt.xlim(0, 1)

plt.legend()

plt.grid(alpha = 0.25)

plt.tight_layout()

plt.show()