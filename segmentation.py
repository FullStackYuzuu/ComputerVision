import cv2
import numpy as np
from PIL import Image

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from skimage.color import label2rgb
from skimage.segmentation import find_boundaries

# =====================================================
# IMAGE LOADING
# =====================================================

def load_image(uploaded_file):
    """
    Membaca gambar dari Streamlit uploader.
    """

    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    return img


# =====================================================
# RGB -> GRAYSCALE
# =====================================================

def rgb_to_gray(img):
    """
    Mengubah citra RGB menjadi grayscale float (0-1).
    """

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = gray.astype(np.float32) / 255.0

    return gray


# =====================================================
# GABOR FILTER BANK
# =====================================================

def apply_gabor_filters(gray):
    """
    Menerapkan Gabor Filter Bank.

    wavelength :
        4, 8, 16

    orientation :
        0, 45, 90, 135
    """

    wavelengths = [4, 8, 16]
    orientations = [0, 45, 90, 135]

    gabor_images = []

    for wavelength in wavelengths:

        for angle in orientations:

            theta = np.deg2rad(angle)

            kernel = cv2.getGaborKernel(
                ksize=(31, 31),
                sigma=4.0,
                theta=theta,
                lambd=wavelength,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F
            )

            filtered = cv2.filter2D(
                gray,
                cv2.CV_32F,
                kernel
            )

            filtered = cv2.normalize(
                filtered,
                None,
                0,
                1,
                cv2.NORM_MINMAX
            )

            gabor_images.append(filtered)

    return gabor_images


# =====================================================
# FEATURE EXTRACTION
# =====================================================

def create_feature_matrix(gabor_images):
    """
    Mengubah seluruh hasil Gabor menjadi feature matrix.

    (jumlah pixel x jumlah filter)
    """

    features = []

    for img in gabor_images:

        features.append(img.reshape(-1))

    features = np.array(features).T

    return features


# =====================================================
# FEATURE NORMALIZATION
# =====================================================

def normalize_features(features):
    """
    Standardisasi feature.
    """

    scaler = StandardScaler()

    normalized = scaler.fit_transform(features)

    return normalized


# =====================================================
# KMEANS
# =====================================================

def perform_kmeans(features, image_shape, k=3):
    """
    Segmentasi menggunakan K-Means Clustering.
    """

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        max_iter=300,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(features)

    segmented = labels.reshape(image_shape)

    # JANGAN ubah ke uint8
    # label2rgb membutuhkan integer biasa

    return segmented


# =====================================================
# COLORED SEGMENTATION
# =====================================================

def create_colored_segmentation(segmented):
    """
    Memberikan warna berbeda pada setiap cluster.
    """

    colored = label2rgb(segmented)

    return colored


# =====================================================
# REGION ANALYSIS
# =====================================================

def calculate_region_statistics(segmented):
    """
    Menghitung jumlah pixel dan persentase
    tiap region.
    """

    k = len(np.unique(segmented))

    total_pixels = segmented.size

    region_pixels = []

    region_percent = []

    for i in range(k):

        pixels = np.sum(segmented == i)

        percent = (pixels / total_pixels) * 100

        region_pixels.append(int(pixels))

        region_percent.append(float(percent))

    return region_pixels, region_percent

def create_boundary_overlay(original, segmented):
    """
    Membuat overlay batas segmentasi pada gambar asli.
    """

    overlay = original.copy()

    boundaries = find_boundaries(segmented, mode="outer")

    # Warna merah untuk batas region
    overlay[boundaries] = [255, 0, 0]

    return overlay