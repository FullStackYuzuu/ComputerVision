import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
from segmentation import (
    load_image,
    rgb_to_gray,
    apply_gabor_filters,
    create_feature_matrix,
    normalize_features,
    perform_kmeans,
    create_colored_segmentation,
    calculate_region_statistics,
    create_boundary_overlay
)
st.set_page_config(
    page_title="Flower Texture Segmentation",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Flower Texture Segmentation")
st.markdown(
    """
Segmentasi tekstur bunga menggunakan **Gabor Filter** dan
**K-Means Clustering**.
"""
)
st.sidebar.header("Segmentation Parameters")

k = st.sidebar.slider(
    "Number of Clusters",
    min_value=2,
    max_value=6,
    value=3
)

st.sidebar.markdown("---")
st.sidebar.write("Gabor Parameters")
st.sidebar.write("Wavelength : 4, 8, 16")
st.sidebar.write("Orientation : 0°, 45°, 90°, 135°")

uploaded = st.file_uploader(
    "Upload Flower Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:

    with st.spinner("Processing Image..."):

        img = load_image(uploaded)

        gray = rgb_to_gray(img)

        gabor_images = apply_gabor_filters(gray)

        features = create_feature_matrix(gabor_images)

        features = normalize_features(features)

        segmented = perform_kmeans(
            features,
            gray.shape,
            k=k
        )

        colored_seg = create_colored_segmentation(segmented)
        overlay = create_boundary_overlay(img, segmented)
    st.success("Segmentation Finished!")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            img,
            use_container_width=True
        )

    with col2:

        st.subheader("Grayscale")

        st.image(
            gray,
            clamp=True,
            use_container_width=True
        )


    st.divider()

    st.subheader("Gabor Filter Responses")

    cols = st.columns(4)

    for i, gabor_img in enumerate(gabor_images):

        display = (gabor_img * 255).astype(np.uint8)

        with cols[i % 4]:

            st.image(
                display,
                caption=f"Gabor {i+1}",
                use_container_width=True
            )

    st.divider()

    st.subheader("Segmentation Result")

    col3, col4 = st.columns(2)

    with col3:

        display_segment = (
            segmented * (255 // (k - 1))
        ).astype(np.uint8)

        st.image(
            display_segment,
            caption="Cluster Labels",
            use_container_width=True
        )

    with col4:

        st.image(
            colored_seg,
            caption="Colored Segmentation",
            use_container_width=True
        )
    st.divider()

    st.subheader("🔍 Boundary Overlay")

    st.image(
        overlay,
        caption="Segmentation Boundary Overlay",
        use_container_width=True
    )
    # ===========================================
    # REGION ANALYSIS
    # ===========================================

    region_pixels, region_percent = calculate_region_statistics(segmented)

    st.divider()

    st.subheader("📊 Region Statistics")

    data = pd.DataFrame({
        "Region": [f"Region {i+1}" for i in range(len(region_pixels))],
        "Pixels": region_pixels,
        "Percentage (%)": [round(x,2) for x in region_percent]
    })

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )

    # ===========================================
    # BAR CHART
    # ===========================================

    st.subheader("📈 Region Area Percentage")

    chart_data = data.set_index("Region")

    st.bar_chart(
        chart_data["Percentage (%)"]
    )
    # ===========================================
    # PIE CHART
    # ===========================================

    st.divider()

    st.subheader("🥧 Region Area Composition")

    fig, ax = plt.subplots(figsize=(6,6))

    ax.pie(
        region_percent,
        labels=[f"Region {i+1}" for i in range(len(region_percent))],
        autopct="%1.2f%%",
        startangle=90
    )

    ax.axis("equal")

    st.pyplot(fig)

    # ===========================================
    # DOWNLOAD RESULT
    # ===========================================

    st.divider()

    st.subheader("💾 Download Result")

    # ubah menjadi gambar PIL
    result_image = Image.fromarray(
        (colored_seg * 255).astype(np.uint8)
    )

    buffer = io.BytesIO()

    result_image.save(buffer, format="PNG")

    buffer.seek(0)

    filename = os.path.splitext(uploaded.name)[0]

    st.download_button(
        label="📥 Download Colored Segmentation",
        data=buffer,
        file_name=f"hasil_segmentasi_{filename}.png",
        mime="image/png"
    )
    with st.expander("Debug Information"):

        st.write("Image Size :", img.shape)

        st.write("Feature Matrix :", features.shape)

        st.write("Clusters :", np.unique(segmented))