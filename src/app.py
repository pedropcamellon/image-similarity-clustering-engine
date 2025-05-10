import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import tempfile
import os
from PIL import Image


def setup_page():
    """Configure the Streamlit page settings for the application.

    Sets the page title, icon, layout, and initial sidebar state.
    """
    st.set_page_config(
        page_title="Image Similarity Clustering",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def setup_header():
    """Create the application header with title and description.

    Displays the main title, descriptive text, and a divider.
    """
    st.title("🖼️ Image Similarity Clustering")
    st.markdown(
        "<p style='font-size: 1.2em; color: #666;'>"
        "Upload images and discover visually similar groups in one click."
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()


def load_image(uploaded_file):
    """Convert an uploaded file into an RGB image array.

    Args:
        uploaded_file: A Streamlit UploadedFile object containing image data

    Returns:
        numpy.ndarray: RGB image array
    """
    bytes_data = uploaded_file.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB immediately


@st.cache_data
def get_embeddings(images, model_path="models/mobilenet_v3_small.tflite"):
    """Generate embeddings for a set of images using MediaPipe.

    Args:
        images (dict): Dictionary of image name to numpy array pairs
        model_path (str): Path to the TFLite model file

    Returns:
        list: List of dictionaries containing file names and their embeddings
    """
    # Initialize MediaPipe Image Embedder with normalized and quantized outputs
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageEmbedderOptions(
        base_options=base_options, l2_normalize=True, quantize=True
    )

    with vision.ImageEmbedder.create_from_options(options) as embedder:
        embedding_results = []
        for img_name, img in images.items():
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            embedding = embedder.embed(mp_image)
            embedding_results.append({"file_name": img_name, "embeddings": embedding})
    return embedding_results


def cluster_images(embedding_results, threshold=5500):
    """Group images into clusters based on their embedding similarities.

    Args:
        embedding_results (list): List of image embeddings from get_embeddings()
        threshold (int): Distance threshold for cluster formation (default: 5500)

    Returns:
        dict: Mapping of cluster IDs to lists of file names
    """
    # Convert embeddings to numpy array for clustering
    embedding_vectors = np.array(
        [result["embeddings"].embeddings[0].embedding for result in embedding_results]
    )

    linkage_matrix = linkage(embedding_vectors, method="ward")
    labels = fcluster(linkage_matrix, t=threshold, criterion="distance")

    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(embedding_results[idx]["file_name"])

    return clusters


def display_upload_section():
    """Render the image upload section of the interface.

    Handles multiple image uploads and displays previews in a grid layout.
    Updates session state with uploaded images.
    Limits:
        - Maximum 20 images
        - Maximum 10MB per image
    """
    st.subheader("1. Upload Images")

    # Show current count
    current_count = len(st.session_state.get("uploaded_images", {}))
    if current_count > 0:
        st.caption(f"Uploaded: {current_count}/20 images")

    uploaded_files = st.file_uploader(
        "Choose two or more images to find similar groups (max 20 images, 10MB each)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Supported formats: JPG, JPEG, PNG. Limits: 20 images maximum, 10MB per image.",
        key="file_uploader",
    )

    if uploaded_files:
        for file in uploaded_files:
            # Check file size (10MB = 10 * 1024 * 1024 bytes)
            if file.size > 10 * 1024 * 1024:
                st.error(f"⚠️ {file.name} exceeds 10MB limit. Skipping...")
                continue

            # Check total image count
            if len(st.session_state.uploaded_images) >= 20:
                st.warning(
                    "⚠️ Maximum image limit (20) reached. Additional images will be ignored."
                )
                break

            if file.name not in st.session_state.uploaded_images:
                st.session_state.uploaded_images[file.name] = load_image(file)

    # Display grid of images
    if st.session_state.uploaded_images:
        cols = st.columns(4)
        for idx, (name, img) in enumerate(st.session_state.uploaded_images.items()):
            with cols[idx % 4]:
                st.image(img, caption=name, use_container_width=True)


def display_clustering_section():
    """Render the clustering controls section.

    Shows clustering button and handles the clustering process.
    Requires at least 2 images to be uploaded before enabling clustering.
    """
    st.subheader("2. Find Similar Groups")

    if len(st.session_state.uploaded_images) < 2:
        st.info("⚠️ Please upload at least 2 images to start clustering")
        st.button("Start Clustering", disabled=True)
        return

    # Add threshold control
    threshold = st.slider(
        "Similarity Threshold",
        min_value=1000,
        max_value=10000,
        value=5500,
        step=500,
        help="Lower values create more groups, higher values create fewer groups. Adjust based on your needs.",
    )

    if st.button("Start Clustering", type="primary"):
        with st.spinner("🔍 Finding similar image groups..."):
            embedding_results = get_embeddings(st.session_state.uploaded_images)
            clusters = cluster_images(embedding_results, threshold=threshold)
            st.session_state.clusters = clusters
            st.session_state.show_results = True

            # Display clustering results
            st.success(f"✅ Found {len(clusters)} groups of similar images!")
            st.balloons()
            st.session_state.show_results = True
            st.rerun()


def display_results_section():
    """Display clustering results in expandable sections.

    Shows grouped images in a grid layout within expandable containers.
    Only visible after clustering has been performed.
    """
    if not st.session_state.show_results:
        return

    st.subheader(f"3. Results: {len(st.session_state.clusters)} Groups Found")

    for cluster_id, image_files in st.session_state.clusters.items():
        with st.expander(
            f"Group {cluster_id} ({len(image_files)} images)", expanded=True
        ):
            cols = st.columns(4)
            for idx, img_name in enumerate(image_files):
                with cols[idx % 4]:
                    st.image(
                        st.session_state.uploaded_images[img_name],
                        caption=img_name,
                        use_container_width=True,
                    )


def display_reset_section():
    """Render the reset controls section.

    Provides functionality to clear all uploaded images and results,
    allowing the user to start over.
    """
    st.divider()
    if st.button("Delete All & Start Over", type="secondary"):
        for key in ["uploaded_images", "clusters", "show_results"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


def main():
    """Main application entry point.

    Initializes the application, sets up the UI components,
    and manages the application state and workflow.
    """
    # Configure page and header
    setup_page()
    setup_header()

    # Initialize session state for persistence
    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = {}
    if "show_results" not in st.session_state:
        st.session_state.show_results = False

    # Render main application sections
    display_upload_section()
    display_clustering_section()
    display_results_section()
    display_reset_section()

    # Display footer information
    st.markdown("---")
    st.caption("Made with ❤️ using MediaPipe and Streamlit")


# Entry point guard
if __name__ == "__main__":
    main()
