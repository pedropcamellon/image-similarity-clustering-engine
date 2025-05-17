from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from models import ImageData
from numpy.typing import NDArray
from pydantic import ConfigDict, validate_call
from scipy.cluster.hierarchy import linkage, fcluster
from state_manager import SessionStateManager, StateAction, StateUpdate
from streamlit.runtime.uploaded_file_manager import UploadedFile
import cv2
import mediapipe as mp
import numpy as np
import os
import streamlit as st


def setup_page() -> None:
    """Configure the Streamlit page settings for the application.

    Sets the page title, icon, layout, and initial sidebar state.
    """
    st.set_page_config(
        page_title="Image Similarity Clustering",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def setup_header() -> None:
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


def init_session_state() -> None:
    """Initialize the application state"""
    global state_manager
    state_manager = SessionStateManager()


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def load_image(uploaded_file: UploadedFile) -> NDArray:
    """Convert an uploaded file into an RGB image array.

    Args:
        uploaded_file: A Streamlit UploadedFile object containing image data

    Returns:
        NDArray: RGB image array

    Raises:
        ValueError: If file is empty or cannot be decoded
    """
    bytes_data = uploaded_file.getvalue()
    if not bytes_data:
        raise ValueError("Empty file provided")

    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@st.cache_data
def get_embeddings(
    images: dict[str, NDArray], model_path: str = "models/mobilenet_v3_small.tflite"
) -> list[dict]:
    """Generate embeddings for a set of images using MediaPipe.

    Args:
        images: Dictionary of image name to numpy array pairs
        model_path: Path to the TFLite model file

    Returns:
        list: A list of dictionaries containing file names and their embeddings
        List of dictionaries containing file names and their embeddings

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If images dict is empty
    """
    if not images:
        raise ValueError("No images provided")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageEmbedderOptions(
        base_options=base_options, l2_normalize=True, quantize=True
    )

    embedding_results = []
    with vision.ImageEmbedder.create_from_options(options) as embedder:
        for img_name, img in images.items():
            if img is None or img.size == 0:
                continue
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            embedding = embedder.embed(mp_image)
            embedding_results.append({"file_name": img_name, "embeddings": embedding})

    return embedding_results


def cluster_images(
    embedding_results: list[dict],
    threshold: int = 5500,
) -> dict[int, list[str]]:
    """Group images into clusters based on their embedding similarities.

    Args:
        embedding_results (list): List of image embeddings from get_embeddings()
        threshold (int): Distance threshold for cluster formation (default: 5500)

    Returns:
        dict: Mapping of cluster IDs to lists of file names
        Mapping of cluster IDs to lists of file names

    Raises:
        ValueError: If embedding_results is empty or threshold is invalid
    """
    if not embedding_results:
        raise ValueError("No embedding results provided")

    if threshold <= 0:
        raise ValueError("Threshold must be positive")

    embedding_vectors = np.array(
        [result["embeddings"].embeddings[0].embedding for result in embedding_results]
    )

    linkage_matrix = linkage(embedding_vectors, method="ward")
    labels = fcluster(linkage_matrix, t=threshold, criterion="distance")

    clusters: dict[int, list[str]] = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(embedding_results[idx]["file_name"])

    return clusters


def display_upload_section() -> None:
    """Render the image upload section of the interface.

    Handles multiple image uploads and displays previews in a grid layout.
    Updates session state with uploaded images.
    Limits:
        - Maximum 20 images
        - Maximum 10MB per image
    """
    st.subheader("1. Upload Images")

    config = state_manager.state.clustering_config
    current_count = len(state_manager.state.uploaded_images)

    if current_count > 0:
        st.caption(f"Uploaded: {current_count}/{config.max_images} images")

    uploaded_files = st.file_uploader(
        f"Choose two or more images to find similar groups (max {config.max_images} images, {config.max_file_size_mb}MB each)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help=f"Supported formats: JPG, JPEG, PNG. Limits: {config.max_images} images maximum, {config.max_file_size_mb}MB per image.",
        key="file_uploader",
    )

    if uploaded_files:
        # Track processed files to avoid duplicates
        processed_names = {img.name for img in state_manager.state.uploaded_images}

        for file in uploaded_files:
            # Check file size
            if file.size > config.max_file_size_mb * 1024 * 1024:
                st.error(
                    f"⚠️ {file.name} exceeds {config.max_file_size_mb}MB limit. Skipping..."
                )
                continue

            # Check total image count
            if len(state_manager.state.uploaded_images) >= config.max_images:
                st.warning(
                    f"⚠️ Maximum image limit ({config.max_images}) reached. Additional images will be ignored."
                )
                break

            # Only process new files
            if file.name not in processed_names:
                state_manager.dispatch(
                    StateUpdate(
                        action=StateAction.ADD_IMAGE,
                        payload=ImageData(name=file.name, data=load_image(file)),
                    )
                )

    # Display grid of images
    if state_manager.state.uploaded_images:
        cols = st.columns(4)
        for idx, img in enumerate(state_manager.state.uploaded_images):
            with cols[idx % 4]:
                st.image(img.data, caption=img.name, use_container_width=True)


def perform_clustering(images: list[ImageData], threshold: int) -> None:
    """Perform clustering on the provided images and update app state."""
    images_dict = {img.name: img.data for img in images}
    embedding_results = get_embeddings(images_dict)
    clusters = cluster_images(embedding_results, threshold=threshold)
    state_manager.dispatch(
        StateUpdate(action=StateAction.SET_CLUSTERS, payload=clusters)
    )
    state_manager.dispatch(
        StateUpdate(action=StateAction.SET_SHOW_RESULTS, payload=True)
    )

    st.success(f"✅ Found {len(clusters)} groups of similar images!")
    st.balloons()


def display_clustering_section() -> None:
    """Render the clustering controls section.

    Shows clustering button and handles the clustering process.
    Requires at least 2 images to be uploaded before enabling clustering.
    """
    st.subheader("2. Find Similar Groups")

    if len(state_manager.state.uploaded_images) < 2:
        st.info("⚠️ Please upload at least 2 images to start clustering")
        st.button("Start Clustering", disabled=True)
        return

    config = state_manager.state.clustering_config
    threshold = st.slider(
        "Similarity Threshold",
        min_value=1000,
        max_value=10000,
        value=config.threshold,
        step=500,
        help="Lower values create more groups, higher values create fewer groups. Adjust based on your needs.",
    )
    config.threshold = threshold

    if st.button("Start Clustering", type="primary"):
        with st.spinner("🔍 Finding similar image groups..."):
            perform_clustering(state_manager.state.uploaded_images, threshold)
        st.rerun()


def get_uploaded_image_by_name(name: str) -> NDArray:
    for img in state_manager.state.uploaded_images:
        if img.name == name:
            return img.data
    raise ValueError(f"Image with name {name} not found in uploaded images.")


def display_results_section() -> None:
    """Display clustering results in expandable sections.

    Shows grouped images in a grid layout within expandable containers.
    Only visible after clustering has been performed.
    """
    if not state_manager.state.show_results:
        return

    if state_manager.state.clusters is None:
        st.warning("No clusters found. Please try again.")
        return

    st.subheader("3. Results")
    st.write(f"{len(state_manager.state.clusters)} Groups Found")

    for cluster_id, image_files in state_manager.state.clusters.items():
        with st.expander(
            f"Group {cluster_id} ({len(image_files)} images)", expanded=True
        ):
            cols = st.columns(4)
            for idx, img_name in enumerate(image_files):
                with cols[idx % 4]:
                    st.image(
                        get_uploaded_image_by_name(img_name),
                        caption=img_name,
                        use_container_width=True,
                    )


def display_reset_section() -> None:
    """Render the reset controls section."""
    st.divider()
    if st.button("Delete All & Start Over", type="secondary"):
        state_manager.dispatch(StateUpdate(action=StateAction.RESET, payload=None))
        st.rerun()


def main() -> None:
    """Main application entry point.

    Initializes the application, sets up the UI components,
    and manages the application state and workflow.
    """
    # Configure page and header
    setup_page()
    setup_header()
    init_session_state()

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
