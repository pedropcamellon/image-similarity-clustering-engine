from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from models import AppState, ImageData
from typing import Any, Callable
import streamlit as st


class StateAction(Enum):
    ADD_IMAGE = auto()
    REMOVE_IMAGE = auto()
    SET_CLUSTERS = auto()
    UPDATE_CONFIG = auto()
    RESET = auto()
    SET_SHOW_RESULTS = auto()


@dataclass
class StateUpdate:
    action: StateAction
    payload: Any


class SessionStateManager:
    """Manages application state operations and transitions."""

    def __init__(self, state_key: str = "app_state"):
        self._state_key = state_key
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize or migrate existing state."""
        if self._state_key not in st.session_state:
            st.session_state[self._state_key] = AppState()

    @property
    def state(self) -> AppState:
        """Get current application state."""
        return st.session_state[self._state_key]

    def dispatch(self, action: StateUpdate) -> None:
        """Process state updates through actions."""
        if action.action == StateAction.ADD_IMAGE:
            self._add_image(action.payload)
        elif action.action == StateAction.REMOVE_IMAGE:
            self._remove_image(action.payload)
        elif action.action == StateAction.SET_CLUSTERS:
            self._set_clusters(action.payload)
        elif action.action == StateAction.RESET:
            self._reset_state()

        elif action.action == StateAction.SET_SHOW_RESULTS:
            self._set_show_results(action.payload)

        self._update_timestamp()

    def _add_image(self, image: ImageData) -> None:
        """Add new image to state."""
        if not any(img.name == image.name for img in self.state.uploaded_images):
            self.state.uploaded_images.append(image)

    def _remove_image(self, image_name: str) -> None:
        """Remove image from state."""
        self.state.uploaded_images = [
            img for img in self.state.uploaded_images if img.name != image_name
        ]

    def _set_clusters(self, clusters: dict) -> None:
        """Update clustering results."""
        self.state.clusters = clusters
        self.state.show_results = True

    def _reset_state(self) -> None:
        """Reset to initial state."""
        st.session_state[self._state_key] = AppState()

    def _update_timestamp(self) -> None:
        """Update last modified timestamp."""
        self.state.last_updated = datetime.now()

    def _set_show_results(self, show: bool) -> None:
        """Set visibility of results."""
        self.state.show_results = show

    def subscribe(self, callback: Callable[[AppState], None]) -> None:
        """Subscribe to state changes."""
        # Implementation for state change notifications
        pass
