import os
import streamlit.components.v1 as components

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

_component_func = components.declare_component(
    "back_camera_capture",
    path=_FRONTEND_DIR,
)


def back_camera_capture(key=None):
    """
    Renders a back-camera (rear camera) capture widget with an explicit
    'Capture photo' button and a 'Retake' button.

    Returns:
        A base64 data URL string (e.g. "data:image/jpeg;base64,...") of the
        captured photo, or None if nothing has been captured yet.
    """
    return _component_func(key=key, default=None)
