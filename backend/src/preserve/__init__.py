"""Preserve: Localized AI video editing with pixel-level preservation guarantees."""

from preserve.medialibs import import_media_libraries

# Must run before anything else pulls in cv2 or av, so the two vendored FFmpeg
# copies load together and their duplicate-class warning can be filtered.
import_media_libraries()

__version__ = "0.1.0"
