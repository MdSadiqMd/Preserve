"""Coordinated import of the two libraries that vendor their own FFmpeg

opencv-python-headless and PyAV each bundle a private copy of libavdevice. On
macOS both register the Objective-C classes AVFFrameReceiver and
AVFAudioReceiver, and whichever loads second makes the ObjC runtime print a
duplicate-implementation warning straight to file descriptor 2

The warning is inert here: those classes belong to FFmpeg's AVFoundation
capture device support, which this project never uses — video is read and
written from files. It cannot be silenced with the warnings module because the
runtime writes it during dlopen, below Python

So both libraries are imported once, here, with fd 2 captured. Lines matching the
known duplicate-class warning are dropped; everything else is passed through, so
genuine loader errors are still visible
"""

import os
import re
import sys
import tempfile

_DUPLICATE_CLASS_WARNING = re.compile(
    r"^objc\[\d+\]: (Class (AVF|GLESv|AVAudio)\w+ is implemented in both"
    r"|One of the duplicates must be removed or renamed)"
)


def _is_benign(line: str) -> bool:
    if _DUPLICATE_CLASS_WARNING.match(line):
        return True
    # The warning spans two lines; the continuation names the consequence.
    return "This may cause spurious casting failures and mysterious crashes" in line


def import_media_libraries() -> None:
    """Import cv2 and av, filtering the benign duplicate-class warning."""
    if "cv2" in sys.modules and "av" in sys.modules:
        return

    if sys.platform != "darwin":
        import av  # noqa: F401
        import cv2  # noqa: F401

        return

    sys.stderr.flush()
    saved_fd = os.dup(2)

    with tempfile.TemporaryFile(mode="w+") as buffer:
        try:
            os.dup2(buffer.fileno(), 2)
            try:
                import av  # noqa: F401
                import cv2  # noqa: F401
            finally:
                sys.stderr.flush()
                os.dup2(saved_fd, 2)
        finally:
            os.close(saved_fd)

        buffer.seek(0)
        passthrough = [line for line in buffer.read().splitlines() if not _is_benign(line)]

    for line in passthrough:
        print(line, file=sys.stderr)
