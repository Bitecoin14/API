from . import normal, inverted, hallucinogenic, ascii_art
from . import middle_finger_blur, upside_down, mosaic
from . import black_and_white, flat_2d

# Ordered list of filters cycled by pinch gesture.
# middle_finger_blur is intentionally excluded — it runs on every frame
# as an overlay on top of whichever filter is active.
FILTERS = [
    normal.FILTER,
    inverted.FILTER,
    hallucinogenic.FILTER,
    ascii_art.FILTER,
    upside_down.FILTER,
    mosaic.FILTER,
    black_and_white.FILTER,
    flat_2d.FILTER,
]

# Always-on overlay applied after the active filter.
ALWAYS_ON_OVERLAY = middle_finger_blur._apply
