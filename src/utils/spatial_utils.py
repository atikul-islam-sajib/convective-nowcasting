import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None


def apply_spatial(array, mode, height, width, interpolation):
    if mode == "none":
        return array

    if mode != "resize":
        raise ValueError(f"Unsupported spatial mode: {mode}")

    if cv2 is not None:
        cv_interpolation = (
            cv2.INTER_LINEAR if interpolation == "bilinear" else cv2.INTER_NEAREST
        )
        return cv2.resize(array, (width, height), interpolation=cv_interpolation).astype(array.dtype)

    if Image is None:
        raise ImportError("Neither OpenCV nor PIL is available")

    pil_interpolation = (
        Image.BILINEAR if interpolation == "bilinear" else Image.NEAREST
    )

    image = Image.fromarray(array)
    image = image.resize((width, height), pil_interpolation)
    return np.asarray(image).astype(array.dtype)
