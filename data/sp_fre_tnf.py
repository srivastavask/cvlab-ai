import cv2
import numpy as np

def apply_spatial_transform(frame):
    # Rotate the image 45 degrees around its center
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    angle = 45
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(frame, M, (w, h))
    return rotated

def apply_frequency_transform(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply FFT
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Compute magnitude spectrum
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude += 1e-5  # avoid log(0)
    magnitude_spectrum = 20 * np.log(magnitude)

    # Normalize for display
    cv2.normalize(magnitude_spectrum, magnitude_spectrum, 0, 255, cv2.NORM_MINMAX)
    spectrum_uint8 = np.uint8(magnitude_spectrum)

    return spectrum_uint8

# Capture video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    spatial = apply_spatial_transform(frame)
    frequency = apply_frequency_transform(frame)

    # Show all windows
    cv2.imshow("Original", frame)
    cv2.imshow("Spatial Transform (Rotated)", spatial)
    cv2.imshow("Frequency Transform (FFT Magnitude)", frequency)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
