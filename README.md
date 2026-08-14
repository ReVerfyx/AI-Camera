# AI Camera — Android

Real-time Android camera starter using CameraX + MediaPipe Tasks Vision.

Features:
- Corrected portrait rotation and front-camera mirroring.
- Overlay uses the same `fillCenter` crop as PreviewView, so face/hand/pose landmarks line up with the camera image.
- Face mesh + face box.
- Two hands with 21 landmarks each.
- Body pose skeleton.
- Simple visual-expression heuristic (`SMILE :)`, `SURPRISED?`, `NEUTRAL`). This is not a diagnosis or a reliable measurement of emotion.
- FPS overlay.
- Red danger-box UI is ready for a dedicated weapon/object model.

Age and weight are intentionally shown as `—`: they should not be fabricated from appearance. A dedicated, validated model would be required for any approximate estimation.

## GitHub Actions

Upload the repository and run **Actions → Build Android APK**. The workflow downloads the MediaPipe model bundles and builds `app-debug.apk`.

A real weapon detector is not included merely by labeling arbitrary objects as weapons. To enable it, add a compatible TFLite object-detection model trained for the weapon classes you want to detect, then feed its boxes into `OverlayView.detections`; dangerous boxes are already drawn red.
