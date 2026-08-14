package com.example.aicamera

import android.content.Context
import android.graphics.Matrix
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

private val DANGER_LABELS = setOf(
    "knife", "scissors", "baseball bat", "gun", "pistol", "rifle", "firearm", "weapon"
)

/** Optimized for up to 2 people. Preview is never blocked by inference. */
class VisionAnalyzer(
    private val context: Context,
    private val overlay: OverlayView
) : ImageAnalysis.Analyzer {

    private val face: FaceLandmarker
    private val hands: HandLandmarker
    private val pose: PoseLandmarker
    private val objects: ObjectDetector
    private val executor = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)

    private var lastFpsTime = SystemClock.elapsedRealtime()
    private var analyzedFrames = 0
    private var inferenceFps = 0f

    init {
        face = FaceLandmarker.createFromOptions(
            context,
            FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("face_landmarker.task").build())
                .setRunningMode(RunningMode.VIDEO)
                .setNumFaces(2)
                .build()
        )

        hands = HandLandmarker.createFromOptions(
            context,
            HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build())
                .setRunningMode(RunningMode.VIDEO)
                .setNumHands(4)
                .build()
        )

        pose = PoseLandmarker.createFromOptions(
            context,
            PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").build())
                .setRunningMode(RunningMode.VIDEO)
                .setNumPoses(2)
                .build()
        )

        objects = ObjectDetector.createFromOptions(
            context,
            ObjectDetector.ObjectDetectorOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("efficientdet_lite2.tflite").build())
                .setRunningMode(RunningMode.VIDEO)
                .setMaxResults(20)
                .setScoreThreshold(0.4f)
                .build()
        )
    }

    override fun analyze(image: ImageProxy) {
        // Drop frames while inference is busy. This keeps PreviewView smooth.
        if (!busy.compareAndSet(false, true)) {
            image.close()
            return
        }

        try {
            val raw = image.toBitmap()
            val rotation = image.imageInfo.rotationDegrees
            image.close()

            executor.execute {
                try {
                    analyzeFrame(raw, rotation)
                } catch (t: Throwable) {
                    t.printStackTrace()
                } finally {
                    busy.set(false)
                }
            }
        } catch (t: Throwable) {
            image.close()
            busy.set(false)
            t.printStackTrace()
        }
    }

    private fun analyzeFrame(raw: android.graphics.Bitmap, rotation: Int) {
        val rotated = rotateBitmap(raw, rotation)
        val mpImage = BitmapImageBuilder(rotated).build()
        val timestamp = SystemClock.elapsedRealtime()

        val faceResult = face.detectForVideo(mpImage, timestamp)
        val handResult = hands.detectForVideo(mpImage, timestamp)
        val poseResult = pose.detectForVideo(mpImage, timestamp)
        val objectResult = objects.detectForVideo(mpImage, timestamp)

        val faces = faceResult.faceLandmarks().take(2).map { list ->
            list.map { P(it.x(), it.y()) }
        }
        val handPoints = handResult.landmarks().take(4).map { list ->
            list.map { P(it.x(), it.y()) }
        }
        val poses = poseResult.landmarks().take(2).map { person ->
            person.map { P(it.x(), it.y()) }
        }

        val palmCenters = handPoints.mapNotNull { h ->
            if (h.size < 21) null
            else {
                val ids = intArrayOf(0, 5, 9, 13, 17)
                P(
                    ids.map { h[it].x }.average().toFloat(),
                    ids.map { h[it].y }.average().toFloat()
                )
            }
        }

        val w = rotated.width.toFloat()
        val h = rotated.height.toFloat()
        val detections = objectResult.detections().mapNotNull { d ->
            val cat = d.categories().firstOrNull() ?: return@mapNotNull null
            val box = d.boundingBox()
            val rect = android.graphics.RectF(
                (box.left / w).coerceIn(0f, 1f),
                (box.top / h).coerceIn(0f, 1f),
                (box.right / w).coerceIn(0f, 1f),
                (box.bottom / h).coerceIn(0f, 1f)
            )
            val label = cat.categoryName() ?: "object"
            val held = palmCenters.any { p ->
                val padX = (rect.width() * 0.3f).coerceAtLeast(0.02f)
                val padY = (rect.height() * 0.3f).coerceAtLeast(0.02f)
                p.x in (rect.left - padX)..(rect.right + padX) &&
                    p.y in (rect.top - padY)..(rect.bottom + padY)
            }
            Detection(
                rect = rect,
                label = if (held) "$label • in hand" else label,
                score = cat.score(),
                danger = label.lowercase() in DANGER_LABELS
            )
        }

        val expression = if (faces.isNotEmpty()) classifyExpression(faces[0]) else "NO FACE"

        analyzedFrames++
        val now = SystemClock.elapsedRealtime()
        val elapsed = now - lastFpsTime
        if (elapsed >= 1000) {
            inferenceFps = analyzedFrames * 1000f / elapsed
            analyzedFrames = 0
            lastFpsTime = now
        }

        overlay.post {
            overlay.setSourceSize(rotated.width, rotated.height, mirrorX = true)
            overlay.update(
                faces = faces,
                hands = handPoints,
                poses = poses,
                detections = detections,
                expression = expression,
                fps = inferenceFps
            )
        }
    }

    private fun rotateBitmap(bitmap: android.graphics.Bitmap, degrees: Int): android.graphics.Bitmap {
        if (degrees == 0) return bitmap
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return android.graphics.Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
        )
    }

    private fun classifyExpression(face: List<P>): String {
        if (face.size < 300) return "FACE"
        val left = face[61]
        val right = face[291]
        val upper = face[13]
        val lower = face[14]
        val width = distance(left, right)
        val height = distance(upper, lower)
        if (width <= 0.001f) return "FACE"
        val ratio = height / width
        val centerY = (upper.y + lower.y) / 2f
        val cornersY = (left.y + right.y) / 2f
        val cornersUp = centerY - cornersY
        return when {
            cornersUp > 0.008f && width > 0.10f -> "SMILE :)"
            ratio > 0.28f -> "SURPRISED?"
            else -> "NEUTRAL"
        }
    }

    private fun distance(a: P, b: P): Float {
        val dx = a.x - b.x
        val dy = a.y - b.y
        return sqrt(dx * dx + dy * dy)
    }

    fun close() {
        executor.shutdownNow()
        face.close()
        hands.close()
        pose.close()
        objects.close()
    }
}
