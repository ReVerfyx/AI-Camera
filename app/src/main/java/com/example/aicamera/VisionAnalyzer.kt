package com.example.aicamera

import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

/** Two-person pipeline: landmarks and the heavy YOLOv8x detector never block CameraX together. */
class VisionAnalyzer(
    private val context: android.content.Context,
    private val overlay: OverlayView
) : ImageAnalysis.Analyzer {

    private val face = FaceLandmarker.createFromOptions(context, FaceLandmarker.FaceLandmarkerOptions.builder()
        .setBaseOptions(BaseOptions.builder().setModelAssetPath("face_landmarker.task").build())
        .setRunningMode(RunningMode.VIDEO).setNumFaces(2).build())

    private val hands = HandLandmarker.createFromOptions(context, HandLandmarker.HandLandmarkerOptions.builder()
        .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build())
        .setRunningMode(RunningMode.VIDEO).setNumHands(4).build())

    private val pose = PoseLandmarker.createFromOptions(context, PoseLandmarker.PoseLandmarkerOptions.builder()
        .setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").build())
        .setRunningMode(RunningMode.VIDEO).setNumPoses(2).build())

    private val yolo = YoloDetector(context)
    private val landmarkExecutor = Executors.newSingleThreadExecutor()
    private val yoloExecutor = Executors.newSingleThreadExecutor()
    private val landmarkBusy = AtomicBoolean(false)
    private val yoloBusy = AtomicBoolean(false)

    @Volatile private var latestDetections: List<Detection> = emptyList()
    @Volatile private var lastYoloAt = 0L
    @Volatile private var latestInferenceFps = 0f
    private var fpsStart = SystemClock.elapsedRealtime()
    private var fpsFrames = 0

    override fun analyze(image: ImageProxy) {
        val rotation = image.imageInfo.rotationDegrees
        val bitmap = try {
            image.toBitmap().copy(Bitmap.Config.ARGB_8888, false)
        } finally {
            image.close()
        }
        val rotated = rotateBitmap(bitmap, rotation)

        if (landmarkBusy.compareAndSet(false, true)) {
            landmarkExecutor.execute {
                try { analyzeLandmarks(rotated) }
                catch (t: Throwable) { t.printStackTrace() }
                finally { landmarkBusy.set(false) }
            }
        }

        val now = SystemClock.elapsedRealtime()
        if (now - lastYoloAt >= 150L && yoloBusy.compareAndSet(false, true)) {
            lastYoloAt = now
            yoloExecutor.execute {
                try { latestDetections = yolo.detect(rotated, 0.35f, 0.45f) }
                catch (t: Throwable) { t.printStackTrace() }
                finally { yoloBusy.set(false) }
            }
        }
    }

    private fun analyzeLandmarks(bitmap: Bitmap) {
        val mpImage = BitmapImageBuilder(bitmap).build()
        val timestamp = SystemClock.elapsedRealtime()
        val faceResult = face.detectForVideo(mpImage, timestamp)
        val handResult = hands.detectForVideo(mpImage, timestamp)
        val poseResult = pose.detectForVideo(mpImage, timestamp)

        val faces = faceResult.faceLandmarks().take(2).map { list -> list.map { P(it.x(), it.y()) } }
        val handPoints = handResult.landmarks().take(4).map { list -> list.map { P(it.x(), it.y()) } }
        val poses = poseResult.landmarks().take(2).map { list -> list.map { P(it.x(), it.y()) } }
        val expression = if (faces.isNotEmpty()) classifyExpression(faces[0]) else "NO FACE"

        fpsFrames++
        val now = SystemClock.elapsedRealtime()
        val elapsed = now - fpsStart
        if (elapsed >= 1000L) {
            latestInferenceFps = fpsFrames * 1000f / elapsed
            fpsFrames = 0
            fpsStart = now
        }

        overlay.post {
            overlay.setSourceSize(bitmap.width, bitmap.height, mirrorX = true)
            overlay.update(faces, handPoints, poses, latestDetections, expression, latestInferenceFps)
        }
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun classifyExpression(face: List<P>): String {
        if (face.size < 300) return "FACE"
        val left = face[61]; val right = face[291]; val upper = face[13]; val lower = face[14]
        val width = distance(left, right); val height = distance(upper, lower)
        if (width <= 0.001f) return "FACE"
        val ratio = height / width
        val cornersUp = (upper.y + lower.y) / 2f - (left.y + right.y) / 2f
        return when {
            cornersUp > 0.008f && width > 0.10f -> "SMILE :)"
            ratio > 0.28f -> "SURPRISED?"
            else -> "NEUTRAL"
        }
    }

    private fun distance(a: P, b: P): Float {
        val dx = a.x - b.x; val dy = a.y - b.y
        return sqrt(dx * dx + dy * dy)
    }

    fun close() {
        landmarkExecutor.shutdownNow(); yoloExecutor.shutdownNow()
        face.close(); hands.close(); pose.close(); yolo.close()
    }
}
