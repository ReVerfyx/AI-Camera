package com.example.aicamera

import android.content.Context
import android.graphics.Matrix
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import kotlin.math.sqrt

// COCO labels где предмет в руке потенциально опасен — подсвечиваем красным.
private val DANGER_LABELS = setOf("knife", "scissors", "baseball bat")

class VisionAnalyzer(
    private val context: Context,
    private val overlay: OverlayView
) : ImageAnalysis.Analyzer {

    private val face: FaceLandmarker
    private val hands: HandLandmarker
    private val pose: PoseLandmarker
    private val objects: ObjectDetector

    private var lastFpsTime = System.currentTimeMillis()
    private var frameCount = 0
    private var fps = 0f

    init {
        face = FaceLandmarker.createFromOptions(
            context,
            FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("face_landmarker.task").build())
                .setRunningMode(RunningMode.VIDEO)
                .setNumFaces(4) // несколько лиц/людей в кадре
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
                .setNumPoses(4)
                .build()
        )

        // Общий детектор предметов: 80 классов COCO — человек, кошка, собака,
        // бутылка, телефон, книга, нож, ножницы, сумка и т.д.
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
        try {
            val raw = image.toBitmap()
            val rotated = rotateBitmap(raw, image.imageInfo.rotationDegrees)
            val mpImage = BitmapImageBuilder(rotated).build()
            val timestamp = System.currentTimeMillis()

            val faceResult = face.detectForVideo(mpImage, timestamp)
            val handResult = hands.detectForVideo(mpImage, timestamp)
            val poseResult = pose.detectForVideo(mpImage, timestamp)
            val objectResult = objects.detectForVideo(mpImage, timestamp)

            val faces = faceResult.faceLandmarks().map { list ->
                list.map { P(it.x(), it.y()) }
            }
            val handPoints = handResult.landmarks().map { list ->
                list.map { P(it.x(), it.y()) }
            }
            val posePoints = if (poseResult.landmarks().isNotEmpty()) {
                poseResult.landmarks()[0].map { P(it.x(), it.y()) }
            } else emptyList()

            // Центр ладони каждой руки (для привязки "предмет в руке").
            val palmCenters = handPoints.mapNotNull { h ->
                if (h.size < 21) null
                else {
                    val ids = listOf(0, 5, 9, 13, 17)
                    val cx = ids.map { h[it].x }.average().toFloat()
                    val cy = ids.map { h[it].y }.average().toFloat()
                    P(cx, cy)
                }
            }

            val w = rotated.width.toFloat()
            val h = rotated.height.toFloat()
            val detections = objectResult.detections().mapNotNull { d ->
                val cat = d.categories().firstOrNull() ?: return@mapNotNull null
                val box = d.boundingBox() // пиксели исходного (повёрнутого) кадра
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
                    danger = label in DANGER_LABELS
                )
            }

            val expression = if (faces.isNotEmpty()) classifyExpression(faces[0]) else "NO FACE"

            frameCount++
            val now = System.currentTimeMillis()
            val elapsed = now - lastFpsTime
            if (elapsed >= 1000) {
                fps = frameCount * 1000f / elapsed
                frameCount = 0
                lastFpsTime = now
            }

            overlay.setSourceSize(rotated.width, rotated.height, mirrorX = true)
            overlay.update(
                faces = faces,
                hands = handPoints,
                pose = posePoints,
                detections = detections,
                expression = expression,
                fps = fps
            )
        } catch (t: Throwable) {
            t.printStackTrace()
        } finally {
            image.close()
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
}
