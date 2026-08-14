package com.example.aicamera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

class YoloDetector(context: Context) : AutoCloseable {
    private val inputSize: Int
    private val interpreter: Interpreter
    private val labels: List<String>

    init {
        val afd = context.assets.openFd("yolov8x-oiv7.tflite")
        val mapped = FileInputStream(afd.fileDescriptor).channel.use { channel ->
            channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
        afd.close()
        interpreter = Interpreter(mapped, Interpreter.Options().setNumThreads(4))
        val shape = interpreter.getInputTensor(0).shape()
        inputSize = if (shape.size >= 3) max(shape[1], shape[2]) else 640
        labels = runCatching {
            context.assets.open("oiv7_labels.txt").bufferedReader().useLines { it.toList() }
        }.getOrElse { emptyList() }
    }

    fun detect(bitmap: Bitmap, scoreThreshold: Float = 0.35f, iouThreshold: Float = 0.45f): List<Detection> {
        val srcW = bitmap.width.toFloat()
        val srcH = bitmap.height.toFloat()
        val scale = min(inputSize / srcW, inputSize / srcH)
        val resizedW = (srcW * scale).toInt().coerceAtLeast(1)
        val resizedH = (srcH * scale).toInt().coerceAtLeast(1)
        val padX = (inputSize - resizedW) / 2f
        val padY = (inputSize - resizedH) / 2f
        val resized = Bitmap.createScaledBitmap(bitmap, resizedW, resizedH, true)
        val input = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4).order(ByteOrder.nativeOrder())
        val pixels = IntArray(resizedW * resizedH)
        resized.getPixels(pixels, 0, resizedW, 0, 0, resizedW, resizedH)
        for (y in 0 until inputSize) for (x in 0 until inputSize) {
            val inside = x >= padX.toInt() && x < (padX + resizedW).toInt() && y >= padY.toInt() && y < (padY + resizedH).toInt()
            if (inside) {
                val sx = (x - padX).coerceIn(0f, (resizedW - 1).toFloat()).toInt()
                val sy = (y - padY).coerceIn(0f, (resizedH - 1).toFloat()).toInt()
                val c = pixels[sy * resizedW + sx]
                input.putFloat(((c shr 16) and 255) / 255f)
                input.putFloat(((c shr 8) and 255) / 255f)
                input.putFloat((c and 255) / 255f)
            } else {
                input.putFloat(114f / 255f); input.putFloat(114f / 255f); input.putFloat(114f / 255f)
            }
        }
        input.rewind()

        val shape = interpreter.getOutputTensor(0).shape()
        if (shape.size != 3) return emptyList()
        val a = shape[1]
        val b = shape[2]
        val channels: Int
        val count: Int
        val transposed: Boolean
        if (a <= 700) { channels = a; count = b; transposed = false }
        else { channels = b; count = a; transposed = true }
        if (channels < 6 || channels > 1000) return emptyList()

        // Keep the exact tensor layout expected by TFLite; value() handles both YOLO layouts.
        val output = Array(1) { Array(a) { FloatArray(b) } }
        interpreter.run(input, output)

        val classes = channels - 4
        val candidates = ArrayList<Candidate>()
        for (i in 0 until count) {
            val cx = value(output, i, 0, transposed)
            val cy = value(output, i, 1, transposed)
            val w = value(output, i, 2, transposed)
            val h = value(output, i, 3, transposed)
            var bestClass = -1
            var bestScore = 0f
            for (c in 0 until classes) {
                val score = value(output, i, 4 + c, transposed)
                if (score > bestScore) { bestScore = score; bestClass = c }
            }
            if (bestClass < 0 || bestScore < scoreThreshold) continue
            val left = ((cx - w / 2f - padX) / scale / srcW).coerceIn(0f, 1f)
            val top = ((cy - h / 2f - padY) / scale / srcH).coerceIn(0f, 1f)
            val right = ((cx + w / 2f - padX) / scale / srcW).coerceIn(0f, 1f)
            val bottom = ((cy + h / 2f - padY) / scale / srcH).coerceIn(0f, 1f)
            if (right <= left || bottom <= top) continue
            val name = labels.getOrNull(bestClass) ?: "class_$bestClass"
            candidates += Candidate(RectF(left, top, right, bottom), name, bestScore, isDanger(name))
        }
        return nms(candidates, iouThreshold).map { Detection(it.rect, it.label, it.score, it.danger) }
    }

    private fun value(out: Array<Array<FloatArray>>, i: Int, c: Int, transposed: Boolean): Float =
        if (!transposed) out[0][c][i] else out[0][i][c]

    private fun nms(items: List<Candidate>, threshold: Float): List<Candidate> {
        val result = ArrayList<Candidate>()
        items.sortedByDescending { it.score }.forEach { item ->
            if (result.none { it.label == item.label && iou(it.rect, item.rect) > threshold }) result += item
        }
        return result.take(30)
    }

    private fun iou(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left); val top = max(a.top, b.top)
        val right = min(a.right, b.right); val bottom = min(a.bottom, b.bottom)
        val inter = max(0f, right - left) * max(0f, bottom - top)
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }

    private fun isDanger(name: String): Boolean {
        val n = name.lowercase()
        return n.contains("handgun") || n.contains("rifle") || n.contains("shotgun") ||
            n.contains("knife") || n.contains("dagger") || n.contains("sword") ||
            n.contains("missile") || n.contains("bomb") || n.contains("cannon") ||
            n.contains("axe") || n.contains("firearm") || n.contains("gun") || n == "weapon"
    }

    override fun close() = interpreter.close()
    private data class Candidate(val rect: RectF, val label: String, val score: Float, val danger: Boolean)
}
