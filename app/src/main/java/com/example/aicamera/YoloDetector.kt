package com.example.aicamera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

/**
 * YOLOv8 TFLite decoder. Supports both [1,640,640,3] and [1,3,640,640]
 * inputs and float/quantized tensors. OIV7 export normally has 605 channels
 * (4 box values + 601 classes) and either [1,605,8400] or [1,8400,605] output.
 */
class YoloDetector(context: Context) : AutoCloseable {
    private val interpreter: Interpreter
    private val inputTensor: org.tensorflow.lite.Tensor
    private val outputTensor: org.tensorflow.lite.Tensor
    private val inputSize: Int
    private val inputNhwc: Boolean
    private val labels: List<String>

    init {
        val afd = context.assets.openFd("yolov8x-oiv7.tflite")
        val mapped = FileInputStream(afd.fileDescriptor).channel.use { channel ->
            channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
        afd.close()

        interpreter = Interpreter(mapped, Interpreter.Options().apply {
            setNumThreads(4)
        })
        inputTensor = interpreter.getInputTensor(0)
        outputTensor = interpreter.getOutputTensor(0)

        val inShape = inputTensor.shape()
        inputNhwc = inShape.size == 4 && inShape[3] == 3
        inputSize = when {
            inputNhwc && inShape.size == 4 -> inShape[1]
            inShape.size == 4 && inShape[1] == 3 -> inShape[2]
            else -> 640
        }

        labels = runCatching {
            context.assets.open("oiv7_labels.txt").bufferedReader().readLines()
                .map { it.trim() }.filter { it.isNotEmpty() }
        }.getOrElse { emptyList() }
    }

    fun detect(bitmap: Bitmap, scoreThreshold: Float = 0.18f, iouThreshold: Float = 0.45f): List<Detection> {
        val srcW = bitmap.width.toFloat()
        val srcH = bitmap.height.toFloat()

        // Letterbox exactly like Ultralytics: preserve aspect ratio and pad with 114.
        val scale = min(inputSize / srcW, inputSize / srcH)
        val resizedW = (srcW * scale).toInt().coerceAtLeast(1)
        val resizedH = (srcH * scale).toInt().coerceAtLeast(1)
        val padX = (inputSize - resizedW) / 2f
        val padY = (inputSize - resizedH) / 2f
        val resized = Bitmap.createScaledBitmap(bitmap, resizedW, resizedH, true)
        val pixels = IntArray(resizedW * resizedH)
        resized.getPixels(pixels, 0, resizedW, 0, 0, resizedW, resizedH)

        val input = ByteBuffer.allocateDirect(inputTensor.numBytes()).order(ByteOrder.nativeOrder())
        val inputType = inputTensor.dataType()
        val qScale = inputTensor.quantizationParams().scale
        val qZero = inputTensor.quantizationParams().zeroPoint

        fun putChannel(v: Float) {
            when (inputType) {
                DataType.FLOAT32 -> input.putFloat(v / 255f)
                DataType.UINT8 -> input.put((v.toInt().coerceIn(0, 255)).toByte())
                DataType.INT8 -> {
                    val q = if (qScale != 0f) (v / 255f / qScale + qZero).toInt() else v.toInt()
                    input.put(q.coerceIn(-128, 127).toByte())
                }
                else -> input.putFloat(v / 255f)
            }
        }

        if (inputNhwc) {
            for (y in 0 until inputSize) {
                for (x in 0 until inputSize) {
                    val inside = x >= padX.toInt() && x < (padX + resizedW).toInt() &&
                        y >= padY.toInt() && y < (padY + resizedH).toInt()
                    if (inside) {
                        val sx = (x - padX).toInt().coerceIn(0, resizedW - 1)
                        val sy = (y - padY).toInt().coerceIn(0, resizedH - 1)
                        val c = pixels[sy * resizedW + sx]
                        putChannel(((c shr 16) and 255).toFloat())
                        putChannel(((c shr 8) and 255).toFloat())
                        putChannel((c and 255).toFloat())
                    } else {
                        putChannel(114f); putChannel(114f); putChannel(114f)
                    }
                }
            }
        } else {
            // NCHW input: write complete R, G, B planes.
            for (channel in 0..2) {
                for (y in 0 until inputSize) {
                    for (x in 0 until inputSize) {
                        val inside = x >= padX.toInt() && x < (padX + resizedW).toInt() &&
                            y >= padY.toInt() && y < (padY + resizedH).toInt()
                        val v = if (inside) {
                            val sx = (x - padX).toInt().coerceIn(0, resizedW - 1)
                            val sy = (y - padY).toInt().coerceIn(0, resizedH - 1)
                            val c = pixels[sy * resizedW + sx]
                            when (channel) { 0 -> ((c shr 16) and 255).toFloat(); 1 -> ((c shr 8) and 255).toFloat(); else -> (c and 255).toFloat() }
                        } else 114f
                        putChannel(v)
                    }
                }
            }
        }
        input.rewind()

        val shape = outputTensor.shape()
        if (shape.size != 3) return emptyList()
        val a = shape[1]
        val b = shape[2]
        val channels: Int
        val count: Int
        val transposed: Boolean
        if (a <= 700) { channels = a; count = b; transposed = false }
        else { channels = b; count = a; transposed = true }
        if (channels < 6 || channels > 1000 || count <= 0) return emptyList()

        val output = ByteBuffer.allocateDirect(outputTensor.numBytes()).order(ByteOrder.nativeOrder())
        interpreter.run(input, output)
        output.rewind()

        fun value(i: Int, c: Int): Float {
            val index = if (!transposed) c * count + i else i * channels + c
            return readValue(output, index, outputTensor.dataType(), outputTensor.quantizationParams().scale, outputTensor.quantizationParams().zeroPoint)
        }

        val classes = channels - 4
        val candidates = ArrayList<Candidate>()
        for (i in 0 until count) {
            val cx = value(i, 0)
            val cy = value(i, 1)
            val w = value(i, 2)
            val h = value(i, 3)
            if (!cx.isFinite() || !cy.isFinite() || !w.isFinite() || !h.isFinite() || w <= 0f || h <= 0f) continue

            var bestClass = -1
            var bestScore = 0f
            for (c in 0 until classes) {
                val score = value(i, 4 + c)
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

    private fun readValue(buffer: ByteBuffer, index: Int, type: DataType, scale: Float, zero: Int): Float {
        return when (type) {
            DataType.FLOAT32 -> buffer.getFloat(index * 4)
            DataType.FLOAT16 -> (buffer.getShort(index * 2).toInt() and 0xffff).toFloat() // uncommon; model exports normally use float32
            DataType.UINT8 -> ((buffer.get(index).toInt() and 0xff) - zero) * scale
            DataType.INT8 -> (buffer.get(index).toInt() - zero) * scale
            DataType.INT16 -> (buffer.getShort(index * 2).toInt() - zero) * scale
            else -> 0f
        }
    }

    private fun nms(items: List<Candidate>, threshold: Float): List<Candidate> {
        val result = ArrayList<Candidate>()
        for (item in items.sortedByDescending { it.score }) {
            if (result.none { iou(it.rect, item.rect) > threshold && it.label == item.label }) result += item
            if (result.size >= 30) break
        }
        return result
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
