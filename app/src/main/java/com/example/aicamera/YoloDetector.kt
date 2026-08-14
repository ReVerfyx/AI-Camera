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
        interpreter = Interpreter(mapped, Interpreter.Options().apply { setNumThreads(4) })
        inputTensor = interpreter.getInputTensor(0)
        outputTensor = interpreter.getOutputTensor(0)
        val s = inputTensor.shape()
        inputNhwc = s.size == 4 && s[3] == 3
        inputSize = when {
            inputNhwc && s.size == 4 -> s[1]
            s.size == 4 && s[1] == 3 -> s[2]
            else -> 640
        }
        labels = runCatching {
            context.assets.open("oiv7_labels.txt").bufferedReader().readLines().map(String::trim).filter(String::isNotEmpty)
        }.getOrElse { emptyList() }
    }

    fun detect(bitmap: Bitmap, scoreThreshold: Float = 0.18f, iouThreshold: Float = 0.45f): List<Detection> {
        val srcW = bitmap.width.toFloat(); val srcH = bitmap.height.toFloat()
        val scale = min(inputSize / srcW, inputSize / srcH)
        val rw = (srcW * scale).toInt().coerceAtLeast(1); val rh = (srcH * scale).toInt().coerceAtLeast(1)
        val padX = (inputSize - rw) / 2f; val padY = (inputSize - rh) / 2f
        val resized = Bitmap.createScaledBitmap(bitmap, rw, rh, true)
        val pixels = IntArray(rw * rh); resized.getPixels(pixels, 0, rw, 0, 0, rw, rh)
        val input = ByteBuffer.allocateDirect(inputTensor.numBytes()).order(ByteOrder.nativeOrder())
        val type = inputTensor.dataType(); val qs = inputTensor.quantizationParams().scale; val qz = inputTensor.quantizationParams().zeroPoint
        fun put(v: Float) {
            when (type) {
                DataType.FLOAT32 -> input.putFloat(v / 255f)
                DataType.UINT8 -> input.put(v.toInt().coerceIn(0, 255).toByte())
                DataType.INT8 -> input.put(if (qs != 0f) (v / 255f / qs + qz).toInt().coerceIn(-128, 127).toByte() else v.toInt().coerceIn(-128, 127).toByte())
                else -> input.putFloat(v / 255f)
            }
        }
        fun pixel(x: Int, y: Int, ch: Int): Float {
            val inside = x >= padX.toInt() && x < (padX + rw).toInt() && y >= padY.toInt() && y < (padY + rh).toInt()
            if (!inside) return 114f
            val sx = (x - padX).toInt().coerceIn(0, rw - 1); val sy = (y - padY).toInt().coerceIn(0, rh - 1)
            val c = pixels[sy * rw + sx]
            return when (ch) { 0 -> ((c shr 16) and 255).toFloat(); 1 -> ((c shr 8) and 255).toFloat(); else -> (c and 255).toFloat() }
        }
        if (inputNhwc) for (y in 0 until inputSize) for (x in 0 until inputSize) { put(pixel(x,y,0)); put(pixel(x,y,1)); put(pixel(x,y,2)) }
        else for (ch in 0..2) for (y in 0 until inputSize) for (x in 0 until inputSize) put(pixel(x,y,ch))
        input.rewind()

        val os = outputTensor.shape(); if (os.size != 3) return emptyList()
        val a = os[1]; val b = os[2]
        val channels: Int; val count: Int; val transposed: Boolean
        if (a <= 700) { channels = a; count = b; transposed = false } else { channels = b; count = a; transposed = true }
        if (channels < 6 || channels > 1000) return emptyList()
        val output = ByteBuffer.allocateDirect(outputTensor.numBytes()).order(ByteOrder.nativeOrder())
        interpreter.run(input, output); output.rewind()
        val oscale = outputTensor.quantizationParams().scale; val ozero = outputTensor.quantizationParams().zeroPoint; val ot = outputTensor.dataType()
        fun value(i: Int, c: Int): Float {
            val index = if (!transposed) c * count + i else i * channels + c
            return when (ot) {
                DataType.FLOAT32 -> output.getFloat(index * 4)
                DataType.UINT8 -> ((output.get(index).toInt() and 255) - ozero) * oscale
                DataType.INT8 -> (output.get(index).toInt() - ozero) * oscale
                DataType.INT16 -> (output.getShort(index * 2).toInt() - ozero) * oscale
                else -> 0f
            }
        }
        val candidates = ArrayList<Candidate>(); val classes = channels - 4
        for (i in 0 until count) {
            val cx=value(i,0); val cy=value(i,1); val w=value(i,2); val h=value(i,3)
            if (!cx.isFinite() || !cy.isFinite() || w <= 0f || h <= 0f) continue
            var best=-1; var score=0f
            for (c in 0 until classes) { val s=value(i,4+c); if (s > score) { score=s; best=c } }
            if (best < 0 || score < scoreThreshold) continue
            val l=((cx-w/2f-padX)/scale/srcW).coerceIn(0f,1f); val t=((cy-h/2f-padY)/scale/srcH).coerceIn(0f,1f)
            val r=((cx+w/2f-padX)/scale/srcW).coerceIn(0f,1f); val bot=((cy+h/2f-padY)/scale/srcH).coerceIn(0f,1f)
            if (r<=l || bot<=t) continue
            val name=labels.getOrNull(best) ?: "class_$best"
            candidates += Candidate(RectF(l,t,r,bot),name,score,isDanger(name))
        }
        return nms(candidates,iouThreshold).map { Detection(it.rect,it.label,it.score,it.danger) }
    }

    private fun nms(items: List<Candidate>, threshold: Float): List<Candidate> {
        val result=ArrayList<Candidate>()
        for (item in items.sortedByDescending { it.score }) {
            if (result.none { it.label==item.label && iou(it.rect,item.rect)>threshold }) result += item
            if (result.size>=30) break
        }
        return result
    }
    private fun iou(a:RectF,b:RectF):Float { val l=max(a.left,b.left); val t=max(a.top,b.top); val r=min(a.right,b.right); val bot=min(a.bottom,b.bottom); val inter=max(0f,r-l)*max(0f,bot-t); val union=a.width()*a.height()+b.width()*b.height()-inter; return if(union<=0f) 0f else inter/union }
    private fun isDanger(name:String):Boolean { val n=name.lowercase(); return n.contains("handgun")||n.contains("rifle")||n.contains("shotgun")||n.contains("knife")||n.contains("dagger")||n.contains("sword")||n.contains("missile")||n.contains("bomb")||n.contains("cannon")||n.contains("axe")||n.contains("firearm")||n.contains("gun")||n=="weapon" }
    override fun close()=interpreter.close()
    private data class Candidate(val rect:RectF,val label:String,val score:Float,val danger:Boolean)
}
