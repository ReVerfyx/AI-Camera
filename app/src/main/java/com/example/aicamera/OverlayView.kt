package com.example.aicamera

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.max
import kotlin.math.min

data class P(val x: Float, val y: Float)

data class Detection(
    val rect: RectF,
    val label: String,
    val score: Float,
    val danger: Boolean = false
)

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
    }

    private var faces = emptyList<List<P>>()
    private var hands = emptyList<List<P>>()
    private var pose = emptyList<P>()
    private var detections = emptyList<Detection>()
    private var expression = "FACE"
    private var fps = 0f
    private var sourceW = 1
    private var sourceH = 1
    private var mirrorX = true

    fun setSourceSize(width: Int, height: Int, mirrorX: Boolean) {
        sourceW = max(1, width)
        sourceH = max(1, height)
        this.mirrorX = mirrorX
    }

    fun update(
        faces: List<List<P>>,
        hands: List<List<P>>,
        pose: List<P>,
        detections: List<Detection>,
        expression: String,
        fps: Float
    ) {
        this.faces = faces
        this.hands = hands
        this.pose = pose
        this.detections = detections
        this.expression = expression
        this.fps = fps
        postInvalidateOnAnimation()
    }

    override fun onDraw(c: Canvas) {
        super.onDraw(c)

        // Header similar to the reference, without covering the camera too much.
        paint.style = Paint.Style.FILL
        paint.color = Color.argb(175, 0, 0, 0)
        c.drawRect(12f, 12f, 360f, 112f, paint)

        text.color = Color.WHITE
        text.textSize = 22f
        c.drawText("AI CAMERA", 24f, 40f, text)
        text.textSize = 15f
        c.drawText("Возраст: —", 24f, 65f, text)
        c.drawText("Вес: —", 24f, 88f, text)

        drawPose(c, pose)
        hands.forEach { drawHand(c, it) }
        faces.forEach { drawFace(c, it) }
        detections.forEach { drawDetection(c, it) }

        text.textSize = 20f
        text.color = if (detections.any { it.danger }) Color.RED else Color.WHITE
        c.drawText(expression, 18f, height - 48f, text)
        text.textSize = 16f
        text.color = Color.YELLOW
        c.drawText("FPS: %.1f".format(fps), 18f, height - 20f, text)
    }

    /** Maps normalized MediaPipe coordinates to PreviewView coordinates.
     * PreviewView uses fillCenter, so the same center-crop must be applied here.
     */
    private fun xy(p: P): PointF {
        val vw = width.toFloat()
        val vh = height.toFloat()
        val sw = sourceW.toFloat()
        val sh = sourceH.toFloat()
        val scale = max(vw / sw, vh / sh)
        val rw = sw * scale
        val rh = sh * scale
        val left = (vw - rw) / 2f
        val top = (vh - rh) / 2f
        var x = left + p.x * rw
        val y = top + p.y * rh
        if (mirrorX) x = vw - x
        return PointF(x, y)
    }

    private fun drawPose(c: Canvas, p: List<P>) {
        if (p.size < 33) return
        val links = listOf(
            11 to 12, 11 to 13, 13 to 15, 12 to 14, 14 to 16,
            11 to 23, 12 to 24, 23 to 24,
            23 to 25, 25 to 27, 27 to 29, 29 to 31,
            24 to 26, 26 to 28, 28 to 30, 30 to 32
        )
        paint.style = Paint.Style.STROKE
        paint.color = Color.CYAN
        paint.strokeWidth = 4f
        for ((a, b) in links) {
            val pa = xy(p[a]); val pb = xy(p[b])
            c.drawLine(pa.x, pa.y, pb.x, pb.y, paint)
        }
        paint.style = Paint.Style.FILL
        paint.color = Color.YELLOW
        for (point in p) {
            val q = xy(point)
            c.drawCircle(q.x, q.y, 4f, paint)
        }
    }

    private fun drawHand(c: Canvas, hand: List<P>) {
        if (hand.size < 21) return
        val links = listOf(
            0 to 1, 1 to 2, 2 to 3, 3 to 4,
            0 to 5, 5 to 6, 6 to 7, 7 to 8,
            0 to 9, 9 to 10, 10 to 11, 11 to 12,
            0 to 13, 13 to 14, 14 to 15, 15 to 16,
            0 to 17, 17 to 18, 18 to 19, 19 to 20,
            5 to 9, 9 to 13, 13 to 17
        )
        paint.style = Paint.Style.STROKE
        paint.color = Color.MAGENTA
        paint.strokeWidth = 3f
        for ((a, b) in links) {
            val pa = xy(hand[a]); val pb = xy(hand[b])
            c.drawLine(pa.x, pa.y, pb.x, pb.y, paint)
        }
        paint.style = Paint.Style.FILL
        paint.color = Color.YELLOW
        hand.forEach {
            val q = xy(it)
            c.drawCircle(q.x, q.y, 4f, paint)
        }
    }

    private fun drawFace(c: Canvas, face: List<P>) {
        if (face.isEmpty()) return

        var minX = 1f; var minY = 1f; var maxX = 0f; var maxY = 0f
        face.forEach {
            minX = min(minX, it.x); minY = min(minY, it.y)
            maxX = max(maxX, it.x); maxY = max(maxY, it.y)
        }

        val a = xy(P(minX, minY))
        val b = xy(P(maxX, maxY))
        val rect = RectF(min(a.x, b.x) - 8f, min(a.y, b.y) - 8f,
            max(a.x, b.x) + 8f, max(a.y, b.y) + 8f)

        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 3f
        paint.color = Color.YELLOW
        c.drawRect(rect, paint)

        paint.style = Paint.Style.FILL
        paint.color = Color.GREEN
        face.forEach {
            val q = xy(it)
            c.drawCircle(q.x, q.y, 1.4f, paint)
        }

        text.textSize = 18f
        text.color = Color.GREEN
        c.drawText("person", rect.left, max(125f, rect.top - 8f), text)
    }

    private fun drawDetection(c: Canvas, d: Detection) {
        val tl = xy(P(d.rect.left, d.rect.top))
        val br = xy(P(d.rect.right, d.rect.bottom))
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = if (d.danger) 6f else 4f
        paint.color = if (d.danger) Color.RED else Color.GREEN
        c.drawRect(min(tl.x, br.x), min(tl.y, br.y), max(tl.x, br.x), max(tl.y, br.y), paint)
        paint.style = Paint.Style.FILL
        text.textSize = 18f
        text.color = paint.color
        c.drawText(if (d.danger) "WEAPON %.2f".format(d.score) else "${d.label} %.2f".format(d.score),
            min(tl.x, br.x), max(22f, min(tl.y, br.y) - 8f), text)
    }
}
