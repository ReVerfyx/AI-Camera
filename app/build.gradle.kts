import java.net.URI

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.aicamera"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.aicamera"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.10.1")
    implementation("androidx.camera:camera-core:1.4.2")
    implementation("androidx.camera:camera-camera2:1.4.2")
    implementation("androidx.camera:camera-lifecycle:1.4.2")
    implementation("androidx.camera:camera-view:1.4.2")
    implementation("com.google.mediapipe:tasks-vision:0.10.35")
    implementation("org.tensorflow:tensorflow-lite:2.17.0")
}

val modelDownloads = mapOf(
    "face_landmarker.task" to "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task" to "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task" to "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

tasks.register("downloadMediapipeModels") {
    val outFiles = modelDownloads.keys.map { file("src/main/assets/$it") }
    outputs.files(outFiles)
    doLast {
        modelDownloads.forEach { (name, url) ->
            val outFile = file("src/main/assets/$name")
            if (!outFile.exists()) {
                outFile.parentFile.mkdirs()
                ant.withGroovyBuilder { "get"("src" to url, "dest" to outFile, "verbose" to true) }
            }
        }
    }
}

tasks.register("downloadOiv7Labels") {
    val outFile = file("src/main/assets/oiv7_labels.txt")
    outputs.file(outFile)
    doLast {
        if (!outFile.exists()) {
            val yaml = URI("https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/open-images-v7.yaml").toURL().readText()
            val labels = yaml.lineSequence()
                .mapNotNull { line -> Regex("^\\s*(\\d+):\\s*(.+?)\\s*$").find(line)?.let { m -> m.groupValues[1].toInt() to m.groupValues[2].trim() } }
                .sortedBy { it.first }
                .map { it.second }
                .joinToString("\\n") + "\\n"
            outFile.parentFile.mkdirs()
            outFile.writeText(labels)
        }
    }
}

tasks.named("preBuild") {
    dependsOn("downloadMediapipeModels", "downloadOiv7Labels")
}
