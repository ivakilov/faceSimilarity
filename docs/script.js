import { FaceDetector, ImageEmbedder, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const demosSection = document.getElementById("demos");
const imageCanvas = document.getElementsByClassName("image_canvas")[0];
const imageCanvas2 = document.getElementsByClassName("image_canvas2")[0];
const videoCanvas = document.getElementsByClassName("video_canvas")[0];
let imageFaceCropped; // Вырезанная область лица на фотографии
let videoFaceCropped; // Вырезанная область лица на видео
let imageEmbedderResult; // Эмбеддинг для изображения
let videoImageEmbedderResult; // Эмбеддинг для изображения
const embeddingImage = document.getElementById("embedImage");
// const embeddingImage2 = document.getElementById("embedImage2");
const similarity = document.getElementById("similarity");
let faceDetector;
let imageEmbedder;
let runningMode = "IMAGE";
// Initialize the object detector
const initializefaceDetectorAndEmbedder = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    
    // faceDetector
    faceDetector = await FaceDetector.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`,
            delegate: "GPU"
        },
        runningMode: runningMode
    });

    // Embedder
    imageEmbedder = await ImageEmbedder.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite`
        },
        runningMode: runningMode
    });
    demosSection.classList.remove("invisible");
};
initializefaceDetectorAndEmbedder();
/********************************************************************
 // Demo 1: Grab a bunch of images from the page and detection them
 // upon click.
 ********************************************************************/
const imageContainers = document.getElementsByClassName("detectOnClick");

for (let imageContainer of imageContainers) {
    imageContainer.children[0].addEventListener("click", handleClick);
}
/**
 * Detect faces in still images on click
 */
async function handleClick(event) {
    const highlighters = event.target.parentNode.getElementsByClassName("highlighter");
    while (highlighters[0]) {
        highlighters[0].parentNode.removeChild(highlighters[0]);
    }
    const infos = event.target.parentNode.getElementsByClassName("info");
    while (infos[0]) {
        infos[0].parentNode.removeChild(infos[0]);
    }
    const keyPoints = event.target.parentNode.getElementsByClassName("key-point");
    while (keyPoints[0]) {
        keyPoints[0].parentNode.removeChild(keyPoints[0]);
    }
    if (!faceDetector) {
        console.log("Wait for objectDetector to load before clicking");
        return;
    }
    // if video mode is initialized, set runningMode to image
    if (runningMode === "VIDEO") {
        runningMode = "IMAGE";
        await faceDetector.setOptions({ runningMode: "IMAGE" });
    }
    // faceDetector.detect returns a promise which, when resolved, is an array of Detection faces
    const detections = faceDetector.detect(event.target).detections;
    displayImageDetections(detections, event.target);

    if(event.target.id == 'inputImage'){
        imageFaceCropped = cropFace(detections, event.target);
        drawImageOnCanvas(imageFaceCropped, imageCanvas);
        const croppedImageEmbedderResult = await imageEmbedder.embed(imageFaceCropped);
        showEmbedding(croppedImageEmbedderResult, embeddingImage)
    }
}

function cropFace(detections, image) {

    // Определяем границы лица
    let minX = detections[0].boundingBox.originX;
    let maxX = detections[0].boundingBox.originX + detections[0].boundingBox.width;
    let minY = detections[0].boundingBox.originY;
    let maxY = detections[0].boundingBox.originY + detections[0].boundingBox.height;

    let width = maxX - minX;
    let height = maxY - minY;

    // Создаем новый canvas для вырезанного лица
    let faceCanvas = document.createElement("canvas");
    faceCanvas.width = width;
    faceCanvas.height = height;
    let ctx = faceCanvas.getContext("2d");

    // Вырезаем область лица и рисуем в новом canvas
    ctx.drawImage(image, minX, minY, width, height, 0, 0, width, height);

    return faceCanvas;
}

function drawImageOnCanvas(image, locImageCanvas){
    
    // Получаем контекст канваса
    const imageCanvasctx = locImageCanvas.getContext("2d");

    // Размеры изображения
    const imgWidth = image.width;
    const imgHeight = image.height;

    // Размеры canvas
    const canvasWidth = locImageCanvas.width;
    const canvasHeight = locImageCanvas.height;

    // Вычисляем масштаб, чтобы изображение вписывалось в canvas, сохраняя пропорции
    const scale = Math.min(canvasWidth / imgWidth, canvasHeight / imgHeight);

    // Новые размеры изображения
    const newWidth = imgWidth * scale;
    const newHeight = imgHeight * scale;

    // Координаты для центрирования
    const xOffset = (canvasWidth - newWidth) / 2;
    const yOffset = (canvasHeight - newHeight) / 2;    

    // Очистка canvas перед рисованием (если необходимо)
    imageCanvasctx.clearRect(xOffset, yOffset, newWidth, newHeight);

    // Рисуем изображение с учетом масштабирования и центрирования
    imageCanvasctx.drawImage(image, xOffset, yOffset, newWidth, newHeight);
    
    // locImageCanvas.style.filter = "brightness(400%)";
}

function displayImageDetections(detections, resultElement) {
    const ratio = resultElement.height / resultElement.naturalHeight;
    // console.log(ratio);
    for (let detection of detections) {
        const highlighter = document.createElement("div");
        highlighter.setAttribute("class", "highlighter");
        highlighter.style =
            "left: " +
                detection.boundingBox.originX * ratio +
                "px;" +
                "top: " +
                detection.boundingBox.originY * ratio +
                "px;" +
                "width: " +
                detection.boundingBox.width * ratio +
                "px;" +
                "height: " +
                detection.boundingBox.height * ratio +
                "px;";
        resultElement.parentNode.appendChild(highlighter);
    }
}
/********************************************************************
 // Demo 2: Continuously grab image from webcam stream and detect it.
 ********************************************************************/
let video = document.getElementById("webcam");
const liveView = document.getElementById("liveView");
let enableWebcamButton;
// Check if webcam access is supported.
const hasGetUserMedia = () => { var _a; return !!((_a = navigator.mediaDevices) === null || _a === void 0 ? void 0 : _a.getUserMedia); };
// Keep a reference of all the child elements we create
// so we can remove them easilly on each render.
var children = [];
// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
}
else {
    console.warn("getUserMedia() is not supported by your browser");
}
// Enable the live webcam view and start detection.
async function enableCam(event) {
    if (!faceDetector) {
        alert("Face Detector is still loading. Please try again..");
        return;
    }
    // Hide the button.
    enableWebcamButton.classList.add("removed");
    // getUsermedia parameters
    const constraints = {
        video: true
    };
    // Activate the webcam stream.
    navigator.mediaDevices
        .getUserMedia(constraints)
        .then(function (stream) {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    })
        .catch((err) => {
        console.error(err);
    });
}
let lastVideoTime = -1;
async function predictWebcam() {
    // if image mode is initialized, create a new classifier with video runningMode
    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await faceDetector.setOptions({ runningMode: "VIDEO" });
    }
    let startTimeMs = performance.now();
    // Detect faces using detectForVideo
    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const detections = faceDetector.detectForVideo(video, startTimeMs)
            .detections;
        displayVideoDetections(detections);
        const canvasCrop = getVideoImage(video)
        videoFaceCropped = cropFace(detections, canvasCrop)
        drawImageOnCanvas(videoFaceCropped, videoCanvas)
        videoImageEmbedderResult = await imageEmbedder.embed(videoFaceCropped);
        // showEmbedding(videoImageEmbedderResult, embeddingVideoImage)
    }
    if(!imageEmbedderResult && imageFaceCropped != null){
        imageEmbedderResult = await imageEmbedder.embed(imageFaceCropped);
    }
    if(videoFaceCropped != null && imageEmbedderResult != null){
        videoImageEmbedderResult = await imageEmbedder.embed(videoFaceCropped);       
        showSimilarity(similarity)
    }
    // Call this function again to keep predicting when the browser is ready
    window.requestAnimationFrame(predictWebcam);
}

function displayVideoDetections(detections) {
    // Remove any highlighting from previous frame.
    for (let child of children) {
        liveView.removeChild(child);
    }
    children.splice(0);
    // Iterate through predictions and draw them to the live view
    for (let detection of detections) {
        const highlighter = document.createElement("div");
        highlighter.setAttribute("class", "highlighter");
        highlighter.style =
            "left: " +
                (video.offsetWidth -
                    detection.boundingBox.width -
                    detection.boundingBox.originX) +
                "px;" +
                "top: " +
                detection.boundingBox.originY +
                "px;" +
                "width: " +
                (detection.boundingBox.width - 10) +
                "px;" +
                "height: " +
                detection.boundingBox.height +
                "px;";
        liveView.appendChild(highlighter);
        children.push(highlighter);
    }
}

function getVideoImage(video){
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext('2d');    

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas
}

function showEmbedding(imageEmbedderResult, element){
    const truncatedEmbedding = imageEmbedderResult.embeddings[0].floatEmbedding;
    truncatedEmbedding.length = 4;
    element.innerText = `Float Embedding: ${truncatedEmbedding}...`;
}

function showSimilarity(element){
    const similarityImage = ImageEmbedder.cosineSimilarity(imageEmbedderResult.embeddings[0], videoImageEmbedderResult.embeddings[0]);
    element.innerText = `Similarity: ${similarityImage}`;
}