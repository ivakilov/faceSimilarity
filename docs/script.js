import { FaceDetector, ImageEmbedder, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const demosSection = document.getElementById("demos");
const imageCanvas = document.getElementsByClassName("image_canvas")[0];
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
        runningMode: runningMode,
        minDetectionConfidence: 0.5,
        numFaces: 1 // Максимальное количество обнаруживаемых лиц
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

    if(detections.length){ // check detections is not empty  
        const HeadRotationAngle = calculateHeadRotationAngle(detections);    
        displayImageDetections(detections, event.target, -HeadRotationAngle);// * (-1); // домножение на -1 при неотзеркаленном изображении 
        imageFaceCropped = cropFace(detections, event.target, HeadRotationAngle);
        drawImageOnCanvas(imageFaceCropped, imageCanvas);
        const croppedImageEmbedderResult = await imageEmbedder.embed(imageFaceCropped);
        showEmbedding(croppedImageEmbedderResult, embeddingImage)
    }
}


function cropFace(detections, image, angle) {
    if (detections.length === 0) return null;

    const detection = detections[0];
    const bbox = detection.boundingBox;
    
    // Получаем исходные параметры bounding box
    const width = bbox.width;
    const height = bbox.height;
    
    // Корректируем размеры с учетом поворота
    const [adjustedWidth, adjustedHeight] = adjustBoundingBox(width, height, angle);

    // Центр лица до поворота
    const centerX = bbox.originX + width/2;
    const centerY = bbox.originY + height/2;

    // Создаем временный canvas для поворота
    const offscreenCanvas = document.createElement("canvas");
    const ctx = offscreenCanvas.getContext("2d");

    // Увеличиваем размер canvas для аккомодации поворота
    offscreenCanvas.width = Math.max(image.width, centerX * 2);
    offscreenCanvas.height = Math.max(image.height, centerY * 2);

    // Переносим начало координат в центр лица
    ctx.translate(centerX, centerY);
    ctx.rotate(angle * Math.PI / 180);
    ctx.translate(-centerX, -centerY);

    // Рисуем исходное изображение
    ctx.drawImage(image, 0, 0);

    // Рассчитываем новые границы с учетом поворота
    const rotationMatrix = [
        Math.cos(angle * Math.PI / 180), 
        Math.sin(angle * Math.PI / 180),
        -Math.sin(angle * Math.PI / 180), 
        Math.cos(angle * Math.PI / 180)
    ];

    // Корректируем координаты вырезаемой области
    const dx = adjustedWidth/2 - width/2;
    const dy = adjustedHeight/2 - height/2;
    
    const rotatedMinX = centerX - adjustedWidth/2 + rotationMatrix[0] * dx + rotationMatrix[1] * dy;
    const rotatedMinY = centerY - adjustedHeight/2 + rotationMatrix[2] * dx + rotationMatrix[3] * dy;

    // Создаем итоговый canvas
    const faceCanvas = document.createElement("canvas");
    faceCanvas.width = adjustedWidth;
    faceCanvas.height = adjustedHeight;
    const faceCtx = faceCanvas.getContext("2d");

    // Вырезаем корректную область
    faceCtx.drawImage(
        offscreenCanvas,
        Math.max(0, rotatedMinX),
        Math.max(0, rotatedMinY),
        adjustedWidth,
        adjustedHeight,
        0,
        0,
        adjustedWidth,
        adjustedHeight
    );

    return faceCanvas;
}


function adjustBoundingBox(originalWidth, originalHeight, angle) {
    const scaleFactor = 0.2; // Коэффициент расширения (можно настроить)
    const angleLimit = 45; // Максимальный угол для масштабирования (чтобы не раздувалось бесконечно)
    
    const adjustedWidth = originalWidth * (1 + scaleFactor * Math.min(Math.abs(angle), angleLimit) / angleLimit);
    const adjustedHeight = originalHeight * (1 + scaleFactor * Math.min(Math.abs(angle), angleLimit) / angleLimit);
    return [adjustedWidth,adjustedHeight];
}

function drawImageOnCanvas(image, locImageCanvas) {
    const imageCanvasctx = locImageCanvas.getContext("2d");
    const imgWidth = image.width;
    const imgHeight = image.height;
    const canvasWidth = locImageCanvas.width;
    const canvasHeight = locImageCanvas.height;

    const scale = Math.min(canvasWidth / imgWidth, canvasHeight / imgHeight);
    const newWidth = imgWidth * scale;
    const newHeight = imgHeight * scale;

    // Вычисляем координаты для центрирования
    const x = (canvasWidth - newWidth) / 2;
    const y = (canvasHeight - newHeight) / 2;

    imageCanvasctx.clearRect(0, 0, canvasWidth, canvasHeight);
    imageCanvasctx.drawImage(image, x, y, newWidth, newHeight); // Исправлено: добавлены x и y
}


function displayImageDetections(detections, resultElement, angle) {
    const ratio = resultElement.height / resultElement.naturalHeight;
    // console.log(ratio);
    for (let detection of detections) {
        const highlighter = document.createElement("div");
        highlighter.setAttribute("class", "highlighter");

        // Коррекция ширины бокса в зависимости от угла наклона лица
        const [adjustedWidth, adjustedHeight] = adjustBoundingBox(detection.boundingBox.width * ratio, detection.boundingBox.height * ratio, angle);

        // Вычисляем центр прямоугольника
        const centerX = (detection.boundingBox.originX * ratio + detection.boundingBox.width * ratio / 2)
        const centerY = (detection.boundingBox.originY * ratio + detection.boundingBox.height * ratio / 2)

        highlighter.style.left = `${centerX}px`;
        highlighter.style.top = `${centerY}px`;
        highlighter.style.width = `${adjustedWidth}px`;
        highlighter.style.height = `${adjustedHeight}px`;

        // Добавляем поворот вокруг центра
        highlighter.style.transform = `translate(-50%, -50%) rotate(${angle}deg)`;
        highlighter.style.transformOrigin = "center";

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

        if(detections.length){ // check detections is not empty            
            const HeadRotationAngle = calculateHeadRotationAngle(detections);
            displayVideoDetections(detections, HeadRotationAngle);
            const canvasCrop = getVideoImage(video)
            videoFaceCropped = cropFace(detections, canvasCrop, HeadRotationAngle)
            drawImageOnCanvas(videoFaceCropped, videoCanvas)
            videoImageEmbedderResult = await imageEmbedder.embed(videoFaceCropped);
            // showEmbedding(videoImageEmbedderResult, embeddingVideoImage)
        }
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

function displayVideoDetections(detections, angle) {
    // Remove any highlighting from previous frame.
    for (let child of children) {
        liveView.removeChild(child);
    }
    children.splice(0);

    // Iterate through predictions and draw them to the live view
    for (let detection of detections) {
        const highlighter = document.createElement("div");
        highlighter.classList.add("highlighter");

        // Коррекция ширины бокса в зависимости от угла наклона лица
        const [adjustedWidth, adjustedHeight] = adjustBoundingBox(detection.boundingBox.width, detection.boundingBox.height, angle);

        // Вычисляем центр прямоугольника
        const centerX = video.offsetWidth - (detection.boundingBox.originX + detection.boundingBox.width / 2);
        const centerY = detection.boundingBox.originY + detection.boundingBox.height / 2;

        highlighter.style.left = `${centerX}px`;
        highlighter.style.top = `${centerY}px`;
        highlighter.style.width = `${adjustedWidth}px`;
        highlighter.style.height = `${adjustedHeight}px`;

        // Добавляем поворот вокруг центра
        highlighter.style.transform = `translate(-50%, -50%) rotate(${angle}deg)`;
        highlighter.style.transformOrigin = "center";

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

// Функция для вычисления угла наклона головы
function calculateHeadRotationAngle(detections) {
    
    const landmarks = detections[0].keypoints;
    const leftEye = landmarks[0]; // Левая глаз
    const rightEye = landmarks[1]; // Правый глаз

    // Вычисляем разницу по оси Y и X между глазами
    const deltaY = rightEye.y - leftEye.y;
    const deltaX = rightEye.x - leftEye.x;

    // Вычисляем угол наклона головы в радианах
    const angle = Math.atan2(deltaY, deltaX);

    return -angle * 180 / Math.PI;
}