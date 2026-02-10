// Импорт
import * as faceapi from 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.esm.js';

// --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ---
const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';

// Элементы DOM
const demosSection = document.getElementById("demos");
const imageCanvas = document.getElementsByClassName("image_canvas")[0];
const videoCanvas = document.getElementsByClassName("video_canvas")[0];
const embeddingImage = document.getElementById("embedImage");
const similarityElement = document.getElementById("similarity");
const headAngleElement = document.getElementById("headAngle");
const headRotationElement = document.getElementById("headRotation");
const video = document.getElementById("webcam");
const liveView = document.getElementById("liveView");
let enableWebcamButton = document.getElementById("webcamButton");

// Переменные для хранения данных
let targetDescriptor = null; // Эмбеддинг целевого фото
let videoDescriptor = null;  // Эмбеддинг лица с видео
let isModelLoaded = false;
let isVideoPlaying = false;

// Опции детектора (SSD Mobilenet V1 - баланс скорости и точности)
// minConfidence: 0.5 отсекает "шум"
const getDetectorOptions = () => new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 });


// --- ИНИЦИАЛИЗАЦИЯ ---
const initializeFaceModels = async () => {
    console.log("⏳ Загрузка моделей нейросети...");
    try {
        // Загружаем необходимые модели
        await Promise.all([
            faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),    // Детектор
            faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL), // 68 точек лица (для углов)
            faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL) // Распознавание (FaceNet)
        ]);

        console.log("✅ Модели загружены!");
        isModelLoaded = true;
        
        if (demosSection) demosSection.classList.remove("invisible");

    } catch (error) {
        console.error("❌ Ошибка при загрузке моделей:", error);
    }
};

initializeFaceModels();


// --- ОБРАБОТКА КЛИКА ПО КАРТИНКЕ (TARGET IMAGE) ---
const imageContainers = document.getElementsByClassName("detectOnClick");

for (let imageContainer of imageContainers) {
    imageContainer.children[0].addEventListener("click", handleClick);
}

// --- CLICK HANDLER (ФОТО) ---
async function handleClick(event) {
    if (!isModelLoaded) return;

    const imgElement = event.target;
    const parent = imgElement.parentNode;

    // 1. Детекция
    const detections = await faceapi
        .detectAllFaces(imgElement, getDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptors();

    if (!detections.length) return;

    const face = detections[0];
    targetDescriptor = face.descriptor;

    // 2. Считаем угол
    const angles = calculateHeadRotationAngle(face.landmarks);
    showHeadRotation(headRotationElement, angles.yaw.toFixed(1));     
    showHeadAngle(headAngleElement, angles.roll.toFixed(1)); 

    // 3. Рисуем ПОВЕРНУТУЮ рамку
    // Передаем угол Roll
    displayRotatedDetections([face], parent, angles.roll);

    // 4. Делаем "Умный кроп" (вырезаем выпрямленное лицо)
    // Передаем исходную картинку, данные детекции и угол, на который надо "откатить" наклон
    const croppedCanvas = cropRotatedFace(imgElement, face, -angles.roll); // Минус, чтобы компенсировать наклон
    
    // Рисуем полученный кроп в маленький канвас
    drawImageOnCanvas(croppedCanvas, imageCanvas);

    // Показываем эмбеддинг
    showEmbedding(targetDescriptor, embeddingImage);
    
    if (videoDescriptor) calculateAndShowSimilarity();
}

// --- WEBCAM LOOP (ВИДЕО) ---
async function predictWebcam() {
    if (!isVideoPlaying) return;

    // 1. Детекция
    const detections = await faceapi
        .detectAllFaces(video, getDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptors();

    if (detections.length > 0) {
        // Ресайзим результаты под размер видео на экране, чтобы рамка не улетала
        const displaySize = { width: video.offsetWidth, height: video.offsetHeight };
        const resizedDetections = faceapi.resizeResults(detections, displaySize);
        
        const face = resizedDetections[0];
        
        // Обратите внимание: дескриптор берем из ОРИГИНАЛЬНОЙ (не ресайзнутой) детекции для точности,
        // но для простоты здесь возьмем из первой (face-api обычно справляется).
        // Лучше брать дескриптор от `detections[0]`, а координаты для рамки от `resizedDetections[0]`.
        videoDescriptor = detections[0].descriptor; 

        // 2. Углы
        const angles = calculateHeadRotationAngle(face.landmarks);
        showHeadRotation(headRotationElement, angles.yaw.toFixed(1));     
        showHeadAngle(headAngleElement, angles.roll.toFixed(1)); 

        // 3. Рисуем ПОВЕРНУТУЮ рамку на видео
        displayRotatedDetections([face], liveView, angles.roll);

        // 4. Кропаем лицо с видео (Выпрямляем его)
        // Важно: кропаем с самого видео-элемента
        // Используем 'detections[0]' (реальные координаты видео), а не 'face' (экранные координаты),
        // так как cropRotatedFace работает с sourceImage (video) напрямую.
        const croppedCanvas = cropRotatedFace(video, detections[0], -angles.roll);
        
        drawImageOnCanvas(croppedCanvas, videoCanvas);

        if (targetDescriptor) calculateAndShowSimilarity();
    } else {
        // Если лиц нет, убираем рамки
        const old = liveView.querySelectorAll('.highlighter');
        old.forEach(el => el.remove());
    }

    window.requestAnimationFrame(predictWebcam);
}

const hasGetUserMedia = () => !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

if (hasGetUserMedia()) {
    enableWebcamButton.addEventListener("click", enableCam);
} else {
    console.warn("getUserMedia() не поддерживается вашим браузером");
}

async function enableCam(event) {
    if (!isModelLoaded) {
        alert("Подождите, модели еще грузятся...");
        return;
    }

    enableWebcamButton.classList.add("removed");

    const constraints = { video: true };

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
        isVideoPlaying = true;
    } catch (err) {
        console.error("Ошибка доступа к камере:", err);
    }
}

function calculateAndShowSimilarity() {
    if (!targetDescriptor || !videoDescriptor) return;

    // Евклидово расстояние
    const distance = faceapi.euclideanDistance(targetDescriptor, videoDescriptor);
    
    // Перевод в % схожести (примерная формула)
    // 0 distance = 100% similarity
    // > 0.6 distance = 0% similarity (другой человек)
    const similarityVal = Math.max(0, 1 - distance);
    
    similarityElement.innerText = `Similarity: ${similarityVal.toFixed(2)}`;
}

// Расчет поворота головы на основе 68 точек face-api

function calculateHeadRotationAngle(landmarks) {
    const points = landmarks.positions; 
    const leftEye = points[36];  // Внешний уголок левого глаза
    const rightEye = points[45]; // Внешний уголок правого глаза
    const nose = points[30];     // Кончик носа

    // --- 1. ROLL (Наклон) ---
    const deltaY = rightEye.y - leftEye.y;
    const deltaX = rightEye.x - leftEye.x;
    const angleRad = Math.atan2(deltaY, deltaX);
    const rollAngle = angleRad * (180 / Math.PI);

    // --- 2. YAW (Поворот) ---
    const eyeDist = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
    const eyesCenter = { x: (leftEye.x + rightEye.x) / 2, y: (leftEye.y + rightEye.y) / 2 };
    const noseOffset = nose.x - eyesCenter.x;
    const yawAngle = (noseOffset / eyeDist) * 100; // Эмпирический коэффициент

    return {
        roll: rollAngle,
        yaw: yawAngle,
        eyesCenter: eyesCenter // Возвращаем центр глаз для точного кропа
    };
}


// Функция кропа с поворотом
function cropRotatedFace(sourceImage, result, angle) {
    const box = result.detection ? result.detection.box : result;
    
    // Если вдруг box не найден, то выходим
    if (!box || !box.width) { 
        console.error("Не удалось найти координаты лица для кропа", result); 
        return document.createElement("canvas"); 
    }

    // Центр лица (вокруг которого будем вращать)
    const centerX = box.x + box.width / 2;
    const centerY = box.y + box.height / 2;

    // Немного расширяем область (1.5 - чтобы захватить всю голову при повороте)
    const width = box.width * 1.5; 
    const height = box.height * 1.5;

    // Создаем временный канвас
    const offscreenCanvas = document.createElement("canvas");
    const ctx = offscreenCanvas.getContext("2d");

    // Размер канваса должен быть достаточным, чтобы вместить повернутое изображение
    // Берем диагональ (hypot) для гарантии
    const diag = Math.hypot(width, height);
    offscreenCanvas.width = diag;
    offscreenCanvas.height = diag;

    // Переносим начало координат в центр канваса
    ctx.translate(diag / 2, diag / 2);
    
    // Поворачиваем пространство (angle в градусах -> радианы)
    ctx.rotate((angle * Math.PI) / 180);

    // Рисуем исходное изображение со смещением
    // (Сдвигаем так, чтобы центр лица совпал с центром канваса)
    ctx.drawImage(sourceImage, -centerX, -centerY);

    // Теперь вырезаем итоговый вертикальный прямоугольник из центра
    const finalCanvas = document.createElement("canvas");
    finalCanvas.width = box.width;  // Возвращаем к оригинальным размерам лица
    finalCanvas.height = box.height;
    const finalCtx = finalCanvas.getContext("2d");

    finalCtx.drawImage(
        offscreenCanvas,
        (diag - box.width) / 2,  
        (diag - box.height) / 2, 
        box.width, 
        box.height,       
        0, 0,                
        box.width, 
        box.height        
    );

    return finalCanvas;
}

function displayRotatedDetections(detections, parentElement, angle) {
    // 1. Удаляем старые рамки
    const oldHighlighters = parentElement.querySelectorAll('.highlighter');
    oldHighlighters.forEach(el => el.remove());

    // 2. Ищем элемент медиа (видео или картинку) внутри контейнера
    const media = parentElement.querySelector('video') || parentElement.querySelector('img');
    if (!media) return; // Если нет медиа, выходим

    // 3. Вычисление масштаба (ratio)
    let ratioX = 1;
    let ratioY = 1;

    if (media.tagName === 'VIDEO') {
        // Для видео: ширина на экране / реальная ширина потока
        // Проверка на 0, чтобы не делить на ноль, если видео еще не загрузилось
        if (media.videoWidth > 0) {
            ratioX = media.offsetWidth / media.videoWidth;
            ratioY = media.offsetHeight / media.videoHeight;
        }
    } else {
        // Для картинки: ширина на экране / натуральная ширина файла
        if (media.naturalWidth > 0) {
            ratioX = media.width / media.naturalWidth;
            ratioY = media.height / media.naturalHeight;
        }
    }

    // 4. Проходим по всем найденным лицам
    detections.forEach(det => {
        const box = det.detection.box;
        
        // Создаем div для рамки
        const highlighter = document.createElement("div");
        highlighter.className = "highlighter";
                
        // А. Приводим размеры бокса к размеру экрана
        const width = box.width * ratioX;
        const height = box.height * ratioY;
        
        // Б. Приводим координаты (левый верхний угол) к размеру экрана
        const x = box.x * ratioX;
        const y = box.y * ratioY;

        // В. Расчет ЦЕНТРА (cx, cy)
        // cy (по вертикали) всегда одинаковый
        const cy = y + height / 2;
        
        let cx;
        if (media.tagName === 'VIDEO') {
            // ДЛЯ ВИДЕО (Зеркальный режим):
            // Формула: Ширина_Контейнера - (Координата_X + Половина_Ширины)
            // Мы "отступаем" от правого края, а не от левого
            cx = media.offsetWidth - (x + width / 2);
        } else {
            // ДЛЯ ФОТО (Обычный режим):
            cx = x + width / 2;
        }

        // --- СТИЛИ ---

        highlighter.style.position = 'absolute';
        highlighter.style.width = `${width}px`;
        highlighter.style.height = `${height}px`;
        
        // Позиционируем точку в вычисленный центр
        highlighter.style.left = `${cx}px`;
        highlighter.style.top = `${cy}px`;
        
        // Сдвигаем div на -50% от его собственного размера, чтобы центр div совпал с точкой (cx, cy)
        // И поворачиваем на угол
        // Для видео угол часто нужно инвертировать (angle или -angle)
        const rotation = (media.tagName === 'VIDEO') ? -angle : angle; 
        highlighter.style.transform = `translate(-50%, -50%) rotate(${rotation}deg)`;
        
        // Оформление рамки
        highlighter.style.border = '2px solid #00ff00';
        highlighter.style.zIndex = '10'; // Чтобы было поверх видео

        // Добавляем на страницу
        parentElement.appendChild(highlighter);
    });
}

function drawImageOnCanvas(image, canvas) {
    const ctx = canvas.getContext("2d");
    const imgWidth = image.width;
    const imgHeight = image.height;
    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;

    const scale = Math.min(canvasWidth / imgWidth, canvasHeight / imgHeight);
    const newWidth = imgWidth * scale;
    const newHeight = imgHeight * scale;
    const x = (canvasWidth - newWidth) / 2;
    const y = (canvasHeight - newHeight) / 2;

    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    ctx.drawImage(image, x, y, newWidth, newHeight);
}

function showEmbedding(descriptor, element) {
    if (!descriptor) return;
    // Descriptor - это Float32Array(128)
    const sliced = descriptor.slice(0, 5); // берем первые 5 чисел для показа
    element.innerText = `Float Embedding: [${sliced.map(n => n.toFixed(3)).join(', ')}...]`;
}

function showHeadAngle(element, val){
    element.innerText = `Head Roll: ${val}°`;
}

function showHeadRotation(element, val){
    element.innerText = `Head Yaw: ${val}°`;
}