import * as faceapi from 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.esm.js';

// --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ---
const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';

// Элементы DOM
const video = document.getElementById("webcam");
const liveView = document.getElementById("liveView");
const videoCanvas = document.getElementsByClassName("video_canvas")[0];
// Элементы для управления базой
const nameInput = document.getElementById("personName");
const saveBtn = document.getElementById("saveFaceBtn");
const statusMsg = document.getElementById("saveStatus");
const dbCountSpan = document.getElementById("dbCount");

// Переменные состояния
let isVideoPlaying = false;
let knownFaces = []; // Наша локальная база данных: [{id, name, descriptor, date}]
let faceMatcher = null; // Объект face-api для сравнения лиц
let currentPrimaryDescriptor = null; // Дескриптор лица, которое сейчас в фокусе (для сохранения)



// // Импорт
// import * as faceapi from 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.esm.js';

// // --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ---
// const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';

// // Элементы DOM
const demosSection = document.getElementById("demos");
const imageCanvas = document.getElementsByClassName("image_canvas")[0];
// const videoCanvas = document.getElementsByClassName("video_canvas")[0];
const embeddingImage = document.getElementById("embedImage");
const similarityElement = document.getElementById("similarity");
const headAngleElement = document.getElementById("headAngle");
const headRotationElement = document.getElementById("headRotation");
let enableWebcamButton = document.getElementById("webcamButton");

// // Переменные для хранения данных
let targetDescriptor = null; // Эмбеддинг целевого фото
let videoDescriptor = null;  // Эмбеддинг лица с видео
let isModelLoaded = false;
// let isVideoPlaying = false;

// Опции детектора (SSD Mobilenet V1 - баланс скорости и точности)
// minConfidence: 0.5 отсекает "шум"
const getDetectorOptions = () => new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 });


// --- 1. ЗАГРУЗКА И ИНИЦИАЛИЗАЦИЯ ---

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

// --- 2. МЕНЕДЖЕР БАЗЫ ДАННЫХ (LOCAL STORAGE) ---

// Загрузка данных
function loadFacesFromStorage() {
    const data = localStorage.getItem('face_db_resource');
    if (data) {
        const parsedData = JSON.parse(data);
        // Важно: JSON превращает Float32Array в обычный массив. 
        // Face-api требует Float32Array, конвертируем обратно.
        knownFaces = parsedData.map(item => ({
            ...item,
            descriptor: new Float32Array(Object.values(item.vector)) 
        }));
    } else {
        knownFaces = [];
    }
    updateFaceMatcher();
    updateUI();
}

// Сохранение данных
function saveFacesToStorage() {
    // Преобразуем descriptor (Float32Array) в обычный массив для JSON
    const dataToSave = knownFaces.map(item => ({
        id: item.id,
        name: item.name,
        vector: Array.from(item.descriptor), // Конвертация для JSON
        date: item.date
    }));
    
    localStorage.setItem('face_db_resource', JSON.stringify(dataToSave));
    updateFaceMatcher();
    updateUI();
}

// Обновление FaceMatcher (создание индекса для быстрого поиска)
function updateFaceMatcher() {
    if (knownFaces.length === 0) {
        faceMatcher = null;
        return;
    }

    // Создаем LabeledFaceDescriptors для face-api
    const labeledDescriptors = knownFaces.map(face => {
        return new faceapi.LabeledFaceDescriptors(face.name, [face.descriptor]);
    });

    // 0.6 - порог схожести (чем меньше, тем строже)
    faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);
}

// Обновление интерфейса (счетчик и дефолтное имя)
function updateUI() {
    if(dbCountSpan) dbCountSpan.innerText = knownFaces.length;
}

// Генерация следующего имени по умолчанию
function getNextDefaultName() {
    // Находим максимальный ID
    const maxId = knownFaces.reduce((max, p) => (p.id > max ? p.id : max), 0);
    const nextId = maxId + 1;
    return { name: `name${nextId}`, id: nextId };
}

// Очистка базы (для тестов, добавить в window чтобы вызывать из HTML)
window.clearDatabase = () => {
    localStorage.removeItem('face_db_resource');
    loadFacesFromStorage();
    alert("База очищена");
};


// --- 3. ОБРАБОТЧИК КНОПКИ "СОХРАНИТЬ" ---

saveBtn.addEventListener('click', () => {
    if (!currentPrimaryDescriptor) {
        statusMsg.innerText = "Нет лица для сохранения!";
        return;
    }

    const inputName = nameInput.value.trim();
    if (!inputName) {
        statusMsg.innerText = "Введите имя!";
        return;
    }

    // Проверяем, есть ли уже такой человек (по вектору)
    let bestMatchIndex = -1;
    let minDistance = 1.0;

    // Поиск самого похожего вектора в нашей базе
    knownFaces.forEach((face, index) => {
        const dist = faceapi.euclideanDistance(face.descriptor, currentPrimaryDescriptor);
        if (dist < minDistance) {
            minDistance = dist;
            bestMatchIndex = index;
        }
    });

    const THRESHOLD = 0.6; // Порог узнавания

    if (bestMatchIndex !== -1 && minDistance < THRESHOLD) {
        // --- ЧЕЛОВЕК УЖЕ ЕСТЬ: ОБНОВЛЯЕМ ---
        // Если имя в поле отличается от сохраненного, обновляем имя тоже, иначе оставляем старое
        knownFaces[bestMatchIndex].descriptor = currentPrimaryDescriptor;
        knownFaces[bestMatchIndex].date = new Date().toISOString();
        // Можно решить: перезаписывать имя всегда или только если совпадает
        knownFaces[bestMatchIndex].name = inputName; 
        
        statusMsg.innerText = `Обновлен: ${inputName} (Схожесть: ${(1-minDistance).toFixed(2)})`;
    } else {
        // --- НОВЫЙ ЧЕЛОВЕК ---
        const newIdData = getNextDefaultName();
        // Используем ID из генератора, если это совсем новый, но имя берем из инпута
        // (Инпут мог быть автозаполнен `nameX`, а пользователь мог его не менять)
        
        // Нам нужно определить ID. Если мы ввели уникальное имя вручную, ID все равно должен быть уникальным числом.
        const maxId = knownFaces.reduce((max, p) => (p.id > max ? p.id : max), 0);
        
        const newPerson = {
            id: maxId + 1,
            name: inputName,
            descriptor: currentPrimaryDescriptor,
            date: new Date().toISOString()
        };
        knownFaces.push(newPerson);
        statusMsg.innerText = `Сохранен новый: ${inputName}`;
    }

    saveFacesToStorage();
    
    // Очистка сообщения через 2 сек
    setTimeout(() => { statusMsg.innerText = ""; }, 2000);
});


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
    displayRotatedDetections(detections, undefined, parent, angles.roll);

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

        // 1. Получаем имена (results)
        const results = resizedDetections.map(d => {
            return faceMatcher ? faceMatcher.findBestMatch(d.descriptor) : { label: "Unknown", distance: 1 };
        });
        
        const face = resizedDetections[0];
        
        // Обратите внимание: дескриптор берем из первой детекции
        videoDescriptor = detections[0].descriptor; 

        currentPrimaryDescriptor = videoDescriptor;

        // 2. Углы
        const angles = calculateHeadRotationAngle(face.landmarks);
        showHeadRotation(headRotationElement, angles.yaw.toFixed(1));     
        showHeadAngle(headAngleElement, angles.roll.toFixed(1)); 

        // 3. Рисуем ПОВЕРНУТУЮ рамку на видео
        displayRotatedDetections(resizedDetections, results, liveView, angles.roll);

        // 4. Кропаем лицо с видео (Выпрямляем его)
        const croppedCanvas = cropRotatedFace(video, detections[0], -angles.roll);
        
        drawImageOnCanvas(croppedCanvas, videoCanvas);

        if (targetDescriptor) calculateAndShowSimilarity();
    } else {
        currentPrimaryDescriptor = null;
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

function displayRotatedDetections(detections, results, parentElement, angle) {
    // 1. Удаляем старые рамки
    const oldHighlighters = parentElement.querySelectorAll('.highlighter');
    oldHighlighters.forEach(el => el.remove());

    const media = parentElement.querySelector('video') || parentElement.querySelector('img');
    if (!media) return;

    let ratioX = 1;
    let ratioY = 1;

    if (media.tagName === 'VIDEO') {
        if (media.videoWidth > 0) {
            ratioX = media.offsetWidth / media.videoWidth;
            ratioY = media.offsetHeight / media.videoHeight;
        }
    } else {
        if (media.naturalWidth > 0) {
            ratioX = media.width / media.naturalWidth;
            ratioY = media.height / media.naturalHeight;
        }
    }

    // 4. Проходим по всем найденным лицам
    detections.forEach((det, i) => {
        const box = det.detection.box;
        
        // Получаем имя и дистанцию для текущего лица
        // Если results передали, берем i-й элемент, иначе ставим заглушку
        const match = results ? results[i] : { label: 'detecting...', distance: 0 };
        const labelText = match.label;
        const isUnknown = labelText === 'unknown';

        // Выбираем цвет: Красный если неизвестен, Зеленый если известен
        const boxColor = isUnknown ? 'red' : '#00ff00';

        const highlighter = document.createElement("div");
        highlighter.className = "highlighter";
                
        const width = box.width * ratioX;
        const height = box.height * ratioY;
        const x = box.x * ratioX;
        const y = box.y * ratioY;
        const cy = y + height / 2;
        
        let cx;
        if (media.tagName === 'VIDEO') {
            cx = media.offsetWidth - (x + width / 2);
        } else {
            cx = x + width / 2;
        }

        // Стили рамки
        highlighter.style.position = 'absolute';
        highlighter.style.width = `${width}px`;
        highlighter.style.height = `${height}px`;
        highlighter.style.left = `${cx}px`;
        highlighter.style.top = `${cy}px`;

        const angles = calculateHeadRotationAngle(det.landmarks);
        
        const rotation = (media.tagName === 'VIDEO') ? -angles.roll : angles.roll; 
        highlighter.style.transform = `translate(-50%, -50%) rotate(${rotation}deg)`;
        
        // Используем динамический цвет
        highlighter.style.border = `2px solid ${boxColor}`;
        highlighter.style.zIndex = '10';

        // Создаем плашку с именем и дистанцией
        const nameTag = document.createElement("div");
        nameTag.innerText = `${labelText} (${(match.distance ? (1 - match.distance).toFixed(2) : '')})`;
        
        // Стили текста
        nameTag.style.position = 'absolute';
        nameTag.style.bottom = '100%'; // Прилепить сверху рамки
        nameTag.style.left = '-2px';   // Выровнять по левому краю (с учетом бордера)
        nameTag.style.backgroundColor = boxColor;
        nameTag.style.color = 'white';
        nameTag.style.padding = '2px 5px';
        nameTag.style.fontSize = '14px';
        nameTag.style.fontWeight = 'bold';
        nameTag.style.whiteSpace = 'nowrap'; // Чтобы текст не переносился
        
        // Добавляем текст ВНУТРЬ рамки (чтобы он вращался вместе с ней)
        highlighter.appendChild(nameTag);

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