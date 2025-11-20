// Audio utils mejorados

// Follow audio utils are referred from magenta-js 
// https://github.com/magenta/magenta-js/blob/master/music/src/core/audio_utils.ts

const SAMPLE_RATE = 16000
const isSafari = window.webkitOfflineAudioContext;
const offlineCtx = new OfflineAudioContext(1, SAMPLE_RATE, SAMPLE_RATE);

async function loadAudioFromUrl(url) {
    return fetch(url)
        .then((body) => body.arrayBuffer())
        .then((buffer) => offlineCtx.decodeAudioData(buffer));
}

function getMonoAudio(audioBuffer) {
    if (audioBuffer.numberOfChannels === 1) {
        return audioBuffer.getChannelData(0);
    }
    if (audioBuffer.numberOfChannels !== 2) {
        throw Error(
            `${audioBuffer.numberOfChannels} channel audio is not supported.`);
    }
    const ch0 = audioBuffer.getChannelData(0);
    const ch1 = audioBuffer.getChannelData(1);

    const mono = new Float32Array(audioBuffer.length);
    for (let i = 0; i < audioBuffer.length; ++i) {
        mono[i] = (ch0[i] + ch1[i]) / 2;
    }
    return mono;
}

async function resampleAndMakeMono(audioBuffer, targetSr = SAMPLE_RATE) {
    if (audioBuffer.sampleRate === targetSr) {
        return getMonoAudio(audioBuffer);
    }
    const sourceSr = audioBuffer.sampleRate;
    const lengthRes = (audioBuffer.length * targetSr) / sourceSr;
    if (!isSafari) {
        const _offlineCtx = new OfflineAudioContext(
            audioBuffer.numberOfChannels, audioBuffer.duration * targetSr,
            targetSr);
        const bufferSource = _offlineCtx.createBufferSource();
        bufferSource.buffer = audioBuffer;
        bufferSource.connect(_offlineCtx.destination);
        bufferSource.start();
        return _offlineCtx.startRendering().then(
            (buffer) => buffer.getChannelData(0));
    } else {
        // Safari does not support resampling with WebAudio.
        console.log(
            'Safari does not support WebAudio resampling, so this may be slow.',
            'O&F', logging.Level.WARN);

        const originalAudio = getMonoAudio(audioBuffer);
        const resampledAudio = new Float32Array(lengthRes);
        resample(
            ndarray(resampledAudio, [lengthRes]),
            ndarray(originalAudio, [originalAudio.length]));
        return resampledAudio;
    }
}

function applyWindow(buffer, win) {
    if (buffer.length !== win.length) {
        console.error(
            `Buffer length ${buffer.length} != window length ${win.length}.`);
        return null;
    }
    const out = new Float32Array(buffer.length);
    for (let i = 0; i < buffer.length; i++) {
        out[i] = win[i] * buffer[i];
    }
    return out;
}

/**
 * Given a timeseries, returns an array of timeseries that are windowed
 * according to the params specified.
 */
function frame(data, frameLength, hopLength) {
    const bufferCount = Math.floor((data.length - frameLength) / hopLength) + 1;
    const buffers = Array.from(
        { length: bufferCount }, (x, i) => new Float32Array(frameLength));
    for (let i = 0; i < bufferCount; i++) {
        const ind = i * hopLength;
        const buffer = data.slice(ind, ind + frameLength);
        buffers[i].set(buffer);
        // In the end, we will likely have an incomplete buffer, which we should
        // just ignore.
        if (buffer.length !== frameLength) {
            continue;
        }
    }
    return buffers;
}

function padReflect(data, padding) {
    const out = padConstant(data, padding);
    for (let i = 0; i < padding; i++) {
        // Pad the beginning with reflected values.
        out[i] = out[2 * padding - i];
        // Pad the end with reflected values.
        out[out.length - i - 1] = out[out.length - 2 * padding + i - 1];
    }
    return out;
}

function padConstant(data, padding) {
    let padLeft, padRight;
    if (typeof padding === 'object') {
        [padLeft, padRight] = padding;
    } else {
        padLeft = padRight = padding;
    }
    const out = new Float32Array(data.length + padLeft + padRight);
    out.set(data, padLeft);
    return out;
}

function padCenterToLength(data, length) {
    // If data is longer than length, error!
    if (data.length > length) {
        throw new Error('Data is longer than length.');
    }
    const paddingLeft = Math.floor((length - data.length) / 2);
    const paddingRight = length - data.length - paddingLeft;
    return padConstant(data, [paddingLeft, paddingRight]);
}

function hannWindow(length) {
    const win = new Float32Array(length);
    for (let i = 0; i < length; i++) {
        win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (length - 1)));
    }
    return win;
}

function hzToMel(hz) {
    return 1125.0 * Math.log(1 + hz / 700.0);
}

function melToHz(mel) {
    return 700.0 * (Math.exp(mel / 1125.0) - 1);
}

function fft(y) {
    const fft = new FFTJS(y.length);
    const out = fft.createComplexArray();
    const data = fft.toComplexArray(y);
    fft.transform(out, data);
    return out;
}

function stft(y, params) {
    const nFft = params.nFft || 2048;
    const winLength = params.winLength || nFft;
    const hopLength = params.hopLength || Math.floor(winLength / 4);

    let fftWindow = hannWindow(winLength);

    // Pad the window to be the size of nFft.
    fftWindow = padCenterToLength(fftWindow, nFft);

    // Pad the time series so that the frames are centered.
    y = padReflect(y, Math.floor(nFft / 2));

    // Window the time series.
    const yFrames = frame(y, nFft, hopLength);
    // Pre-allocate the STFT matrix.
    const stftMatrix = [];

    const width = yFrames.length;
    const height = nFft + 2;
    for (let i = 0; i < width; i++) {
        // Each column is a Float32Array of size height.
        const col = new Float32Array(height);
        stftMatrix[i] = col;
    }

    for (let i = 0; i < width; i++) {
        // Populate the STFT matrix.
        const winBuffer = applyWindow(yFrames[i], fftWindow);
        const col = fft(winBuffer);
        stftMatrix[i].set(col.slice(0, height));
    }

    return stftMatrix;
}

/**
 * Given an interlaced complex array (y_i is real, y_(i+1) is imaginary),
 * calculates the energies. Output is half the size.
 */
function mag(y) {
    const out = new Float32Array(y.length / 2);
    for (let i = 0; i < y.length / 2; i++) {
        out[i] = Math.sqrt(y[i * 2] * y[i * 2] + y[i * 2 + 1] * y[i * 2 + 1]);
    }
    return out;
}

function pow(arr, power) {
    return arr.map((v) => Math.pow(v, power));
}

function magSpectrogram(stft, power) {
    const spec = stft.map((fft) => pow(mag(fft), power));
    const nFft = stft[0].length - 1;
    return [spec, nFft];
}

function linearSpace(start, end, count) {
    // Include start and endpoints.
    const delta = (end - start) / (count - 1);
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) {
        out[i] = start + delta * i;
    }
    return out;
}

function calculateFftFreqs(sampleRate, nFft) {
    return linearSpace(0, sampleRate / 2, Math.floor(1 + nFft / 2));
}

function calculateMelFreqs(nMels, fMin, fMax) {
    const melMin = hzToMel(fMin);
    const melMax = hzToMel(fMax);

    // Construct linearly spaced array of nMel intervals, between melMin and
    // melMax.
    const mels = linearSpace(melMin, melMax, nMels);
    const hzs = mels.map((mel) => melToHz(mel));
    return hzs;
}

function internalDiff(arr) {
    const out = new Float32Array(arr.length - 1);
    for (let i = 0; i < arr.length - 1; i++) {
        out[i] = arr[i + 1] - arr[i];
    }
    return out;
}

function outerSubtract(arr, arr2) {
    const out = [];
    for (let i = 0; i < arr.length; i++) {
        out[i] = new Float32Array(arr2.length);
    }
    for (let i = 0; i < arr.length; i++) {
        for (let j = 0; j < arr2.length; j++) {
            out[i][j] = arr[i] - arr2[j];
        }
    }
    return out;
}

function applyFilterbank(mags, filterbank) {
    if (mags.length !== filterbank[0].length) {
        throw new Error(
            `Each entry in filterbank should have dimensions ` +
            `matching FFT. |mags| = ${mags.length}, ` +
            `|filterbank[0]| = ${filterbank[0].length}.`);
    }

    // Apply each filter to the whole FFT signal to get one value.
    const out = new Float32Array(filterbank.length);
    for (let i = 0; i < filterbank.length; i++) {
        // To calculate filterbank energies we multiply each filterbank with the
        // power spectrum.
        const win = applyWindow(mags, filterbank[i]);
        // Then add up the coefficents.
        out[i] = win.reduce((a, b) => a + b);
    }
    return out;
}

function applyWholeFilterbank(spec, filterbank) {
    // Apply a point-wise dot product between the array of arrays.
    const out = [];
    for (let i = 0; i < spec.length; i++) {
        out[i] = applyFilterbank(spec[i], filterbank);
    }
    return out;
}

function createMelFilterbank(params) {
    const fMin = params.fMin || 0;
    const fMax = params.fMax || params.sampleRate / 2;
    const nMels = params.nMels || 128;
    const nFft = params.nFft || 2048;

    // Center freqs of each FFT band.
    const fftFreqs = calculateFftFreqs(params.sampleRate, nFft);
    // (Pseudo) center freqs of each Mel band.
    const melFreqs = calculateMelFreqs(nMels + 2, fMin, fMax);

    const melDiff = internalDiff(melFreqs);
    const ramps = outerSubtract(melFreqs, fftFreqs);
    const filterSize = ramps[0].length;

    const weights = [];
    for (let i = 0; i < nMels; i++) {
        weights[i] = new Float32Array(filterSize);
        for (let j = 0; j < ramps[i].length; j++) {
            const lower = -ramps[i][j] / melDiff[i];
            const upper = ramps[i + 2][j] / melDiff[i + 1];
            const weight = Math.max(0, Math.min(lower, upper));
            weights[i][j] = weight;
        }
    }

    // Slaney-style mel is scaled to be approx constant energy per channel.
    for (let i = 0; i < weights.length; i++) {
        // How much energy per channel.
        const enorm = 2.0 / (melFreqs[2 + i] - melFreqs[i]);
        // Normalize by that amount.
        weights[i] = weights[i].map((val) => val * enorm);
    }

    return weights;
}

// Función melSpectrogram mejorada para coincidir con PyTorch
function melSpectrogram(y, params) {
    if (!params.power) {
        params.power = 2.0;
    }
    const stftMatrix = stft(y, params);
    const [spec, nFft] = magSpectrogram(stftMatrix, params.power);

    params.nFft = nFft;
    const melBasis = createMelFilterbank(params);
    const melSpec = applyWholeFilterbank(spec, melBasis);

    // Aplicar log_offset y log transform para coincidir con PyTorch
    const logMelSpec = melSpec.map(frame =>
        frame.map(value => Math.log(Math.max(value, log_offset)))
    );

    return logMelSpec;
}

// ---------- NORMALIZATION IMPROVEMENTS -----------

// Implementación de PerUtteranceNorm para JavaScript (equivalente a PyTorch)
function perUtteranceNormalization(melSpectrogram, eps = 1e-8) {
    const numFrames = melSpectrogram.length;
    const numMels = melSpectrogram[0].length;

    // Calcular media sobre toda la utterance
    let sum = 0;
    let count = 0;

    for (let i = 0; i < numFrames; i++) {
        for (let j = 0; j < numMels; j++) {
            sum += melSpectrogram[i][j];
            count++;
        }
    }
    const mean = sum / count;

    // Calcular desviación estándar
    let sumSquaredDiff = 0;
    for (let i = 0; i < numFrames; i++) {
        for (let j = 0; j < numMels; j++) {
            const diff = melSpectrogram[i][j] - mean;
            sumSquaredDiff += diff * diff;
        }
    }
    const std = Math.sqrt(sumSquaredDiff / count) + eps;

    // Normalizar
    const normalized = [];
    for (let i = 0; i < numFrames; i++) {
        normalized[i] = new Float32Array(numMels);
        for (let j = 0; j < numMels; j++) {
            normalized[i][j] = (melSpectrogram[i][j] - mean) / std;
        }
    }

    return normalized;
}

// Clase para detector de palabras clave con suavizado
class KeywordDetectorJS {
    constructor(wakeWords, confidenceThreshold = 0.7, smoothingWindow = 3) {
        this.wakeWords = wakeWords;
        this.confidenceThreshold = confidenceThreshold;
        this.smoothingWindow = smoothingWindow;
        this.predictionBuffer = [];
        this.targetState = 0;
        this.inferenceTrack = [];
    }

    reset() {
        this.targetState = 0;
        this.inferenceTrack = [];
        this.predictionBuffer = [];
    }

    smoothPredictions(currentPred, confidence) {
        this.predictionBuffer.push({ pred: currentPred, conf: confidence });

        // Mantener solo predicciones recientes
        if (this.predictionBuffer.length > this.smoothingWindow) {
            this.predictionBuffer.shift();
        }

        // Obtener predicción más común con alta confianza
        if (this.predictionBuffer.length >= this.smoothingWindow) {
            const highConfPreds = this.predictionBuffer
                .filter(item => item.conf > this.confidenceThreshold)
                .map(item => item.pred);

            if (highConfPreds.length >= Math.floor(this.smoothingWindow / 2)) {
                // Retornar predicción más común
                const counts = {};
                highConfPreds.forEach(pred => {
                    counts[pred] = (counts[pred] || 0) + 1;
                });

                return Object.keys(counts).reduce((a, b) =>
                    counts[a] > counts[b] ? a : b
                );
            }
        }

        return null;
    }

    processPrediction(predWord, confidence, verbose = false) {
        const smoothedPred = this.smoothPredictions(predWord, confidence);

        if (smoothedPred === null) {
            return { complete: false, sequence: null };
        }

        if (verbose) {
            console.log(`Detected: ${smoothedPred} (conf: ${confidence.toFixed(3)})`);
        }

        const expectedLabel = this.wakeWords[this.targetState];

        if (smoothedPred === expectedLabel) {
            this.targetState++;
            this.inferenceTrack.push(smoothedPred);

            if (verbose) {
                console.log(`Progress: ${this.inferenceTrack.join(' -> ')}`);
            }

            // Verificar si la secuencia está completa
            if (this.inferenceTrack.length === this.wakeWords.length) {
                const completedSequence = [...this.inferenceTrack];
                this.reset();
                return { complete: true, sequence: completedSequence };
            }
        } else {
            // Resetear si se detecta palabra incorrecta
            if (smoothedPred !== "oov" && this.wakeWords.includes(smoothedPred)) {
                if (smoothedPred === this.wakeWords[0]) {
                    this.targetState = 1;
                    this.inferenceTrack = [smoothedPred];
                    if (verbose) {
                        console.log(`Restarted sequence: ${this.inferenceTrack.join(' -> ')}`);
                    }
                } else {
                    this.reset();
                    if (verbose) {
                        console.log("Sequence reset due to unexpected word");
                    }
                }
            }
        }

        return { complete: false, sequence: null };
    }
}

// ---------- CONFIG ---------------

// Trained wake words
// 3 wake words for print
const wakeWordsEmojis = ["🐚 Almeja", "🧠 Cerebro", "🥔 Patata"];

// URL of web model json file
const modelPath = 'web_model/model.json';
// URL of model_data.json file
const modelDataPath = 'model_data.json';

var minimumProbabilityScore = 0.95;

var myvad;
var vad_speaking = false;

// ---------- CONFIG END ------------

// 3 wake words and "oov" (Out Of Vocabulary)
var classes = ["almeja", "cerebro", "patata", "oov"];
// 3 wake words
var wakeWords = ["almeja", "cerebro", "patata"];

window.AudioContext = window.AudioContext || window.webkitAudioContext;

var audioContext;
var audioInput = null,
    realAudioInput = null,
    inputPoint = null,
    recording = false;
var rafID = null;
var analyserContext = null;
var canvasWidth, canvasHeight;

var bufferSize = 1024;
var channels = 1;
var windowSize = 750;
var zmuv_mean = 0.000016;
var zmuv_std = 0.072771;
var log_offset = 1e-7;
var SPEC_HOP_LENGTH = 200;
var MEL_SPEC_BINS = 40;
var NUM_FFTS = 512;
var sampleRate = 16000;
var numOfBatches = 2;

let predictWords = [];
let arrayBuffer = [];
let targetState = 0;

let bufferMap = {};

var windowBufferSize = windowSize / 1000 * sampleRate;

let tfModel;
async function loadModel() {
    tfModel = await tf.loadGraphModel(modelPath);
}

async function loadModelComplete() {
    fetch(modelDataPath)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Network error: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            windowSize = data['window_size'];
            zmuv_mean = data['zmuv_mean'];
            zmuv_std = data['zmuv_std'];
            log_offset = data['log_offset'];
            SPEC_HOP_LENGTH = data['hop_length'];
            MEL_SPEC_BINS = data['num_mels'];
            NUM_FFTS = data['num_fft'];
            sampleRate = data['sample_rate'];
            windowBufferSize = windowSize / 1000 * sampleRate;
            classes = data['classes'];
            wakeWords = data['classes_base'];
            loadModel();
        })
        .catch(error => {
            console.error('Error getting JSON file:', error);
        });
}

loadModelComplete();

function argMax(array) {
    return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

// Softmax mejorado para mayor estabilidad numérica
function softmax(arr) {
    // Encontrar el máximo para estabilidad numérica
    const maxVal = Math.max(...arr);

    // Calcular exponenciales con offset
    const expVals = arr.map(val => Math.exp(val - maxVal));

    // Calcular suma
    const sumExp = expVals.reduce((sum, val) => sum + val, 0);

    // Retornar probabilidades normalizadas
    return expVals.map(val => val / sumExp);
}

// Función de flatten mejorada para coincidir con PyTorch
function improvedFlatten(log_mels) {
    // Primero aplicar normalización per-utterance (equivalente a PyTorch)
    const perUtteranceNormed = perUtteranceNormalization(log_mels);

    // Luego aplicar ZMUV global
    const flatten_array = [];
    for (let i = 0; i < MEL_SPEC_BINS; i++) {
        for (let j = 0; j < perUtteranceNormed.length; j++) {
            const normalized = (perUtteranceNormed[j][i] - zmuv_mean) / zmuv_std;
            flatten_array.push(normalized);
        }
    }
    return flatten_array;
}

function convertToMono(input) {
    var splitter = audioContext.createChannelSplitter(2);
    var merger = audioContext.createChannelMerger(2);

    input.connect(splitter);
    splitter.connect(merger, 0, 0);
    splitter.connect(merger, 0, 1);
    return merger;
}

function initAudio() {
    if (!navigator.getUserMedia)
        navigator.getUserMedia = navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
    if (!navigator.cancelAnimationFrame)
        navigator.cancelAnimationFrame = navigator.webkitCancelAnimationFrame || navigator.mozCancelAnimationFrame;
    if (!navigator.requestAnimationFrame)
        navigator.requestAnimationFrame = navigator.webkitRequestAnimationFrame || navigator.mozRequestAnimationFrame;

    // Chrome is suspending the audio context on load
    if (audioContext.state == "suspended") {
        audioContext.resume()
    }

    constraints = { audio: true }
    navigator.mediaDevices.getUserMedia(constraints)
        .then(gotStream)
        .catch(function (err) {
            alert('Error getting audio');
            console.log(err);
        });
}

// Inicializar el detector mejorado
let keywordDetector;

window.addEventListener('load', async function () {
    // Inicializar detector con parámetros ajustados
    keywordDetector = new KeywordDetectorJS(wakeWords, 0.6, 2);

    myvad = await vad.MicVAD.new({
        onSpeechStart: () => {
            console.log("Speech start detected");
            vad_speaking = true;
        },
        onSpeechEnd: async (audio) => {
            vad_speaking = false;
            console.log("Speech end detected");

            // Procesar el audio capturado
            arrayBuffer = Array.from(audio); // Asumimos que audio es Float32Array a 16000 Hz mono
            if (arrayBuffer.length < windowBufferSize) {
                console.log("Audio segment too short, skipping");
                return;
            }

            let batchBuffers = [];
            let batchMels = [];
            let dataProcessed = [];
            let batch = 0;

            for (let i = 0; i < arrayBuffer.length; i += windowBufferSize) {
                let batchBuffer = arrayBuffer.slice(i, i + windowBufferSize);
                if (batchBuffer.length < windowBufferSize) {
                    // Descartar último chunk incompleto
                    break;
                }

                // Calcular mel spectrogram mejorado
                let log_mels = melSpectrogram(batchBuffer, {
                    sampleRate: sampleRate,
                    hopLength: SPEC_HOP_LENGTH,
                    nMels: MEL_SPEC_BINS,
                    nFft: NUM_FFTS,
                    power: 2.0
                });

                batchBuffers.push(batchBuffer);
                batchMels.push(log_mels);

                // Usar flatten mejorado con ambas normalizaciones
                const processedData = improvedFlatten(log_mels);
                dataProcessed = [...dataProcessed, ...processedData];
                batch++;
            }

            if (batch === 0) {
                console.log("No valid chunks to process");
                return;
            }

            // Ejecutar el modelo con inputs de Tensor y obtener resultado
            let outputTensor = tf.tidy(() => {
                let inputTensor = tf.tensor(dataProcessed, [batch, 1, MEL_SPEC_BINS, dataProcessed.length / (batch * MEL_SPEC_BINS)], 'float32');
                let outputTensor = tfModel.predict(inputTensor);
                return outputTensor;
            });

            let outputData = await outputTensor.data();

            for (let i = 0; i < outputData.length; i += classes.length) {
                let scores = Array.from(outputData.slice(i, i + classes.length));
                console.log("raw scores", scores);

                // Aplicar softmax mejorado
                let probs = softmax(scores);
                let class_idx = argMax(probs);
                let probs_keywords = probs.slice(0, -1);
                let most_prob_keyword = argMax(probs_keywords);

                console.log("probabilities", probs);
                console.log("> predicted word:", classes[class_idx]);
                console.log("> most probable word:", classes[most_prob_keyword], probs_keywords[most_prob_keyword]);

                // Usar el detector mejorado
                const result = keywordDetector.processPrediction(
                    classes[class_idx],
                    probs[class_idx],
                    true // verbose
                );

                if (result.complete) {
                    console.log(`🎉 WAKE WORD SEQUENCE DETECTED: ${result.sequence.join(' -> ')}`);

                    // Actualizar UI con éxito
                    let successItem = document.createElement('p');
                    const ahora = new Date();
                    const currentTime = String(ahora.getHours()).padStart(2, '0') + ':' +
                        String(ahora.getMinutes()).padStart(2, '0') + ':' +
                        String(ahora.getSeconds()).padStart(2, '0');

                    successItem.innerText = `✅ SEQUENCE COMPLETE: ${result.sequence.join(' -> ')} at ${currentTime}`;
                    successItem.style.color = 'green';
                    successItem.style.fontWeight = 'bold';
                    successItem.style.fontSize = '18px';
                    document.getElementById('predictions_emojis').appendChild(successItem);

                    document.getElementById('prediction_emoji').innerText = '🎉 Success!';

                    // Pausa para evitar re-detección inmediata
                    setTimeout(() => {
                        document.getElementById('prediction_emoji').innerText = 'Listening...';
                    }, 2000);
                }

                // Mostrar predicción individual solo si es una wake word
                if (wakeWordsEmojis[class_idx] !== undefined && probs.length === classes.length && probs[class_idx] !== 1) {
                    let emojiItem = document.createElement('p');
                    const ahora = new Date();
                    var currentTime = String(ahora.getHours()).padStart(2, '0') + ':' + String(ahora.getMinutes()).padStart(2, '0') + ':' + String(ahora.getSeconds()).padStart(2, '0');
                    emojiItem.innerText = wakeWordsEmojis[class_idx] + ' ' + probs[class_idx].toFixed(3) + ' ' + currentTime;

                    if (probs[class_idx] < minimumProbabilityScore) {
                        emojiItem.style.textDecoration = 'line-through';
                        emojiItem.style.color = 'gray';
                    } else {
                        document.getElementById('prediction_emoji').innerText = wakeWordsEmojis[class_idx];
                    }
                    document.getElementById('predictions_emojis').appendChild(emojiItem);
                }
            }

            // Limpiar arrayBuffer después del procesamiento
            arrayBuffer = [];

            // Liberar memoria del tensor
            outputTensor.dispose();
        }
    });

    myvad.start();
});

// Funciones auxiliares adicionales para debugging y optimización

// Función para comparar espectrogramas (útil para debugging)
function compareSpectrograms(melSpec1, melSpec2) {
    if (melSpec1.length !== melSpec2.length) {
        console.log(`Different number of frames: ${melSpec1.length} vs ${melSpec2.length}`);
        return false;
    }

    let totalDiff = 0;
    let maxDiff = 0;
    let count = 0;

    for (let i = 0; i < melSpec1.length; i++) {
        if (melSpec1[i].length !== melSpec2[i].length) {
            console.log(`Different mel bins in frame ${i}: ${melSpec1[i].length} vs ${melSpec2[i].length}`);
            return false;
        }

        for (let j = 0; j < melSpec1[i].length; j++) {
            const diff = Math.abs(melSpec1[i][j] - melSpec2[i][j]);
            totalDiff += diff;
            maxDiff = Math.max(maxDiff, diff);
            count++;
        }
    }

    const avgDiff = totalDiff / count;
    console.log(`Spectrogram comparison - Avg diff: ${avgDiff.toFixed(6)}, Max diff: ${maxDiff.toFixed(6)}`);

    return { avgDiff, maxDiff, similar: avgDiff < 0.01 };
}

// Función para validar que los parámetros coincidan con PyTorch
function validateParameters() {
    console.log("=== Parameter Validation ===");
    console.log(`Sample Rate: ${sampleRate}`);
    console.log(`Window Size: ${windowSize}ms`);
    console.log(`Hop Length: ${SPEC_HOP_LENGTH}`);
    console.log(`Mel Bins: ${MEL_SPEC_BINS}`);
    console.log(`FFT Size: ${NUM_FFTS}`);
    console.log(`Log Offset: ${log_offset}`);
    console.log(`ZMUV Mean: ${zmuv_mean}`);
    console.log(`ZMUV Std: ${zmuv_std}`);
    console.log(`Classes: ${classes.join(', ')}`);
    console.log(`Wake Words: ${wakeWords.join(', ')}`);
    console.log("=============================");
}

// Función para procesar un archivo de audio (útil para testing)
async function processAudioFile(audioFile) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = async function (e) {
            try {
                const arrayBuffer = e.target.result;
                const audioBuffer = await offlineCtx.decodeAudioData(arrayBuffer);
                const monoAudio = await resampleAndMakeMono(audioBuffer, sampleRate);

                // Procesar el audio completo
                const chunks = [];
                for (let i = 0; i < monoAudio.length; i += windowBufferSize) {
                    const chunk = monoAudio.slice(i, i + windowBufferSize);
                    if (chunk.length === windowBufferSize) {
                        chunks.push(Array.from(chunk));
                    }
                }

                console.log(`Processed ${chunks.length} chunks from audio file`);
                resolve(chunks);
            } catch (error) {
                reject(error);
            }
        };
        reader.readAsArrayBuffer(audioFile);
    });
}

// Función para batch processing (procesar múltiples chunks a la vez)
async function processBatchAudio(audioChunks) {
    if (audioChunks.length === 0) {
        console.log("No audio chunks to process");
        return [];
    }

    let allDataProcessed = [];
    let batchMels = [];

    // Procesar todos los chunks
    for (let i = 0; i < audioChunks.length; i++) {
        const chunk = audioChunks[i];

        // Calcular mel spectrogram mejorado
        let log_mels = melSpectrogram(chunk, {
            sampleRate: sampleRate,
            hopLength: SPEC_HOP_LENGTH,
            nMels: MEL_SPEC_BINS,
            nFft: NUM_FFTS,
            power: 2.0
        });

        batchMels.push(log_mels);

        // Usar flatten mejorado con ambas normalizaciones
        const processedData = improvedFlatten(log_mels);
        allDataProcessed = [...allDataProcessed, ...processedData];
    }

    const batch = audioChunks.length;

    // Ejecutar el modelo
    let outputTensor = tf.tidy(() => {
        let inputTensor = tf.tensor(
            allDataProcessed,
            [batch, 1, MEL_SPEC_BINS, allDataProcessed.length / (batch * MEL_SPEC_BINS)],
            'float32'
        );
        return tfModel.predict(inputTensor);
    });

    let outputData = await outputTensor.data();
    let results = [];

    // Procesar resultados
    for (let i = 0; i < outputData.length; i += classes.length) {
        let scores = Array.from(outputData.slice(i, i + classes.length));
        let probs = softmax(scores);
        let class_idx = argMax(probs);

        results.push({
            scores: scores,
            probabilities: probs,
            predicted_class: classes[class_idx],
            confidence: probs[class_idx]
        });
    }

    // Liberar memoria
    outputTensor.dispose();

    return results;
}

// Función de estadísticas de rendimiento
class PerformanceMonitor {
    constructor() {
        this.predictions = [];
        this.processingTimes = [];
        this.confidenceScores = [];
    }

    addPrediction(prediction, confidence, processingTime) {
        this.predictions.push(prediction);
        this.confidenceScores.push(confidence);
        this.processingTimes.push(processingTime);

        // Mantener solo las últimas 100 predicciones
        if (this.predictions.length > 100) {
            this.predictions.shift();
            this.confidenceScores.shift();
            this.processingTimes.shift();
        }
    }

    getStats() {
        if (this.predictions.length === 0) {
            return null;
        }

        const avgProcessingTime = this.processingTimes.reduce((a, b) => a + b, 0) / this.processingTimes.length;
        const avgConfidence = this.confidenceScores.reduce((a, b) => a + b, 0) / this.confidenceScores.length;

        // Contar predicciones por clase
        const classCounts = {};
        this.predictions.forEach(pred => {
            classCounts[pred] = (classCounts[pred] || 0) + 1;
        });

        return {
            totalPredictions: this.predictions.length,
            avgProcessingTime: avgProcessingTime,
            avgConfidence: avgConfidence,
            classCounts: classCounts,
            recentPredictions: this.predictions.slice(-10)
        };
    }

    printStats() {
        const stats = this.getStats();
        if (stats) {
            console.log("=== Performance Stats ===");
            console.log(`Total Predictions: ${stats.totalPredictions}`);
            console.log(`Avg Processing Time: ${stats.avgProcessingTime.toFixed(2)}ms`);
            console.log(`Avg Confidence: ${stats.avgConfidence.toFixed(3)}`);
            console.log("Class Counts:", stats.classCounts);
            console.log("Recent Predictions:", stats.recentPredictions.join(', '));
            console.log("========================");
        }
    }
}

// Inicializar monitor de rendimiento
const performanceMonitor = new PerformanceMonitor();

// Función mejorada para debugging de la pipeline completa
function debugPipeline(audioData, enableLogs = true) {
    if (!enableLogs) return;

    console.log("=== Pipeline Debug ===");
    console.log(`Input audio length: ${audioData.length}`);
    console.log(`Sample rate: ${sampleRate}`);
    console.log(`Window buffer size: ${windowBufferSize}`);

    // Verificar estadísticas del audio
    const audioStats = {
        min: Math.min(...audioData),
        max: Math.max(...audioData),
        mean: audioData.reduce((a, b) => a + b, 0) / audioData.length,
        rms: Math.sqrt(audioData.reduce((a, b) => a + b * b, 0) / audioData.length)
    };

    console.log("Audio stats:", audioStats);

    // Calcular mel spectrogram
    const startTime = performance.now();
    let log_mels = melSpectrogram(audioData, {
        sampleRate: sampleRate,
        hopLength: SPEC_HOP_LENGTH,
        nMels: MEL_SPEC_BINS,
        nFft: NUM_FFTS,
        power: 2.0
    });
    const melTime = performance.now() - startTime;

    console.log(`Mel spectrogram shape: ${log_mels.length} x ${log_mels[0].length}`);
    console.log(`Mel calculation time: ${melTime.toFixed(2)}ms`);

    // Verificar estadísticas del mel spectrogram
    let melMin = Infinity, melMax = -Infinity, melSum = 0, melCount = 0;
    for (let i = 0; i < log_mels.length; i++) {
        for (let j = 0; j < log_mels[i].length; j++) {
            const val = log_mels[i][j];
            melMin = Math.min(melMin, val);
            melMax = Math.max(melMax, val);
            melSum += val;
            melCount++;
        }
    }

    console.log("Mel spectrogram stats:", {
        min: melMin,
        max: melMax,
        mean: melSum / melCount
    });

    // Aplicar normalización per-utterance
    const normStartTime = performance.now();
    const normalizedMels = perUtteranceNormalization(log_mels);
    const normTime = performance.now() - normStartTime;

    console.log(`Per-utterance normalization time: ${normTime.toFixed(2)}ms`);

    // Verificar estadísticas después de normalización
    let normMin = Infinity, normMax = -Infinity, normSum = 0, normCount = 0;
    for (let i = 0; i < normalizedMels.length; i++) {
        for (let j = 0; j < normalizedMels[i].length; j++) {
            const val = normalizedMels[i][j];
            normMin = Math.min(normMin, val);
            normMax = Math.max(normMax, val);
            normSum += val;
            normCount++;
        }
    }

    console.log("After per-utterance norm stats:", {
        min: normMin,
        max: normMax,
        mean: normSum / normCount
    });

    // Flatten con ZMUV
    const flattenStartTime = performance.now();
    const flattened = improvedFlatten(log_mels);
    const flattenTime = performance.now() - flattenStartTime;

    console.log(`Flatten time: ${flattenTime.toFixed(2)}ms`);
    console.log(`Flattened data length: ${flattened.length}`);
    console.log(`Expected length: ${MEL_SPEC_BINS * log_mels.length}`);

    const flatStats = {
        min: Math.min(...flattened),
        max: Math.max(...flattened),
        mean: flattened.reduce((a, b) => a + b, 0) / flattened.length
    };

    console.log("Flattened data stats:", flatStats);
    console.log("=====================");
}