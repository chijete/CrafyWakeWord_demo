// Audio utils

// Follow audio utils are referred from megenta-js 
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
    for (let i = 0; i < arr.length; i++) {
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

function melSpectrogram(y, params) {
    if (!params.power) {
        params.power = 2.0;
    }
    const stftMatrix = stft(y, params);
    const [spec, nFft] = magSpectrogram(stftMatrix, params.power);

    params.nFft = nFft;
    const melBasis = createMelFilterbank(params);
    return applyWholeFilterbank(spec, melBasis);
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

function softmax(arr) {
    return arr.map(function (value, index) {
        return Math.exp(value) / arr.map(function (y /*value*/) { return Math.exp(y) }).reduce(function (a, b) { return a + b })
    })
}

function flatten(log_mels) {
    flatten_arry = []
    for (i = 0; i < MEL_SPEC_BINS; i++) {
        for (j = 0; j < log_mels.length; j++) {
            flatten_arry.push((Math.log(log_mels[j][i] + log_offset) - zmuv_mean) / zmuv_std)
        }
    }
    return flatten_arry
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

window.addEventListener('load', async function () {
    myvad = await vad.MicVAD.new({
        onSpeechStart: () => {
            console.log("Speech start detected");
            vad_speaking = true;
        },
        onSpeechEnd: async (audio) => {
            vad_speaking = false;
            console.log("Speech end detected");
            // Process the captured speech audio
            arrayBuffer = Array.from(audio); // Assuming audio is Float32Array at 16000 Hz mono
            if (arrayBuffer.length < windowBufferSize) {
                // Optionally pad if shorter, but for now, skip or handle
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
                    // Discard last incomplete chunk or pad, here discard
                    break;
                }
                // Calculate log mels
                let log_mels = melSpectrogram(batchBuffer, {
                    sampleRate: sampleRate,
                    hopLength: SPEC_HOP_LENGTH,
                    nMels: MEL_SPEC_BINS,
                    nFft: NUM_FFTS
                });
                batchBuffers.push(batchBuffer);
                batchMels.push(log_mels);
                dataProcessed = [...dataProcessed, ...flatten(log_mels)];
                batch++;
            }
            if (batch === 0) {
                console.log("No valid chunks to process");
                return;
            }
            // Run model with Tensor inputs and get the result.
            let outputTensor = tf.tidy(() => {
                let inputTensor = tf.tensor(dataProcessed, [batch, 1, MEL_SPEC_BINS, dataProcessed.length / (batch * MEL_SPEC_BINS)], 'float32');
                let outputTensor = tfModel.predict(inputTensor);
                return outputTensor;
            });
            let outputData = await outputTensor.data();
            for (let i = 0; i < outputData.length; i += classes.length) {
                let scores = Array.from(outputData.slice(i, i + classes.length));
                console.log("scores", scores);
                let probs = softmax(scores);
                let probs_sum = probs.reduce((sum, x) => sum + x, 0);
                probs = probs.map(x => x / probs_sum);
                let class_idx = argMax(probs);
                console.log("probabilities", probs);
                console.log("predicted word", classes[class_idx]);
                if (wakeWordsEmojis[class_idx] !== undefined && probs.length === classes.length && probs[class_idx] !== 1) {
                    let emojiItem = document.createElement('p');
                    const ahora = new Date();
                    var currentTime = String(ahora.getHours()).padStart(2, '0') + ':' + String(ahora.getMinutes()).padStart(2, '0') + ':' + String(ahora.getSeconds()).padStart(2, '0');
                    emojiItem.innerText = wakeWordsEmojis[class_idx] + ' ' + probs[class_idx] + ' ' + currentTime;
                    if (probs[class_idx] < minimumProbabilityScore) {
                        emojiItem.style.textDecoration = 'line-through';
                    } else {
                        document.getElementById('prediction_emoji').innerText = wakeWordsEmojis[class_idx];
                    }
                    document.getElementById('predictions_emojis').appendChild(emojiItem);
                }
            }
            // Clear arrayBuffer after processing
            arrayBuffer = [];
        }
    });
    myvad.start();
});