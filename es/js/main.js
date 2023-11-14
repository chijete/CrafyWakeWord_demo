window.AudioContext = window.AudioContext || window.webkitAudioContext;

var audioContext;
var audioInput = null,
    realAudioInput = null,
    inputPoint = null,
    recording = false;
var rafID = null;
var analyserContext = null;
var canvasWidth, canvasHeight;

const classes = ["almeja", "cerebro", "patata", "oov"];
const wakeWords = ["almeja", "cerebro", "patata"];
const wakeWordsEmojis = ["🐚 Almeja", "🧠 Cerebro", "🥔 Patata"];
var bufferSize = 1024;
var channels = 1;
var windowSize = 750;
var zmuv_mean = 0.000016;
var zmuv_std = 0.072771;
var log_offset = 1e-7;
var SPEC_HOP_LENGTH = 200;
var MEL_SPEC_BINS = 40;
var NUM_FFTS = 512;
var audioFloatSize = 32767;
var sampleRate = 16000;
var numOfBatches = 2;

let predictWords = [];
let arrayBuffer = [];
let targetState = 0;

let bufferMap = {};

const windowBufferSize = windowSize / 1000 * sampleRate;

let tfModel;
async function loadModel() {
    tfModel = await tf.loadGraphModel('web_model/model.json');
}
loadModel();

function loadModelData() {
    return new Promise(function (resolve, reject) {
        fetch('model_data.json')
            .then(response => {
                // Verificar si la respuesta de la solicitud es exitosa (código de estado 200)
                if (response.status === 200) {
                    // Convertir la respuesta a JSON
                    return response.json();
                } else {
                    throw new Error('Error al cargar el archivo JSON');
                }
            })
            .then(data => {
                // Manejar los datos del archivo JSON
                windowSize = data['window_size'];
                zmuv_mean = data['zmuv_mean'];
                zmuv_std = data['zmuv_std'];
                SPEC_HOP_LENGTH = data['hop_length'];
                MEL_SPEC_BINS = data['num_mels'];
                NUM_FFTS = data['num_fft'];
                sampleRate = data['sample_rate'];
                resolve(data);
            })
            .catch(error => {
                // Manejar errores
                console.error('Error:', error);
                reject(error);
            });
    });
}

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

function gotStream(stream) {
    inputPoint = audioContext.createGain();

    // Create an AudioNode from the stream.
    realAudioInput = audioContext.createMediaStreamSource(stream);
    audioInput = realAudioInput;

    audioInput = convertToMono( audioInput );
    audioInput.connect(inputPoint);

    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 2048;
    inputPoint.connect( analyserNode );

    // bufferSize, in_channels, out_channels
    scriptNode = (audioContext.createScriptProcessor || audioContext.createJavaScriptNode).call(audioContext, bufferSize, channels, channels);
    scriptNode.onaudioprocess = async function (audioEvent) {
        if (recording) {
            let resampledMonoAudio = await resampleAndMakeMono(audioEvent.inputBuffer);
            arrayBuffer = [...arrayBuffer, ...resampledMonoAudio]
            batchSize = Math.floor(arrayBuffer.length/windowBufferSize)
            // if we got batches * 750 ms seconds of buffer 
            let batchBuffers = []
            let batchMels = []
            if (arrayBuffer.length >= numOfBatches * windowBufferSize) {
                let batch = 0
                let dataProcessed, log_mels;
                for (let i = 0; i < arrayBuffer.length; i = i + windowBufferSize) {
                    batchBuffer = arrayBuffer.slice(i, i+windowBufferSize)
                    //  if it is less than 750 ms then pad it with ones
                    if (batchBuffer.length < windowBufferSize) {
                        //batchBuffer = padArray(batchBuffer, windowBufferSize, 1)
                        // discard last slice
                        break
                    }
                    // arrayBuffer = arrayBuffer.filter(x => x/audioFloatSize)
                    // calculate log mels
                    log_mels = melSpectrogram(batchBuffer, {
                        sampleRate: sampleRate,
                        hopLength: SPEC_HOP_LENGTH,
                        nMels: MEL_SPEC_BINS,
                        nFft: NUM_FFTS
                    });
                    batchBuffers.push(batchBuffer)
                    batchMels.push(log_mels)
                    if (batch == 0) {
                        dataProcessed = []
                    }
                    dataProcessed = [...dataProcessed, ...flatten(log_mels)]
                    batch = batch + 1
                }
                // clear buffer
                arrayBuffer = []
                // Run model with Tensor inputs and get the result.
                let outputTensor = tf.tidy(() => {
                    let inputTensor = tf.tensor(dataProcessed, [batch, 1, MEL_SPEC_BINS, dataProcessed.length/(batch * MEL_SPEC_BINS)], 'float32');
                    let outputTensor = tfModel.predict(inputTensor);
                    return outputTensor
                  });
                let outputData = await outputTensor.data();
                for (let i = 0; i<outputData.length; i = i+classes.length) {
                    let scores = Array.from(outputData.slice(i,i+classes.length))
                    console.log("scores", scores)
                    let probs = softmax(scores)
                    probs_sum = probs.reduce( (sum, x) => x+sum)
                    probs = probs.filter(x => x/probs_sum)
                    let class_idx = argMax(probs)
                    console.log("probabilities", probs)
                    console.log("predicted word", classes[class_idx])
                    if (wakeWordsEmojis[class_idx] !== undefined) {
                        document.getElementById('prediction_emoji').innerText = wakeWordsEmojis[class_idx];
                        let emojiItem = document.createElement('p');
                        emojiItem.innerText = wakeWordsEmojis[class_idx];
                        document.getElementById('predictions_emojis').appendChild(emojiItem);
                    }
                    // if (classes[targetState] == classes[class_idx]) {
                    //     bufferMap[`${classes[targetState]}_buffer`] = batchBuffers[Math.floor(i/classes.length)]
                    //     bufferMap[`${classes[targetState]}_mels`] = batchMels[Math.floor(i/classes.length)]
                    //     console.log(classes[class_idx])
                    //     addprediction(classes[class_idx])
                    //     predictWords.push(classes[class_idx]) 
                    //     targetState += 1
                    //     if (wakeWords.join(' ') == predictWords.join(' ')) {
                    //         addprediction(`Wake word detected - ${predictWords.join(' ')}`)
                    //         let prompt = new Audio("static/audio/prompt.mp3");
                    //         prompt.play()
                    //         // stop recording
                    //         document.getElementById("record").click();
                    //         predictWords = []
                    //         targetState = 0
                    //     }
                    // }
                }
            }
        }
    }
    inputPoint.connect(scriptNode);
    scriptNode.connect(audioContext.destination);

    zeroGain = audioContext.createGain();
    zeroGain.gain.value = 0.0;
    inputPoint.connect( zeroGain );
    zeroGain.connect( audioContext.destination );
    // updateAnalysers();
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

    constraints = {audio: true}
    navigator.mediaDevices.getUserMedia(constraints)
        .then(gotStream)
        .catch(function(err) {
            alert('Error getting audio');
            console.log(err);
        });
}

window.addEventListener('load', function () {
    start_button.onclick = function () {
        loadModelData().then(function (result) {
            audioContext = new AudioContext();
            recording = true;
            initAudio();
        }).catch(function(error) {});
    };
} );