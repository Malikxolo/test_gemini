// Gemini Live Voice Agent - Optimized Client
// Uses binary WebSocket frames for reduced latency

import { config } from './config.js';

const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');

let audioContext = null;
let playbackContext = null; // Separate context for 24kHz playback
let mediaStream = null;
let audioWorkletNode = null;
let websocket = null;
let isRecording = false;
let nextStartTime = 0;
let scheduledAudioSources = [];

// Latency tracking
let speechStartTime = null;
let firstAudioTime = null;

function log(level, msg, data = null) {
    const timestamp = new Date().toISOString().split('T')[1].slice(0, 12);
    const prefix = `[${timestamp}] [GeminiLive]`;
    if (data) {
        console[level](`${prefix} ${msg}`, data);
    } else {
        console[level](`${prefix} ${msg}`);
    }
}

function updateStatus(active, text) {
    const appContainer = document.querySelector('.app-container');
    if (active) {
        statusIndicator.classList.add('active');
        if (appContainer) appContainer.classList.add('session-active');
    } else {
        statusIndicator.classList.remove('active');
        if (appContainer) appContainer.classList.remove('session-active');
    }
    statusText.textContent = text;
}

async function startSession() {
    try {
        log('info', 'Starting session...');
        updateStatus(false, 'Connecting...');

        const wsUrl = config.getWebSocketUrl();
        log('info', `Connecting to: ${wsUrl}`);
        websocket = new WebSocket(wsUrl);
        websocket.binaryType = 'arraybuffer'; // Enable binary messages

        websocket.onopen = async () => {
            log('info', '✓ Connected to server');
            updateStatus(true, 'Listening');
            await startAudioCapture();
            startBtn.disabled = true;
            stopBtn.disabled = false;
        };

        websocket.onmessage = (event) => {
            if (event.data instanceof ArrayBuffer) {
                // Binary PCM16 audio - direct 24kHz from Gemini
                // log('debug', `📥 Received binary audio: ${event.data.byteLength} bytes`);
                playPcmBinary(new Int16Array(event.data));
            } else if (event.data instanceof Blob) {
                // Handle Blob if browser defaults to it
                log('info', `📥 Received Blob: ${event.data.size} bytes - converting...`);
                const reader = new FileReader();
                reader.onload = () => {
                    playPcmBinary(new Int16Array(reader.result));
                };
                reader.readAsArrayBuffer(event.data);
            } else if (typeof event.data === 'string') {
                // JSON control message
                const message = JSON.parse(event.data);
                handleServerMessage(message);
            } else {
                log('warn', 'Received unknown message type', event.data);
            }
        };

        websocket.onclose = (event) => {
            log('info', 'Connection closed', { code: event.code });
            stopSession();
        };

        websocket.onerror = (err) => {
            log('error', 'Connection error', err);
        };

    } catch (err) {
        log('error', 'Session failed', err.message);
        updateStatus(false, 'Error');
    }
}

async function startAudioCapture() {
    isRecording = true;

    // Create separate contexts for input (16kHz) and output (24kHz source, system rate context)
    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    // Use system default sample rate for playback (more robust)
    playbackContext = new (window.AudioContext || window.webkitAudioContext)();

    // Resume playback context (browsers require this after user interaction)
    await playbackContext.resume();
    log('info', `🔊 Playback context ready (state: ${playbackContext.state}, rate: ${playbackContext.sampleRate}Hz)`);

    await audioContext.audioWorklet.addModule('pcm-processor.js');

    mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, sampleRate: 16000 }
    });
    log('info', '🎤 Microphone ready');

    const source = audioContext.createMediaStreamSource(mediaStream);
    audioWorkletNode = new AudioWorkletNode(audioContext, 'pcm-processor');

    let chunkCount = 0;
    let isSpeaking = false;

    audioWorkletNode.port.onmessage = (event) => {
        if (!isRecording || websocket.readyState !== WebSocket.OPEN) return;

        const pcmInt16 = event.data;
        chunkCount++;

        // Detect speech start (simple energy detection)
        if (!isSpeaking) {
            let energy = 0;
            for (let i = 0; i < pcmInt16.length; i++) {
                energy += Math.abs(pcmInt16[i]);
            }
            energy /= pcmInt16.length;

            if (energy > 500) { // Threshold for voice activity
                isSpeaking = true;
                speechStartTime = performance.now();
                log('info', '🎤 Speech detected');
            }
        }

        // Send as binary frame (more efficient than JSON+base64)
        websocket.send(pcmInt16.buffer);

        if (chunkCount % 200 === 0) {
            log('debug', `Sent ${chunkCount} chunks`);
        }
    };

    source.connect(audioWorkletNode);
    log('info', 'Audio streaming started');
}

function handleServerMessage(message) {
    // Connection established
    if (message.type === 'connected') {
        log('info', '✓ ' + message.message);
        return;
    }

    // Turn complete
    if (message.type === 'turnComplete') {
        log('info', '✓ Turn complete');
        speechStartTime = null;
        firstAudioTime = null;
        return;
    }

    // Interrupted by user
    if (message.type === 'interrupted') {
        log('info', '⚡ Interrupted');
        stopAudioQueue();
        speechStartTime = null;
        firstAudioTime = null;
        return;
    }

    // Error
    if (message.type === 'error') {
        log('error', 'Server error: ' + message.message);
        return;
    }
}

// Play binary PCM16 audio directly (24kHz from Gemini) with resampling
function playPcmBinary(int16Data) {
    // Ensure playback context exists and is running
    if (!playbackContext || playbackContext.state !== 'running') {
        log('warn', `Playback context not ready: ${playbackContext?.state || 'null'}`);
        return;
    }

    // Measure latency on first audio chunk
    if (firstAudioTime === null && speechStartTime !== null) {
        firstAudioTime = performance.now();
        const latency = firstAudioTime - speechStartTime;
        log('info', `⚡ CLIENT LATENCY: ${latency.toFixed(0)}ms (speech → first audio)`);
    }

    // 1. Convert Int16 to Float32 (Standard Web Audio format)
    const float32Data = new Float32Array(int16Data.length);
    for (let i = 0; i < int16Data.length; i++) {
        float32Data[i] = int16Data[i] / 32768.0;
    }

    // 2. Prepare Source Buffer (24kHz)
    // We cannot create a buffer with a different sample rate than the context in all browsers reliably.
    // Instead, we create a buffer at the system rate and resample, OR we let the browser handle it if supported.
    // Most modern browsers support creating a buffer with a specific sampleRate, and the SourceNode resamples automatically.

    // Let's try the native resampling first which is efficient (createBuffer with 24000)
    // If your previous attempt failed, it might be that the browser silences it if rates mismatch wildly without a proper node.
    // BUT since you heard NOTHING, it might be a context time issue.

    const sourceRate = 24000;
    const buffer = playbackContext.createBuffer(1, float32Data.length, sourceRate);
    buffer.getChannelData(0).set(float32Data);

    const source = playbackContext.createBufferSource();
    source.buffer = buffer;
    source.connect(playbackContext.destination);

    // Schedule for gapless playback
    const now = playbackContext.currentTime;

    // CRITICAL FIX: Ensure we don't schedule in the past
    // If nextStartTime is way behind 'now', we must reset it, otherwise audio plays "instantly" to catch up (chipmunk) or is dropped.
    if (nextStartTime < now) {
        nextStartTime = now + 0.05; // Add small buffer (50ms) to prevent glitch
    }

    source.start(nextStartTime);
    nextStartTime += buffer.duration;

    source.onended = () => {
        scheduledAudioSources = scheduledAudioSources.filter(s => s !== source);
    };
    scheduledAudioSources.push(source);
}

function stopAudioQueue() {
    const count = scheduledAudioSources.length;
    scheduledAudioSources.forEach(source => {
        try { source.stop(); } catch (e) { }
    });
    scheduledAudioSources = [];
    nextStartTime = 0;
    if (count > 0) {
        log('info', `Cleared ${count} audio buffers`);
    }
}

function stopSession() {
    log('info', 'Stopping...');
    isRecording = false;
    stopAudioQueue();
    updateStatus(false, 'Stopped');
    startBtn.disabled = false;
    stopBtn.disabled = true;

    if (audioWorkletNode) audioWorkletNode.disconnect();
    if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
    if (audioContext) audioContext.close();
    if (playbackContext) playbackContext.close();
    if (websocket) websocket.close();

    speechStartTime = null;
    firstAudioTime = null;
    log('info', 'Session ended');
}

startBtn.addEventListener('click', startSession);
stopBtn.addEventListener('click', stopSession);

log('info', 'Client ready (binary frames, latency tracking)');
