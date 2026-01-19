class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = [];
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input.length > 0) {
            const channelData = input[0];

            // Just pass raw float32 data to the main thread
            // We'll convert to Int16 PCM in the main thread for simplicity 
            // or we could do it here. Doing it here saves message passing overhead.

            // Convert Float32 (-1.0 to 1.0) to Int16 (-32768 to 32767)
            const int16Data = new Int16Array(channelData.length);
            for (let i = 0; i < channelData.length; i++) {
                const s = Math.max(-1, Math.min(1, channelData[i]));
                int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }

            this.port.postMessage(int16Data);
        }
        return true;
    }
}

registerProcessor('pcm-processor', PCMProcessor);
