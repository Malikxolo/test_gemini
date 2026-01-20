// Configuration for the Gemini Live Audio Frontend

export const config = {
    // 1. Local Development: Use "localhost"
    // 2. Production: Replace with your Render/Railway backend URL (e.g., "voice-backend.onrender.com")
    BACKEND_URL: "voice-backend-m6w6.onrender.com",

    // Auto-detect secure/insecure protocol (ws:// vs wss://)
    // If on https (Vercel), use wss. If on http (localhost), use ws.
    getWebSocketUrl: function () {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

        // If we are on Vercel (or any non-localhost domain), and BACKEND_URL is still localhost,
        // we should probably warn or try to respect the hardcoded value if the user forgot to change it.
        // But generally, for local dev:
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            // When running locally, assume backend is at localhost:8000
            // This works if accessing via http://localhost:8000 (served by backend)
            // OR if running frontend separately (e.g. Live Server), assuming backend is up on 8000.
            return `ws://localhost:8000/ws`;
        } else {
            // Production (Vercel)
            // Use the configured BACKEND_URL
            return `${protocol}//${this.BACKEND_URL}/ws`;
        }
    }
};
