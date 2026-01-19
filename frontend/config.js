// Configuration for the Gemini Live Audio Frontend

export const config = {
    // 1. Local Development: Use "localhost"
    // 2. Production: Replace with your Render/Railway backend URL (e.g., "voice-backend.onrender.com")
    BACKEND_URL: "https://voice-backend-m6w6.onrender.com",

    // Auto-detect secure/insecure protocol (ws:// vs wss://)
    // If on https (Vercel), use wss. If on http (localhost), use ws.
    getWebSocketUrl: function () {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

        // If we are on Vercel (or any non-localhost domain), and BACKEND_URL is still localhost,
        // we should probably warn or try to respect the hardcoded value if the user forgot to change it.
        // But generally, for local dev:
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return `ws://${this.BACKEND_URL}/ws`; // Local development
        } else {
            // Production (Vercel)
            // Replace THIS string with your actual backend URL after deploying the backend
            // Example: return `wss://your-app-name.onrender.com/ws`;

            // For now, returning a placeholder or the config value
            return `${protocol}//${this.BACKEND_URL}/ws`;
        }
    }
};
