/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                primary: "#76B900", // NVIDIA Green
                "primary-dim": "#5e9400",
                surface: "#111111",
                "surface-hover": "#1a1a1a",
                "glass-bg": "rgba(20, 20, 20, 0.7)",
                "glass-border": "rgba(255, 255, 255, 0.1)",
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
            }
        },
    },
    plugins: [],
}
