/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        slate: { 50: '#faf9f5', 100: '#f0eee6', 200: '#deded3', 300: '#c4c9bf', 400: '#a5afa4', 500: '#879387', 600: '#5b695f', 700: '#394a40', 800: '#28392f', 900: '#19271f', 950: '#101a15' },
        brand: {
          50: "#eef2ff",
          100: "#e0e7ff",
          400: "#c0d7a1",
          500: "#a9c783",
          600: "#47633b",
          700: "#3a5230",
        },
      },
    },
  },
  plugins: [],
};
