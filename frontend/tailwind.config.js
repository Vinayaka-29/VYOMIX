/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        space: {
          950: '#040711',
          900: '#070c18',
          850: '#0c1326',
          800: '#111c38',
          700: '#1c2b50',
          600: '#2a3e72',
        },
        satcyan: {
          DEFAULT: '#00f2fe',
          400: '#22d3ee',
          500: '#06b6d4',
          600: '#0891b2',
        },
        satteal: '#059669',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      backgroundImage: {
        'grid-pattern': "radial-gradient(rgba(6, 182, 212, 0.08) 1px, transparent 1px)",
      },
    },
  },
  plugins: [],
};
