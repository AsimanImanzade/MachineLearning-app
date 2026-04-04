/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        base: {
          900: '#07070f',
          800: '#0d0d1a',
          700: '#12121f',
          600: '#1a1a2e',
          500: '#252540',
        },
        accent: {
          purple: '#7c3aed',
          'purple-light': '#a78bfa',
          blue: '#2563eb',
          'blue-light': '#60a5fa',
          cyan: '#06b6d4',
          pink: '#ec4899',
        },
        glass: 'rgba(255,255,255,0.04)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'hero-glow': 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(124,58,237,0.25), transparent)',
      },
      animation: {
        'fade-in': 'fadeIn 0.4s ease-out',
        'slide-up': 'slideUp 0.4s cubic-bezier(0.16,1,0.3,1)',
        'glow-pulse': 'glowPulse 3s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: { from: { opacity: '0' }, to: { opacity: '1' } },
        slideUp: { from: { opacity: '0', transform: 'translateY(16px)' }, to: { opacity: '1', transform: 'translateY(0)' } },
        glowPulse: { '0%,100%': { opacity: '0.6' }, '50%': { opacity: '1' } },
      },
      boxShadow: {
        glass: '0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06)',
        glow: '0 0 30px rgba(124,58,237,0.3)',
        'glow-blue': '0 0 30px rgba(37,99,235,0.3)',
      },
    },
  },
  plugins: [],
}
