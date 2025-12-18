import React from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App';

console.log('🔥 index.tsx loaded');

const container = document.getElementById('root');
if (container) {
  console.log('✅ Root element found');
  const root = createRoot(container);
  console.log('🎨 Rendering App...');
  
  root.render(React.createElement(App));
  console.log('✅ App rendered!');
} else {
  console.error('❌ Root element not found');
}

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
