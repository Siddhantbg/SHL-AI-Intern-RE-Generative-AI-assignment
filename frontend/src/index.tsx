import React from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

console.log('🔥 index.tsx loaded');
console.log('📦 React imported:', !!React);
console.log('🎯 createRoot imported:', !!createRoot);

try {
  console.log('🔍 Looking for root element...');
  const container = document.getElementById('root');
  
  if (!container) {
    console.error('❌ Root element not found!');
    throw new Error('Failed to find the root element');
  }
  
  console.log('✅ Root element found:', container);
  console.log('🏗️ Creating React root...');
  
  const root = createRoot(container);
  console.log('✅ React root created');
  
  console.log('🎨 Rendering App component...');
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
  
  console.log('✅ App rendered successfully!');
  
} catch (error) {
  console.error('💥 Error in index.tsx:', error);
  
  // Fallback rendering
  const container = document.getElementById('root');
  if (container) {
    container.innerHTML = `
      <div style="padding: 20px; font-family: Arial, sans-serif;">
        <h1 style="color: red;">React App Error</h1>
        <p>Error: ${error.message}</p>
        <p>Check console for details.</p>
      </div>
    `;
  }
}

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
