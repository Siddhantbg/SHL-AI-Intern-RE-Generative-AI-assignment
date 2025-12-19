import React from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

console.log('🚀 [INDEX] Starting React app initialization...');
console.log('🔍 [INDEX] React version:', React.version);
console.log('🔍 [INDEX] Environment:', process.env.NODE_ENV);
console.log('🔍 [INDEX] API URL:', process.env.REACT_APP_API_URL);

try {
  console.log('🔍 [INDEX] Looking for root element...');
  const container = document.getElementById('root');
  
  if (container) {
    console.log('✅ [INDEX] Root element found:', container);
    console.log('🔍 [INDEX] Creating React root...');
    
    const root = createRoot(container);
    console.log('✅ [INDEX] React root created successfully');
    
    console.log('🎨 [INDEX] Rendering App component...');
    root.render(<App />);
    console.log('✅ [INDEX] App component rendered successfully!');
    
  } else {
    console.error('❌ [INDEX] Root element not found in DOM');
    console.log('🔍 [INDEX] Available elements:', document.body.innerHTML);
  }
} catch (error) {
  console.error('💥 [INDEX] Critical error during React initialization:', error);
  console.error('📋 [INDEX] Error stack:', error instanceof Error ? error.stack : 'No stack trace');
}

// Performance monitoring
try {
  reportWebVitals((metric) => {
    console.log('📊 [PERF]', metric);
  });
} catch (error) {
  console.error('⚠️ [INDEX] reportWebVitals error:', error);
}
