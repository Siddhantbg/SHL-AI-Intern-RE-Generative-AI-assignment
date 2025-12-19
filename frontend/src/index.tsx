import React from 'react';
import { createRoot } from 'react-dom/client';

console.log('🚀 [INDEX] Starting React app initialization...');

// Simple test component
const TestApp = () => {
  console.log('🎯 [TEST] TestApp rendering...');
  return (
    <div style={{ 
      padding: '20px', 
      fontFamily: 'Arial, sans-serif',
      backgroundColor: '#f0f0f0',
      minHeight: '100vh'
    }}>
      <h1 style={{ color: 'blue', textAlign: 'center' }}>
        🎯 React Test - Working!
      </h1>
      <p style={{ textAlign: 'center', marginTop: '20px' }}>
        If you can see this, React is loading properly.
      </p>
      <div style={{ 
        backgroundColor: 'white', 
        padding: '20px', 
        margin: '20px auto',
        maxWidth: '600px',
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <h2>✅ System Check</h2>
        <p>✅ React: Working</p>
        <p>✅ JavaScript: Loading</p>
        <p>✅ DOM: Rendering</p>
      </div>
    </div>
  );
};

try {
  console.log('🔍 [INDEX] Looking for root element...');
  const container = document.getElementById('root');
  
  if (container) {
    console.log('✅ [INDEX] Root element found');
    const root = createRoot(container);
    console.log('🎨 [INDEX] Rendering test app...');
    root.render(<TestApp />);
    console.log('✅ [INDEX] Test app rendered!');
  } else {
    console.error('❌ [INDEX] Root element not found');
  }
} catch (error) {
  console.error('💥 [INDEX] Error:', error);
}
