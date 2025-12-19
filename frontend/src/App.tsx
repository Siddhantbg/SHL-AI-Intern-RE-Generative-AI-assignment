import React, { useState, useEffect } from 'react';

console.log('🚀 [APP] App.tsx loaded');

const App: React.FC = () => {
  console.log('🎯 [APP] App component initializing...');
  
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const apiUrl = import.meta.env.VITE_API_URL || 'https://shl-assessment-recommender.onrender.com';
  
  console.log('🔗 [APP] API URL configured:', apiUrl);

  useEffect(() => {
    console.log('🔄 [APP] Testing API connection...');
    
    fetch(`${apiUrl}/health`)
      .then(response => {
        console.log('📡 [APP] Health check response:', response.status);
        setApiStatus(response.ok ? 'online' : 'offline');
      })
      .catch(error => {
        console.error('❌ [APP] API connection failed:', error);
        setApiStatus('offline');
      });
  }, [apiUrl]);

  console.log('🎨 [APP] Rendering App component...');

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: '#f3f4f6', 
      padding: '20px',
      fontFamily: 'Arial, sans-serif'
    }}>
      <div style={{ 
        backgroundColor: 'white', 
        padding: '30px', 
        borderRadius: '8px', 
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        maxWidth: '800px',
        margin: '0 auto'
      }}>
        <h1 style={{ 
          fontSize: '32px', 
          fontWeight: 'bold', 
          color: '#1f2937', 
          margin: '0 0 16px 0',
          textAlign: 'center'
        }}>
          🎯 SHL Assessment Recommender
        </h1>
        
        <p style={{ 
          fontSize: '16px', 
          color: '#6b7280', 
          margin: '0 0 24px 0',
          textAlign: 'center'
        }}>
          Find the right assessments for your hiring needs
        </p>

        <div style={{
          backgroundColor: '#f9fafb',
          padding: '20px',
          borderRadius: '6px',
          border: '1px solid #e5e7eb'
        }}>
          <h2 style={{ fontSize: '18px', margin: '0 0 12px 0', color: '#374151' }}>
            🚀 System Status
          </h2>
          <p style={{ margin: '0', color: '#059669' }}>
            ✅ Frontend: React app loaded successfully
          </p>
          <p style={{ 
            margin: '8px 0 0 0', 
            color: apiStatus === 'online' ? '#059669' : '#dc2626' 
          }}>
            {apiStatus === 'online' ? '✅' : '❌'} Backend: {apiStatus} ({apiUrl})
          </p>
        </div>

        <div style={{
          marginTop: '24px',
          padding: '20px',
          backgroundColor: '#eff6ff',
          borderRadius: '6px',
          border: '1px solid #bfdbfe'
        }}>
          <h3 style={{ fontSize: '16px', margin: '0 0 12px 0', color: '#1e40af' }}>
            ✅ React is Working!
          </h3>
          <p style={{ margin: '0 0 8px 0', color: '#1e40af' }}>
            ✅ React components are rendering properly
          </p>
          <p style={{ margin: '0 0 8px 0', color: '#1e40af' }}>
            ✅ JavaScript bundle loaded successfully
          </p>
          <p style={{ margin: '0', color: '#1e40af' }}>
            ✅ No more blank screen - React is functional!
          </p>
        </div>
      </div>
    </div>
  );
};

export default App;
