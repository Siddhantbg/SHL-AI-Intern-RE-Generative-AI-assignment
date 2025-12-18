import React, { useState, useEffect } from 'react';
import ErrorBoundary from './components/ErrorBoundary';
import QueryInput from './components/QueryInput';
import RecommendationTable from './components/RecommendationTable';
import LoadingSpinner from './components/LoadingSpinner';
import { apiService } from './services/api';
import { Assessment, RecommendationResponse, ApiError } from './types';

function App() {
  const [recommendations, setRecommendations] = useState<Assessment[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentQuery, setCurrentQuery] = useState<string>('');
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  // Debug: Add console log to verify React is running
  console.log('App component rendering...');

  // Check API health on component mount
  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      await apiService.getHealth();
      setApiStatus('online');
    } catch (error) {
      setApiStatus('offline');
      console.error('API health check failed:', error);
    }
  };

  const handleQuerySubmit = async (query: string) => {
    setIsLoading(true);
    setError(null);
    setCurrentQuery(query);

    try {
      const response: RecommendationResponse = await apiService.getRecommendations(query);
      setRecommendations(response.recommendations || []);
    } catch (error: any) {
      console.error('Error getting recommendations:', error);
      
      let errorMessage = 'Failed to get recommendations. Please try again.';
      
      if (error.response?.data) {
        const apiError: ApiError = error.response.data;
        errorMessage = apiError.message || errorMessage;
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      setError(errorMessage);
      setRecommendations([]);
    } finally {
      setIsLoading(false);
    }
  };

  const getApiStatusIndicator = () => {
    switch (apiStatus) {
      case 'checking':
        return (
          <div className="flex items-center text-yellow-600">
            <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-yellow-600 mr-2"></div>
            <span className="text-xs">Checking API...</span>
          </div>
        );
      case 'online':
        return (
          <div className="flex items-center text-green-600">
            <div className="h-2 w-2 bg-green-600 rounded-full mr-2"></div>
            <span className="text-xs">API Online</span>
          </div>
        );
      case 'offline':
        return (
          <div className="flex items-center text-red-600">
            <div className="h-2 w-2 bg-red-600 rounded-full mr-2"></div>
            <span className="text-xs">API Offline</span>
          </div>
        );
    }
  };

  // Simple test to ensure React is working
  if (typeof window !== 'undefined') {
    console.log('React App is running in browser');
  }

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50" style={{ minHeight: '100vh', backgroundColor: '#f9fafb', padding: '20px' }}>
        {/* Simple test header */}
        <div style={{ backgroundColor: 'white', padding: '20px', marginBottom: '20px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
          <h1 style={{ fontSize: '24px', fontWeight: 'bold', color: '#111827', margin: '0 0 8px 0' }}>
            SHL Assessment Recommender
          </h1>
          <p style={{ fontSize: '14px', color: '#6b7280', margin: '0' }}>
            Find the right assessments for your hiring needs
          </p>
          <div style={{ marginTop: '10px' }}>
            {getApiStatusIndicator()}
          </div>
        </div>

        {/* Header */}
        <header className="bg-white shadow-sm border-b border-gray-200" style={{ backgroundColor: 'white', borderBottom: '1px solid #e5e7eb', boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)', display: 'none' }}>
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center py-4">
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  SHL Assessment Recommender
                </h1>
                <p className="text-sm text-gray-600">
                  Find the right assessments for your hiring needs
                </p>
              </div>
              <div className="flex items-center space-x-4">
                {getApiStatusIndicator()}
                <button
                  onClick={checkApiHealth}
                  className="text-xs text-gray-500 hover:text-gray-700 px-2 py-1 rounded border border-gray-300 hover:border-gray-400"
                >
                  Refresh
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* API Offline Warning */}
          {apiStatus === 'offline' && (
            <div className="mb-6 bg-red-50 border border-red-200 rounded-md p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">
                    API Connection Failed
                  </h3>
                  <div className="mt-2 text-sm text-red-700">
                    <p>
                      Unable to connect to the recommendation API. Please ensure the backend service is running 
                      and accessible at the configured endpoint.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Query Input */}
          <QueryInput onSubmit={handleQuerySubmit} isLoading={isLoading} />

          {/* Error Display */}
          {error && (
            <div className="mb-6 bg-red-50 border border-red-200 rounded-md p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">Error</h3>
                  <div className="mt-2 text-sm text-red-700">
                    <p>{error}</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Loading State */}
          {isLoading && (
            <LoadingSpinner message="Finding the best assessment recommendations for your query..." />
          )}

          {/* Results */}
          {!isLoading && currentQuery && (
            <RecommendationTable 
              recommendations={recommendations} 
              query={currentQuery}
            />
          )}

          {/* Welcome Message */}
          {!isLoading && !currentQuery && !error && (
            <div className="bg-white shadow-sm rounded-lg p-8 text-center">
              <svg
                className="mx-auto h-16 w-16 text-gray-400 mb-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                Welcome to SHL Assessment Recommender
              </h3>
              <p className="text-gray-600 max-w-2xl mx-auto">
                Enter a job description, role requirements, or specific skills above to get personalized 
                recommendations for SHL assessments. Our AI-powered system will analyze your query and 
                suggest the most relevant assessments from SHL's comprehensive catalog.
              </p>
            </div>
          )}
        </main>

        {/* Footer */}
        <footer className="bg-white border-t border-gray-200 mt-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <div className="text-center text-sm text-gray-500">
              <p>
                SHL Assessment Recommendation System - Powered by AI and Machine Learning
              </p>
              <p className="mt-1">
                Built with React, TypeScript, and Tailwind CSS
              </p>
            </div>
          </div>
        </footer>
      </div>
    </ErrorBoundary>
  );
}

export default App;
