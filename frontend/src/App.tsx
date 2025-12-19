import React, { useState, useEffect } from 'react';

// Components
import QueryInput from './components/QueryInput';
import RecommendationTable from './components/RecommendationTable';
import LoadingSpinner from './components/LoadingSpinner';
import ErrorBoundary from './components/ErrorBoundary';

interface Recommendation {
  assessment_name: string;
  url: string;
  test_type: string;
  category: string;
  relevance_score: number;
  explanation: string;
  skills_matched: string[];
}

interface ApiResponse {
  query: string;
  recommendations: Recommendation[];
  total_results: number;
  processing_time: number;
  query_info: {
    processing_method: string;
    confidence_score: number;
  };
}

const App: React.FC = () => {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentQuery, setCurrentQuery] = useState<string>('');
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline' | 'error'>('checking');

  const apiUrl = process.env.REACT_APP_API_URL || 'https://shl-assessment-recommender.onrender.com';

  useEffect(() => {
    // Test API connection on component mount
    const testApiConnection = async () => {
      try {
        const response = await fetch(`${apiUrl}/health`);
        setApiStatus(response.ok ? 'online' : 'error');
      } catch (error) {
        console.error('API connection test failed:', error);
        setApiStatus('offline');
      }
    };

    testApiConnection();
  }, [apiUrl]);

  const handleQuerySubmit = async (query: string) => {
    if (!query.trim()) {
      setError('Please enter a job description or requirements');
      return;
    }

    setIsLoading(true);
    setError(null);
    setCurrentQuery(query);

    try {
      const response = await fetch(`${apiUrl}/recommend`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      const data: ApiResponse = await response.json();
      setRecommendations(data.recommendations || []);
      
      if (data.recommendations.length === 0) {
        setError('No matching assessments found for your query. Try using different keywords or a more general description.');
      }
    } catch (err) {
      console.error('Error fetching recommendations:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch recommendations. Please try again.');
      setRecommendations([]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="flex-shrink-0">
                  <svg className="h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
                  </svg>
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">SHL Assessment Recommender</h1>
                  <p className="text-sm text-gray-600">Find the right assessments for your hiring needs</p>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
                  apiStatus === 'online' 
                    ? 'bg-green-100 text-green-800' 
                    : apiStatus === 'offline' 
                    ? 'bg-red-100 text-red-800'
                    : 'bg-yellow-100 text-yellow-800'
                }`}>
                  <div className={`w-2 h-2 rounded-full ${
                    apiStatus === 'online' 
                      ? 'bg-green-500' 
                      : apiStatus === 'offline' 
                      ? 'bg-red-500'
                      : 'bg-yellow-500'
                  }`}></div>
                  <span className="font-medium">
                    {apiStatus === 'online' ? 'API Online' : 
                     apiStatus === 'offline' ? 'API Offline' : 
                     'Checking API...'}
                  </span>
                </div>
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
};

export default App;
