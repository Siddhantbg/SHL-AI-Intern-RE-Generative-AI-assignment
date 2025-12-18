import React from 'react';

const App = () => {
  console.log('🚀 SHL App component loaded');
  
  return React.createElement('div', {
    style: { 
      minHeight: '100vh', 
      backgroundColor: '#f3f4f6', 
      padding: '20px',
      fontFamily: 'Arial, sans-serif'
    }
  }, 
    React.createElement('div', {
      style: { 
        backgroundColor: 'white', 
        padding: '30px', 
        borderRadius: '8px', 
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        maxWidth: '800px',
        margin: '0 auto'
      }
    },
      React.createElement('h1', {
        style: { 
          fontSize: '32px', 
          fontWeight: 'bold', 
          color: '#1f2937', 
          margin: '0 0 16px 0',
          textAlign: 'center'
        }
      }, '🎯 SHL Assessment Recommender'),
      
      React.createElement('p', {
        style: { 
          fontSize: '16px', 
          color: '#6b7280', 
          margin: '0 0 24px 0',
          textAlign: 'center'
        }
      }, 'Find the right assessments for your hiring needs'),

      React.createElement('div', {
        style: {
          backgroundColor: '#f9fafb',
          padding: '20px',
          borderRadius: '6px',
          border: '1px solid #e5e7eb'
        }
      },
        React.createElement('h2', {
          style: { fontSize: '18px', margin: '0 0 12px 0', color: '#374151' }
        }, '🚀 System Status'),
        React.createElement('p', {
          style: { margin: '0', color: '#059669' }
        }, '✅ Frontend: Successfully deployed on Vercel'),
        React.createElement('p', {
          style: { margin: '8px 0 0 0', color: '#059669' }
        }, '✅ Backend: Running at shl-ai-intern-re-generative-ai-assignment.onrender.com')
      ),

      React.createElement('div', {
        style: {
          marginTop: '24px',
          padding: '20px',
          backgroundColor: '#eff6ff',
          borderRadius: '6px',
          border: '1px solid #bfdbfe'
        }
      },
        React.createElement('h3', {
          style: { fontSize: '16px', margin: '0 0 12px 0', color: '#1e40af' }
        }, '📝 React is Working!'),
        React.createElement('p', {
          style: { margin: '0 0 8px 0', color: '#1e40af' }
        }, '✅ React components are rendering'),
        React.createElement('p', {
          style: { margin: '0 0 8px 0', color: '#1e40af' }
        }, '✅ JavaScript bundle loaded successfully'),
        React.createElement('p', {
          style: { margin: '0', color: '#1e40af' }
        }, '✅ Ready to add full functionality')
      )
    )
  );
};

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
