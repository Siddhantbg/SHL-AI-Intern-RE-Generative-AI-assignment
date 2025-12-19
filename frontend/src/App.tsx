import React, { useState, useEffect } from 'react';

// Types for API responses
interface AssessmentRecommendation {
  assessment_name: string;
  url: string;
  test_type: string;
  category: string;
  relevance_score: number;
  explanation: string;
  skills_matched: string[];
}

interface RecommendationResponse {
  query: string;
  recommendations: AssessmentRecommendation[];
  total_results: number;
  processing_time: number;
  query_info: {
    processing_method: string;
    confidence_score: number;
  };
  balance_info: Record<string, any>;
}

interface HealthResponse {
  status: string;
  version: string;
  uptime: number;
  assessment_count: number;
  environment: string;
}

const App: React.FC = () => {
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [query, setQuery] = useState('');
  const [recommendations, setRecommendations] = useState<AssessmentRecommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [healthData, setHealthData] = useState<HealthResponse | null>(null);
  const [maxResults, setMaxResults] = useState(5);
  
  const apiUrl = import.meta.env.VITE_API_URL || 'https://shl-ai-intern-re-generative-ai-assignment.onrender.com';

  // Check API health on component mount
  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await fetch(`${apiUrl}/health`);
      if (response.ok) {
        const data: HealthResponse = await response.json();
        setHealthData(data);
        setApiStatus('online');
      } else {
        setApiStatus('offline');
      }
    } catch (error) {
      console.error('API health check failed:', error);
      setApiStatus('offline');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!query.trim()) {
      setError('Please enter a job description or query');
      return;
    }

    setLoading(true);
    setError(null);
    setRecommendations([]);

    try {
      const response = await fetch(`${apiUrl}/recommend`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          max_results: maxResults,
          balance_domains: true
        }),
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      const data: RecommendationResponse = await response.json();
      setRecommendations(data.recommendations);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get recommendations');
    } finally {
      setLoading(false);
    }
  };

  const exampleQueries = [
    "Software Developer with Python and JavaScript skills",
    "Sales Manager with leadership and communication skills",
    "Data Analyst with statistical analysis and Excel skills",
    "Customer Service Representative with problem-solving abilities",
    "Project Manager with organizational and planning skills"
  ];

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            🎯 SHL Assessment Recommender
          </h1>
          <p className="text-xl text-gray-600 mb-6">
            Find the right assessments for your hiring needs
          </p>
          
          {/* System Status */}
          <div className="bg-white rounded-lg shadow-sm border p-4 mb-6 max-w-2xl mx-auto">
            <h2 className="text-lg font-semibold text-gray-800 mb-3">🚀 System Status</h2>
            <div className="space-y-2 text-sm">
              <div className="flex items-center justify-center space-x-2">
                <span className="text-green-600">✅ Frontend: React app loaded successfully</span>
              </div>
              <div className="flex items-center justify-center space-x-2">
                <span className={apiStatus === 'online' ? 'text-green-600' : 'text-red-600'}>
                  {apiStatus === 'online' ? '✅' : '❌'} Backend: {apiStatus}
                </span>
              </div>
              {healthData && (
                <div className="text-gray-600 text-xs">
                  Version: {healthData.version} | Assessments: {healthData.assessment_count} | 
                  Uptime: {Math.round(healthData.uptime)}s
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Query Input Section */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-sm border p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">
                📝 Enter Job Description
              </h2>
              
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
                    Job Description or Query
                  </label>
                  <textarea
                    id="query"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Enter a job description, required skills, or role requirements..."
                    className="w-full h-32 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                    disabled={loading || apiStatus !== 'online'}
                  />
                </div>

                <div>
                  <label htmlFor="maxResults" className="block text-sm font-medium text-gray-700 mb-2">
                    Max Results: {maxResults}
                  </label>
                  <input
                    type="range"
                    id="maxResults"
                    min="1"
                    max="10"
                    value={maxResults}
                    onChange={(e) => setMaxResults(parseInt(e.target.value))}
                    className="w-full"
                    disabled={loading}
                  />
                </div>

                <button
                  type="submit"
                  disabled={loading || apiStatus !== 'online' || !query.trim()}
                  className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                >
                  {loading ? '🔄 Getting Recommendations...' : '🎯 Get Recommendations'}
                </button>
              </form>

              {/* Example Queries */}
              <div className="mt-6">
                <h3 className="text-sm font-medium text-gray-700 mb-3">💡 Example Queries:</h3>
                <div className="space-y-2">
                  {exampleQueries.map((example, index) => (
                    <button
                      key={index}
                      onClick={() => setQuery(example)}
                      className="w-full text-left text-xs text-blue-600 hover:text-blue-800 hover:bg-blue-50 p-2 rounded border border-blue-200 transition-colors"
                      disabled={loading}
                    >
                      {example}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-sm border p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">
                📊 Recommended Assessments
              </h2>

              {error && (
                <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-4">
                  <div className="flex">
                    <div className="text-red-400">❌</div>
                    <div className="ml-3">
                      <h3 className="text-sm font-medium text-red-800">Error</h3>
                      <p className="text-sm text-red-700 mt-1">{error}</p>
                    </div>
                  </div>
                </div>
              )}

              {loading && (
                <div className="text-center py-8">
                  <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                  <p className="mt-2 text-gray-600">Processing your query...</p>
                </div>
              )}

              {recommendations.length > 0 && !loading && (
                <div className="space-y-4">
                  <div className="text-sm text-gray-600 mb-4">
                    Found {recommendations.length} recommended assessments
                  </div>
                  
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Assessment
                          </th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Type
                          </th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Relevance
                          </th>
                          <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Skills
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {recommendations.map((rec, index) => (
                          <tr key={index} className="hover:bg-gray-50">
                            <td className="px-4 py-4">
                              <div>
                                <a
                                  href={rec.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-blue-600 hover:text-blue-800 font-medium hover:underline"
                                >
                                  {rec.assessment_name}
                                </a>
                                <p className="text-xs text-gray-500 mt-1">{rec.explanation}</p>
                              </div>
                            </td>
                            <td className="px-4 py-4">
                              <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                                rec.test_type === 'K' 
                                  ? 'bg-blue-100 text-blue-800' 
                                  : 'bg-green-100 text-green-800'
                              }`}>
                                {rec.test_type === 'K' ? 'Knowledge' : 'Personality'}
                              </span>
                              <div className="text-xs text-gray-500 mt-1">{rec.category}</div>
                            </td>
                            <td className="px-4 py-4">
                              <div className="flex items-center">
                                <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                                  <div 
                                    className="bg-blue-600 h-2 rounded-full" 
                                    style={{ width: `${rec.relevance_score * 100}%` }}
                                  ></div>
                                </div>
                                <span className="text-sm text-gray-600">
                                  {Math.round(rec.relevance_score * 100)}%
                                </span>
                              </div>
                            </td>
                            <td className="px-4 py-4">
                              <div className="flex flex-wrap gap-1">
                                {rec.skills_matched.slice(0, 3).map((skill, skillIndex) => (
                                  <span
                                    key={skillIndex}
                                    className="inline-flex px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded"
                                  >
                                    {skill}
                                  </span>
                                ))}
                                {rec.skills_matched.length > 3 && (
                                  <span className="text-xs text-gray-500">
                                    +{rec.skills_matched.length - 3} more
                                  </span>
                                )}
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {!loading && !error && recommendations.length === 0 && query && (
                <div className="text-center py-8 text-gray-500">
                  <div className="text-4xl mb-2">🔍</div>
                  <p>No recommendations found. Try a different query.</p>
                </div>
              )}

              {!loading && !error && recommendations.length === 0 && !query && (
                <div className="text-center py-8 text-gray-500">
                  <div className="text-4xl mb-2">👋</div>
                  <p>Enter a job description to get started!</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
