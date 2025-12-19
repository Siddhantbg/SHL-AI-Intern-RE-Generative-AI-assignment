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
    <div className="min-h-screen" style={{ backgroundColor: '#F5F7FA' }}>
      {/* Professional Navbar */}
      <nav className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-6xl mx-auto px-4">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-semibold" style={{ color: '#1F3A5F' }}>
                SHL Assessment Recommender
              </h1>
            </div>
            <div className="flex space-x-8">
              <a 
                href="#" 
                className="text-sm font-medium transition-colors duration-200 hover:text-blue-600" 
                style={{ color: '#6B7280' }}
              >
                Home
              </a>
              <a 
                href="#" 
                className="text-sm font-medium transition-colors duration-200 hover:text-blue-600" 
                style={{ color: '#6B7280' }}
              >
                Assessments
              </a>
              <a 
                href="#" 
                className="text-sm font-medium transition-colors duration-200 hover:text-blue-600" 
                style={{ color: '#6B7280' }}
              >
                About
              </a>
            </div>
          </div>
        </div>
      </nav>

      <div className="py-8 px-4">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold mb-4" style={{ color: '#111827' }}>
              Find the Right Assessments
            </h1>
            <p className="text-xl mb-6" style={{ color: '#6B7280' }}>
              Intelligent recommendations for your hiring needs
            </p>
            
            {/* System Status */}
            <div className="bg-white rounded-xl shadow-md p-6 mb-6 max-w-2xl mx-auto">
              <h2 className="text-lg font-semibold mb-4" style={{ color: '#1F3A5F' }}>System Status</h2>
              <div className="space-y-3 text-sm">
                <div className="flex items-center justify-center space-x-2">
                  <span style={{ color: '#22C55E' }}>✅ Frontend: React app loaded successfully</span>
                </div>
                <div className="flex items-center justify-center space-x-2">
                  <span style={{ color: apiStatus === 'online' ? '#22C55E' : '#EF4444' }}>
                    {apiStatus === 'online' ? '✅' : '❌'} Backend: {apiStatus}
                  </span>
                </div>
                {healthData && (
                  <div style={{ color: '#6B7280' }} className="text-xs">
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
              <div className="bg-white rounded-xl shadow-md p-6">
                <h2 className="text-xl font-semibold mb-4" style={{ color: '#1F3A5F' }}>
                  Enter Job Description
                </h2>
                
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div>
                    <label htmlFor="query" className="block text-sm font-medium mb-2" style={{ color: '#111827' }}>
                      Job Description or Query
                    </label>
                    <textarea
                      id="query"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Enter a job description, required skills, or role requirements..."
                      className="w-full h-32 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:border-transparent resize-none transition-colors"
                      style={{ 
                        focusRingColor: '#3B82F6',
                        borderColor: '#D1D5DB'
                      }}
                      onFocus={(e) => {
                        e.target.style.borderColor = '#3B82F6';
                        e.target.style.boxShadow = '0 0 0 2px rgba(59, 130, 246, 0.2)';
                      }}
                      onBlur={(e) => {
                        e.target.style.borderColor = '#D1D5DB';
                        e.target.style.boxShadow = 'none';
                      }}
                      disabled={loading || apiStatus !== 'online'}
                    />
                  </div>

                  <div>
                    <label htmlFor="maxResults" className="block text-sm font-medium mb-2" style={{ color: '#111827' }}>
                      Max Results: {maxResults}
                    </label>
                    <input
                      type="range"
                      id="maxResults"
                      min="1"
                      max="10"
                      value={maxResults}
                      onChange={(e) => setMaxResults(parseInt(e.target.value))}
                      className="w-full accent-blue-600"
                      disabled={loading}
                    />
                  </div>

                  <button
                    type="submit"
                    disabled={loading || apiStatus !== 'online' || !query.trim()}
                    className="w-full text-white py-3 px-4 rounded-lg font-medium focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:cursor-not-allowed transition-colors"
                    style={{
                      backgroundColor: loading || apiStatus !== 'online' || !query.trim() ? '#9CA3AF' : '#3B82F6',
                      focusRingColor: '#3B82F6'
                    }}
                    onMouseEnter={(e) => {
                      if (!loading && apiStatus === 'online' && query.trim()) {
                        e.currentTarget.style.backgroundColor = '#2563EB';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!loading && apiStatus === 'online' && query.trim()) {
                        e.currentTarget.style.backgroundColor = '#3B82F6';
                      }
                    }}
                  >
                    {loading ? 'Getting Recommendations...' : 'Get Recommendations'}
                  </button>
                </form>

                {/* Example Queries */}
                <div className="mt-6">
                  <h3 className="text-sm font-medium mb-3" style={{ color: '#6B7280' }}>Example Queries:</h3>
                  <div className="space-y-2">
                    {exampleQueries.map((example, index) => (
                      <button
                        key={index}
                        onClick={() => setQuery(example)}
                        className="w-full text-left text-xs p-3 rounded-lg border transition-colors"
                        style={{
                          color: '#3B82F6',
                          borderColor: '#E5E7EB',
                          backgroundColor: '#FFFFFF'
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = '#F8FAFC';
                          e.currentTarget.style.borderColor = '#3B82F6';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = '#FFFFFF';
                          e.currentTarget.style.borderColor = '#E5E7EB';
                        }}
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
              <div className="bg-white rounded-xl shadow-md p-6">
                <h2 className="text-xl font-semibold mb-4" style={{ color: '#1F3A5F' }}>
                  Recommended Assessments
                </h2>

                {error && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
                    <div className="flex">
                      <div className="text-red-400">⚠️</div>
                      <div className="ml-3">
                        <h3 className="text-sm font-medium text-red-800">Error</h3>
                        <p className="text-sm text-red-700 mt-1">{error}</p>
                      </div>
                    </div>
                  </div>
                )}

                {loading && (
                  <div className="text-center py-12">
                    <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2" style={{ borderColor: '#3B82F6' }}></div>
                    <p className="mt-3" style={{ color: '#6B7280' }}>Processing your query...</p>
                  </div>
                )}

                {recommendations.length > 0 && !loading && (
                  <div className="space-y-4">
                    <div className="text-sm mb-4" style={{ color: '#6B7280' }}>
                      Found {recommendations.length} recommended assessments
                    </div>
                    
                    <div className="overflow-x-auto">
                      <table className="min-w-full divide-y" style={{ borderColor: '#E5E7EB' }}>
                        <thead style={{ backgroundColor: '#F9FAFB' }}>
                          <tr>
                            <th className="px-6 py-4 text-left text-xs font-medium uppercase tracking-wider" style={{ color: '#6B7280' }}>
                              Assessment
                            </th>
                            <th className="px-6 py-4 text-left text-xs font-medium uppercase tracking-wider" style={{ color: '#6B7280' }}>
                              Type
                            </th>
                            <th className="px-6 py-4 text-left text-xs font-medium uppercase tracking-wider" style={{ color: '#6B7280' }}>
                              Relevance
                            </th>
                            <th className="px-6 py-4 text-left text-xs font-medium uppercase tracking-wider" style={{ color: '#6B7280' }}>
                              Skills
                            </th>
                          </tr>
                        </thead>
                        <tbody className="bg-white divide-y" style={{ borderColor: '#F3F4F6' }}>
                          {recommendations.map((rec, index) => (
                            <tr 
                              key={index} 
                              className="transition-colors"
                              onMouseEnter={(e) => {
                                e.currentTarget.style.backgroundColor = '#F8FAFC';
                              }}
                              onMouseLeave={(e) => {
                                e.currentTarget.style.backgroundColor = '#FFFFFF';
                              }}
                            >
                              <td className="px-6 py-4">
                                <div>
                                  <a
                                    href={rec.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="font-medium hover:underline transition-colors"
                                    style={{ color: '#3B82F6' }}
                                    onMouseEnter={(e) => {
                                      e.currentTarget.style.color = '#1D4ED8';
                                    }}
                                    onMouseLeave={(e) => {
                                      e.currentTarget.style.color = '#3B82F6';
                                    }}
                                  >
                                    {rec.assessment_name}
                                  </a>
                                  <p className="text-xs mt-1" style={{ color: '#6B7280' }}>{rec.explanation}</p>
                                </div>
                              </td>
                              <td className="px-6 py-4">
                                <span 
                                  className="inline-flex px-3 py-1 text-xs font-medium rounded-full"
                                  style={{
                                    backgroundColor: rec.test_type === 'K' ? '#EBF4FF' : '#F0FDF4',
                                    color: rec.test_type === 'K' ? '#1F3A5F' : '#22C55E'
                                  }}
                                >
                                  {rec.test_type === 'K' ? 'Knowledge' : 'Personality'}
                                </span>
                                <div className="text-xs mt-1" style={{ color: '#6B7280' }}>{rec.category}</div>
                              </td>
                              <td className="px-6 py-4">
                                <div className="flex items-center">
                                  <div className="w-20 rounded-full h-2 mr-3" style={{ backgroundColor: '#E5E7EB' }}>
                                    <div 
                                      className="h-2 rounded-full transition-all duration-300" 
                                      style={{ 
                                        width: `${rec.relevance_score * 100}%`,
                                        backgroundColor: '#3B82F6'
                                      }}
                                    ></div>
                                  </div>
                                  <span className="text-sm font-medium" style={{ color: '#111827' }}>
                                    {Math.round(rec.relevance_score * 100)}%
                                  </span>
                                </div>
                              </td>
                              <td className="px-6 py-4">
                                <div className="flex flex-wrap gap-1">
                                  {rec.skills_matched.slice(0, 3).map((skill, skillIndex) => (
                                    <span
                                      key={skillIndex}
                                      className="inline-flex px-2 py-1 text-xs rounded-md"
                                      style={{
                                        backgroundColor: '#F3F4F6',
                                        color: '#374151'
                                      }}
                                    >
                                      {skill}
                                    </span>
                                  ))}
                                  {rec.skills_matched.length > 3 && (
                                    <span className="text-xs" style={{ color: '#6B7280' }}>
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
                  <div className="text-center py-12" style={{ color: '#6B7280' }}>
                    <div className="text-4xl mb-3">🔍</div>
                    <p>No recommendations found. Try a different query.</p>
                  </div>
                )}

                {!loading && !error && recommendations.length === 0 && !query && (
                  <div className="text-center py-12" style={{ color: '#6B7280' }}>
                    <div className="text-4xl mb-3">👋</div>
                    <p>Enter a job description to get started!</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
