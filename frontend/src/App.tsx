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
    <div className="min-h-screen" style={{ backgroundColor: '#F3F6FA' }}>
      {/* Enterprise Navbar */}
      <nav className="bg-white" style={{ borderBottom: '1px solid #E2E8F0' }}>
        <div className="max-w-6xl mx-auto px-4">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-semibold" style={{ color: '#0B2A4A' }}>
                SHL Assessment Recommender
              </h1>
            </div>
            <div className="flex space-x-8">
              <a 
                href="#" 
                className="text-sm font-medium" 
                style={{ color: '#475569' }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = '#2563EB';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = '#475569';
                }}
              >
                Home
              </a>
              <a 
                href="#" 
                className="text-sm font-medium" 
                style={{ color: '#475569' }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = '#2563EB';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = '#475569';
                }}
              >
                Assessments
              </a>
              <a 
                href="#" 
                className="text-sm font-medium" 
                style={{ color: '#475569' }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = '#2563EB';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = '#475569';
                }}
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
            <h1 className="text-4xl font-bold mb-4" style={{ color: '#0F172A' }}>
              Assessment Recommendations
            </h1>
            <p className="text-xl mb-6" style={{ color: '#475569' }}>
              Find the right assessments for your hiring requirements
            </p>
            
            {/* System Status */}
            <div className="bg-white rounded-lg shadow-sm p-6 mb-6 max-w-2xl mx-auto">
              <h2 className="text-lg font-semibold mb-4" style={{ color: '#0B2A4A' }}>System Status</h2>
              <div className="space-y-3 text-sm">
                <div className="flex items-center justify-center space-x-2">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#15803D' }}></div>
                  <span style={{ color: '#0F172A' }}>Frontend: Application loaded</span>
                </div>
                <div className="flex items-center justify-center space-x-2">
                  <div 
                    className="w-2 h-2 rounded-full" 
                    style={{ backgroundColor: apiStatus === 'online' ? '#15803D' : '#DC2626' }}
                  ></div>
                  <span style={{ color: '#0F172A' }}>
                    Backend: {apiStatus === 'online' ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
                {healthData && (
                  <div style={{ color: '#64748B' }} className="text-xs">
                    Version {healthData.version} | {healthData.assessment_count} assessments | 
                    Uptime {Math.round(healthData.uptime)}s
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Query Input Section */}
            <div className="lg:col-span-1">
              <div className="bg-white rounded-lg shadow-sm p-6">
                <h2 className="text-xl font-semibold mb-4" style={{ color: '#0B2A4A' }}>
                  Job Description
                </h2>
                
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div>
                    <label htmlFor="query" className="block text-sm font-medium mb-2" style={{ color: '#0F172A' }}>
                      Enter job requirements
                    </label>
                    <textarea
                      id="query"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Describe the role, required skills, and qualifications..."
                      className="w-full h-32 px-3 py-2 rounded-lg resize-none"
                      style={{ 
                        border: '1px solid #CBD5E1',
                        color: '#0F172A'
                      }}
                      onFocus={(e) => {
                        e.target.style.borderColor = '#2563EB';
                        e.target.style.outline = 'none';
                      }}
                      onBlur={(e) => {
                        e.target.style.borderColor = '#CBD5E1';
                      }}
                      disabled={loading || apiStatus !== 'online'}
                    />
                  </div>

                  <div>
                    <label htmlFor="maxResults" className="block text-sm font-medium mb-2" style={{ color: '#0F172A' }}>
                      Maximum results: {maxResults}
                    </label>
                    <input
                      type="range"
                      id="maxResults"
                      min="1"
                      max="10"
                      value={maxResults}
                      onChange={(e) => setMaxResults(parseInt(e.target.value))}
                      className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                      style={{
                        background: `linear-gradient(to right, #2563EB 0%, #2563EB ${(maxResults - 1) * 11.11}%, #E5E7EB ${(maxResults - 1) * 11.11}%, #E5E7EB 100%)`
                      }}
                      disabled={loading}
                    />
                  </div>

                  <button
                    type="submit"
                    disabled={loading || apiStatus !== 'online' || !query.trim()}
                    className="w-full text-white py-3 px-4 font-medium"
                    style={{
                      backgroundColor: loading || apiStatus !== 'online' || !query.trim() ? '#CBD5E1' : '#2563EB',
                      borderRadius: '10px',
                      color: loading || apiStatus !== 'online' || !query.trim() ? '#64748B' : '#FFFFFF'
                    }}
                    onMouseEnter={(e) => {
                      if (!loading && apiStatus === 'online' && query.trim()) {
                        e.currentTarget.style.backgroundColor = '#1D4ED8';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!loading && apiStatus === 'online' && query.trim()) {
                        e.currentTarget.style.backgroundColor = '#2563EB';
                      }
                    }}
                  >
                    {loading ? 'Processing...' : 'Get Recommendations'}
                  </button>
                </form>

                {/* Example Queries */}
                <div className="mt-6">
                  <h3 className="text-sm font-medium mb-3" style={{ color: '#64748B' }}>Sample job descriptions:</h3>
                  <div className="space-y-2">
                    {exampleQueries.map((example, index) => (
                      <button
                        key={index}
                        onClick={() => setQuery(example)}
                        className="w-full text-left text-sm p-3 rounded-lg"
                        style={{
                          color: '#2563EB',
                          border: '1px solid #E2E8F0',
                          backgroundColor: '#FFFFFF'
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = '#F8FAFC';
                          e.currentTarget.style.borderColor = '#2563EB';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = '#FFFFFF';
                          e.currentTarget.style.borderColor = '#E2E8F0';
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
              <div className="bg-white rounded-lg shadow-sm p-6">
                <h2 className="text-xl font-semibold mb-4" style={{ color: '#0B2A4A' }}>
                  Assessment Recommendations
                </h2>

                {error && (
                  <div className="rounded-lg p-4 mb-4" style={{ backgroundColor: '#FEF2F2', border: '1px solid #FECACA' }}>
                    <div className="flex">
                      <div className="flex-shrink-0">
                        <div className="w-2 h-2 rounded-full mt-2" style={{ backgroundColor: '#DC2626' }}></div>
                      </div>
                      <div className="ml-3">
                        <h3 className="text-sm font-medium" style={{ color: '#991B1B' }}>Error</h3>
                        <p className="text-sm mt-1" style={{ color: '#B91C1C' }}>{error}</p>
                      </div>
                    </div>
                  </div>
                )}

                {loading && (
                  <div className="text-center py-12">
                    <div className="inline-block animate-spin rounded-full h-8 w-8 border-2 border-gray-200" style={{ borderTopColor: '#2563EB' }}></div>
                    <p className="mt-3" style={{ color: '#475569' }}>Processing request...</p>
                  </div>
                )}

                {recommendations.length > 0 && !loading && (
                  <div className="space-y-4">
                    <div className="text-sm mb-4" style={{ color: '#475569' }}>
                      {recommendations.length} assessment{recommendations.length !== 1 ? 's' : ''} recommended
                    </div>
                    
                    <div className="overflow-x-auto">
                      <table className="min-w-full" style={{ borderCollapse: 'separate', borderSpacing: 0 }}>
                        <thead>
                          <tr style={{ backgroundColor: '#F8FAFC' }}>
                            <th className="px-6 py-4 text-left text-xs font-medium uppercase tracking-wider" style={{ color: '#475569', borderBottom: '1px solid #E2E8F0' }}>
                              Assessment
                            </th>
                            <th className="px-6 py-4 text-left text-xs font-medium uppercase tracking-wider" style={{ color: '#475569', borderBottom: '1px solid #E2E8F0' }}>
                              Type
                            </th>
                            <th className="px-6 py-4 text-left text-xs font-medium uppercase tracking-wider" style={{ color: '#475569', borderBottom: '1px solid #E2E8F0' }}>
                              Match
                            </th>
                            <th className="px-6 py-4 text-left text-xs font-medium uppercase tracking-wider" style={{ color: '#475569', borderBottom: '1px solid #E2E8F0' }}>
                              Skills
                            </th>
                          </tr>
                        </thead>
                        <tbody className="bg-white">
                          {recommendations.map((rec, index) => (
                            <tr 
                              key={index}
                              style={{ borderBottom: index < recommendations.length - 1 ? '1px solid #F1F5F9' : 'none' }}
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
                                    className="font-medium"
                                    style={{ color: '#2563EB' }}
                                    onMouseEnter={(e) => {
                                      e.currentTarget.style.color = '#1D4ED8';
                                    }}
                                    onMouseLeave={(e) => {
                                      e.currentTarget.style.color = '#2563EB';
                                    }}
                                  >
                                    {rec.assessment_name}
                                  </a>
                                  <p className="text-xs mt-1" style={{ color: '#64748B' }}>{rec.explanation}</p>
                                </div>
                              </td>
                              <td className="px-6 py-4">
                                <span 
                                  className="inline-flex px-3 py-1 text-xs font-medium rounded-full"
                                  style={{
                                    backgroundColor: rec.test_type === 'K' ? '#EBF4FF' : '#DCFCE7',
                                    color: rec.test_type === 'K' ? '#0B2A4A' : '#15803D'
                                  }}
                                >
                                  {rec.test_type === 'K' ? 'Knowledge' : 'Personality'}
                                </span>
                                <div className="text-xs mt-1" style={{ color: '#64748B' }}>{rec.category}</div>
                              </td>
                              <td className="px-6 py-4">
                                <div className="flex items-center">
                                  <div className="w-20 rounded-full h-2 mr-3" style={{ backgroundColor: '#E5E7EB' }}>
                                    <div 
                                      className="h-2 rounded-full" 
                                      style={{ 
                                        width: `${rec.relevance_score * 100}%`,
                                        backgroundColor: '#2563EB'
                                      }}
                                    ></div>
                                  </div>
                                  <span className="text-sm font-medium" style={{ color: '#0F172A' }}>
                                    {Math.round(rec.relevance_score * 100)}%
                                  </span>
                                </div>
                              </td>
                              <td className="px-6 py-4">
                                <div className="flex flex-wrap gap-1">
                                  {rec.skills_matched.slice(0, 3).map((skill, skillIndex) => (
                                    <span
                                      key={skillIndex}
                                      className="inline-flex px-2 py-1 text-xs rounded"
                                      style={{
                                        backgroundColor: '#F1F5F9',
                                        color: '#475569'
                                      }}
                                    >
                                      {skill}
                                    </span>
                                  ))}
                                  {rec.skills_matched.length > 3 && (
                                    <span className="text-xs" style={{ color: '#64748B' }}>
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
                  <div className="text-center py-12" style={{ color: '#64748B' }}>
                    <p className="text-lg">No matching assessments found</p>
                    <p className="text-sm mt-2">Try adjusting your job description or requirements</p>
                  </div>
                )}

                {!loading && !error && recommendations.length === 0 && !query && (
                  <div className="text-center py-12" style={{ color: '#64748B' }}>
                    <p className="text-lg">Ready to find assessments</p>
                    <p className="text-sm mt-2">Enter a job description to get personalized recommendations</p>
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
