import React, { useState, useEffect, useRef } from 'react';

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
  const [searchHistory, setSearchHistory] = useState<string[]>([]);
  const [showFilters, setShowFilters] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedTestType, setSelectedTestType] = useState<string>('all');
  const resultsRef = useRef<HTMLDivElement>(null);
  
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
      
      // Add to search history
      if (query.trim() && !searchHistory.includes(query.trim())) {
        setSearchHistory(prev => [query.trim(), ...prev.slice(0, 4)]);
      }
      
      // Scroll to results
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
      
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

  // Filter recommendations based on selected filters
  const filteredRecommendations = recommendations.filter(rec => {
    const typeMatch = selectedTestType === 'all' || rec.test_type === selectedTestType;
    const categoryMatch = selectedCategory === 'all' || rec.category.toLowerCase().includes(selectedCategory.toLowerCase());
    return typeMatch && categoryMatch;
  });

  return (
    <div 
      className="min-h-screen" 
      style={{ 
        background: `
          radial-gradient(ellipse at top right, rgba(179, 107, 255, 0.15) 0%, transparent 50%),
          radial-gradient(ellipse at bottom left, rgba(124, 124, 255, 0.1) 0%, transparent 50%),
          #0B0E1A
        `
      }}
    >
      {/* Dark Navbar */}
      <nav style={{ 
        backgroundColor: 'rgba(15, 19, 37, 0.8)', 
        backdropFilter: 'blur(10px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.06)' 
      }}>
        <div className="max-w-6xl mx-auto px-4">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-semibold" style={{ color: '#E8EBF3' }}>
                SHL Assessment Recommender
              </h1>
            </div>
            <div className="flex space-x-8">
              <a 
                href="#" 
                className="text-sm font-medium" 
                style={{ color: '#A6ADC8' }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = '#7C7CFF';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = '#A6ADC8';
                }}
              >
                Home
              </a>
              <a 
                href="#" 
                className="text-sm font-medium" 
                style={{ color: '#A6ADC8' }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = '#7C7CFF';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = '#A6ADC8';
                }}
              >
                Assessments
              </a>
              <a 
                href="#" 
                className="text-sm font-medium" 
                style={{ color: '#A6ADC8' }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = '#7C7CFF';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = '#A6ADC8';
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
            <h1 className="text-4xl font-bold mb-4" style={{ color: '#E8EBF3' }}>
              Assessment Recommendations
            </h1>
            <p className="text-xl mb-6" style={{ color: '#A6ADC8' }}>
              Find the right assessments for your hiring requirements
            </p>
            
            {/* System Status */}
            <div 
              className="rounded-xl p-6 mb-6 max-w-2xl mx-auto" 
              style={{ 
                backgroundColor: '#12162A',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
                border: '1px solid rgba(255, 255, 255, 0.06)'
              }}
            >
              <h2 className="text-lg font-semibold mb-4" style={{ color: '#E8EBF3' }}>System Status</h2>
              <div className="space-y-3 text-sm">
                <div className="flex items-center justify-center space-x-2">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#2EE59D' }}></div>
                  <span style={{ color: '#E8EBF3' }}>Frontend: Application loaded</span>
                </div>
                <div className="flex items-center justify-center space-x-2">
                  <div 
                    className="w-2 h-2 rounded-full" 
                    style={{ backgroundColor: apiStatus === 'online' ? '#2EE59D' : '#DC2626' }}
                  ></div>
                  <span style={{ color: '#E8EBF3' }}>
                    Backend: {apiStatus === 'online' ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
                {healthData && (
                  <div style={{ color: '#7C84A3' }} className="text-xs">
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
              <div 
                className="rounded-xl p-6" 
                style={{ 
                  backgroundColor: '#12162A',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
                  border: '1px solid rgba(255, 255, 255, 0.06)'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#181E36';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = '#12162A';
                }}
              >
                <h2 className="text-xl font-semibold mb-4" style={{ color: '#E8EBF3' }}>
                  Job Description
                </h2>
                
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div>
                    <label htmlFor="query" className="block text-sm font-medium mb-2" style={{ color: '#A6ADC8' }}>
                      Enter job requirements
                    </label>
                    <textarea
                      id="query"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Describe the role, required skills, and qualifications..."
                      className="w-full h-32 px-3 py-2 rounded-lg resize-none"
                      style={{ 
                        backgroundColor: '#0F1325',
                        border: '1px solid rgba(255, 255, 255, 0.06)',
                        color: '#E8EBF3'
                      }}
                      onFocus={(e) => {
                        e.target.style.borderColor = '#7C7CFF';
                        e.target.style.outline = 'none';
                        e.target.style.boxShadow = '0 0 0 2px rgba(124, 124, 255, 0.2)';
                      }}
                      onBlur={(e) => {
                        e.target.style.borderColor = 'rgba(255, 255, 255, 0.06)';
                        e.target.style.boxShadow = 'none';
                      }}
                      disabled={loading || apiStatus !== 'online'}
                    />
                  </div>

                  <div className="space-y-4">
                    <div>
                      <label htmlFor="maxResults" className="block text-sm font-medium mb-2" style={{ color: '#A6ADC8' }}>
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
                          background: `linear-gradient(to right, #7C7CFF 0%, #7C7CFF ${(maxResults - 1) * 11.11}%, rgba(255, 255, 255, 0.06) ${(maxResults - 1) * 11.11}%, rgba(255, 255, 255, 0.06) 100%)`
                        }}
                        disabled={loading}
                      />
                    </div>

                    {/* Advanced Filters Toggle */}
                    <button
                      type="button"
                      onClick={() => setShowFilters(!showFilters)}
                      className="w-full text-left text-sm p-3 rounded-lg transition-all duration-200 flex items-center justify-between"
                      style={{
                        color: '#7C7CFF',
                        border: '1px solid rgba(255, 255, 255, 0.06)',
                        backgroundColor: '#0F1325'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = '#181E36';
                        e.currentTarget.style.borderColor = '#7C7CFF';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = '#0F1325';
                        e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.06)';
                      }}
                    >
                      <span>Advanced Filters</span>
                      <span style={{ transform: showFilters ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}>
                        ▼
                      </span>
                    </button>

                    {/* Filters Panel */}
                    {showFilters && (
                      <div 
                        className="space-y-3 p-4 rounded-lg"
                        style={{
                          backgroundColor: 'rgba(15, 19, 37, 0.5)',
                          border: '1px solid rgba(255, 255, 255, 0.06)'
                        }}
                      >
                        <div>
                          <label className="block text-xs font-medium mb-2" style={{ color: '#A6ADC8' }}>
                            Test Type
                          </label>
                          <select
                            value={selectedTestType}
                            onChange={(e) => setSelectedTestType(e.target.value)}
                            className="w-full px-3 py-2 rounded-lg text-sm"
                            style={{
                              backgroundColor: '#0F1325',
                              border: '1px solid rgba(255, 255, 255, 0.06)',
                              color: '#E8EBF3'
                            }}
                          >
                            <option value="all">All Types</option>
                            <option value="K">Knowledge</option>
                            <option value="P">Personality</option>
                          </select>
                        </div>
                        
                        <div>
                          <label className="block text-xs font-medium mb-2" style={{ color: '#A6ADC8' }}>
                            Category
                          </label>
                          <select
                            value={selectedCategory}
                            onChange={(e) => setSelectedCategory(e.target.value)}
                            className="w-full px-3 py-2 rounded-lg text-sm"
                            style={{
                              backgroundColor: '#0F1325',
                              border: '1px solid rgba(255, 255, 255, 0.06)',
                              color: '#E8EBF3'
                            }}
                          >
                            <option value="all">All Categories</option>
                            <option value="Technical">Technical</option>
                            <option value="Behavioral">Behavioral</option>
                            <option value="Cognitive">Cognitive</option>
                          </select>
                        </div>
                      </div>
                    )}
                  </div>

                  <button
                    type="submit"
                    disabled={loading || apiStatus !== 'online' || !query.trim()}
                    className="w-full text-white py-3 px-4 font-medium transition-all duration-200"
                    style={{
                      background: loading || apiStatus !== 'online' || !query.trim() 
                        ? '#7C84A3' 
                        : 'linear-gradient(135deg, #7C7CFF 0%, #B36BFF 100%)',
                      borderRadius: '12px',
                      boxShadow: loading || apiStatus !== 'online' || !query.trim() 
                        ? 'none' 
                        : '0 4px 15px rgba(124, 124, 255, 0.4)'
                    }}
                    onMouseEnter={(e) => {
                      if (!loading && apiStatus === 'online' && query.trim()) {
                        e.currentTarget.style.boxShadow = '0 6px 20px rgba(124, 124, 255, 0.6)';
                        e.currentTarget.style.transform = 'translateY(-1px)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!loading && apiStatus === 'online' && query.trim()) {
                        e.currentTarget.style.boxShadow = '0 4px 15px rgba(124, 124, 255, 0.4)';
                        e.currentTarget.style.transform = 'translateY(0)';
                      }
                    }}
                  >
                    {loading ? 'Processing...' : 'Get Recommendations'}
                  </button>
                </form>

                {/* Search History */}
                {searchHistory.length > 0 && (
                  <div className="mt-6">
                    <h3 className="text-sm font-medium mb-3" style={{ color: '#7C84A3' }}>Recent searches:</h3>
                    <div className="space-y-2">
                      {searchHistory.map((search, index) => (
                        <button
                          key={index}
                          onClick={() => setQuery(search)}
                          className="w-full text-left text-sm p-3 rounded-lg transition-all duration-200 flex items-center justify-between group"
                          style={{
                            color: '#A6ADC8',
                            border: '1px solid rgba(255, 255, 255, 0.06)',
                            backgroundColor: '#0F1325'
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = '#181E36';
                            e.currentTarget.style.borderColor = '#7C7CFF';
                            e.currentTarget.style.color = '#7C7CFF';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.backgroundColor = '#0F1325';
                            e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.06)';
                            e.currentTarget.style.color = '#A6ADC8';
                          }}
                          disabled={loading}
                        >
                          <span className="truncate">{search}</span>
                          <span className="text-xs opacity-0 group-hover:opacity-100 transition-opacity">
                            ↗
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Example Queries */}
                <div className="mt-6">
                  <h3 className="text-sm font-medium mb-3" style={{ color: '#7C84A3' }}>
                    {searchHistory.length > 0 ? 'Sample job descriptions:' : 'Try these examples:'}
                  </h3>
                  <div className="space-y-2">
                    {exampleQueries.map((example, index) => (
                      <button
                        key={index}
                        onClick={() => setQuery(example)}
                        className="w-full text-left text-sm p-3 rounded-lg transition-all duration-200"
                        style={{
                          color: '#7C7CFF',
                          border: '1px solid rgba(255, 255, 255, 0.06)',
                          backgroundColor: '#0F1325'
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = '#181E36';
                          e.currentTarget.style.borderColor = '#7C7CFF';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = '#0F1325';
                          e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.06)';
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
            <div className="lg:col-span-2" ref={resultsRef}>
              <div 
                className="rounded-xl p-6" 
                style={{ 
                  backgroundColor: '#12162A',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
                  border: '1px solid rgba(255, 255, 255, 0.06)'
                }}
              >
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-semibold" style={{ color: '#E8EBF3' }}>
                    Assessment Recommendations
                  </h2>
                  {recommendations.length > 0 && (
                    <div className="flex items-center space-x-3">
                      <div 
                        className="px-3 py-1 rounded-full text-sm font-medium"
                        style={{
                          background: 'linear-gradient(135deg, #7C7CFF 0%, #B36BFF 100%)',
                          color: '#FFFFFF'
                        }}
                      >
                        {filteredRecommendations.length} of {recommendations.length} results
                      </div>
                      {filteredRecommendations.length !== recommendations.length && (
                        <button
                          onClick={() => {
                            setSelectedCategory('all');
                            setSelectedTestType('all');
                          }}
                          className="text-xs px-2 py-1 rounded transition-all duration-200"
                          style={{
                            color: '#7C7CFF',
                            border: '1px solid rgba(124, 124, 255, 0.3)',
                            backgroundColor: 'rgba(124, 124, 255, 0.1)'
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = 'rgba(124, 124, 255, 0.2)';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.backgroundColor = 'rgba(124, 124, 255, 0.1)';
                          }}
                        >
                          Clear filters
                        </button>
                      )}
                    </div>
                  )}
                </div>

                {error && (
                  <div 
                    className="rounded-lg p-4 mb-4" 
                    style={{ 
                      backgroundColor: 'rgba(220, 38, 38, 0.1)', 
                      border: '1px solid rgba(220, 38, 38, 0.2)',
                      backdropFilter: 'blur(10px)'
                    }}
                  >
                    <div className="flex">
                      <div className="flex-shrink-0">
                        <div className="w-2 h-2 rounded-full mt-2" style={{ backgroundColor: '#DC2626' }}></div>
                      </div>
                      <div className="ml-3">
                        <h3 className="text-sm font-medium" style={{ color: '#FF6B6B' }}>Error</h3>
                        <p className="text-sm mt-1" style={{ color: '#FFA8A8' }}>{error}</p>
                      </div>
                    </div>
                  </div>
                )}

                {loading && (
                  <div className="text-center py-12">
                    <div className="relative">
                      <div 
                        className="inline-block animate-spin rounded-full h-12 w-12 border-2"
                        style={{ 
                          borderColor: 'rgba(124, 124, 255, 0.2)',
                          borderTopColor: '#7C7CFF'
                        }}
                      ></div>
                      <div 
                        className="absolute inset-0 rounded-full animate-pulse"
                        style={{
                          background: 'radial-gradient(circle, rgba(124, 124, 255, 0.1) 0%, transparent 70%)'
                        }}
                      ></div>
                    </div>
                    <p className="mt-4 text-lg font-medium" style={{ color: '#A6ADC8' }}>
                      Analyzing requirements...
                    </p>
                    <p className="text-sm mt-1" style={{ color: '#7C84A3' }}>
                      Finding the best assessment matches
                    </p>
                  </div>
                )}

                {filteredRecommendations.length > 0 && !loading && (
                  <div className="space-y-6">
                    {/* Results Grid */}
                    <div className="grid gap-4">
                      {filteredRecommendations.map((rec, index) => (
                        <div 
                          key={index}
                          className="rounded-lg p-6 transition-all duration-300 cursor-pointer group"
                          style={{ 
                            backgroundColor: '#0F1325',
                            border: '1px solid rgba(255, 255, 255, 0.06)',
                            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)'
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = '#181E36';
                            e.currentTarget.style.borderColor = 'rgba(124, 124, 255, 0.3)';
                            e.currentTarget.style.transform = 'translateY(-2px)';
                            e.currentTarget.style.boxShadow = '0 8px 25px rgba(124, 124, 255, 0.15)';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.backgroundColor = '#0F1325';
                            e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.06)';
                            e.currentTarget.style.transform = 'translateY(0)';
                            e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
                          }}
                        >
                          <div className="flex items-start justify-between mb-4">
                            <div className="flex-1">
                              <div className="flex items-center space-x-3 mb-2">
                                <a
                                  href={rec.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-lg font-semibold group-hover:underline"
                                  style={{ color: '#7C7CFF' }}
                                  onMouseEnter={(e) => {
                                    e.currentTarget.style.color = '#B36BFF';
                                  }}
                                  onMouseLeave={(e) => {
                                    e.currentTarget.style.color = '#7C7CFF';
                                  }}
                                >
                                  {rec.assessment_name}
                                </a>
                                <span 
                                  className="inline-flex px-3 py-1 text-xs font-medium rounded-full"
                                  style={{
                                    background: rec.test_type === 'K' 
                                      ? 'linear-gradient(135deg, rgba(124, 124, 255, 0.2) 0%, rgba(179, 107, 255, 0.2) 100%)'
                                      : 'linear-gradient(135deg, rgba(46, 229, 157, 0.2) 0%, rgba(124, 255, 124, 0.2) 100%)',
                                    color: rec.test_type === 'K' ? '#7C7CFF' : '#2EE59D',
                                    border: `1px solid ${rec.test_type === 'K' ? 'rgba(124, 124, 255, 0.3)' : 'rgba(46, 229, 157, 0.3)'}`
                                  }}
                                >
                                  {rec.test_type === 'K' ? 'Knowledge' : 'Personality'}
                                </span>
                              </div>
                              <p className="text-sm mb-3" style={{ color: '#A6ADC8' }}>
                                {rec.explanation}
                              </p>
                              <div className="text-xs" style={{ color: '#7C84A3' }}>
                                Category: {rec.category}
                              </div>
                            </div>
                            
                            <div className="ml-6 text-right">
                              <div className="mb-2">
                                <div 
                                  className="text-2xl font-bold"
                                  style={{ 
                                    background: 'linear-gradient(135deg, #7C7CFF 0%, #B36BFF 100%)',
                                    WebkitBackgroundClip: 'text',
                                    WebkitTextFillColor: 'transparent',
                                    backgroundClip: 'text'
                                  }}
                                >
                                  {Math.round(rec.relevance_score * 100)}%
                                </div>
                                <div className="text-xs" style={{ color: '#7C84A3' }}>
                                  Match Score
                                </div>
                              </div>
                              
                              {/* Circular Progress */}
                              <div className="relative w-16 h-16 mx-auto">
                                <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 64 64">
                                  <circle
                                    cx="32"
                                    cy="32"
                                    r="28"
                                    stroke="rgba(255, 255, 255, 0.06)"
                                    strokeWidth="4"
                                    fill="none"
                                  />
                                  <circle
                                    cx="32"
                                    cy="32"
                                    r="28"
                                    stroke="url(#gradient)"
                                    strokeWidth="4"
                                    fill="none"
                                    strokeLinecap="round"
                                    strokeDasharray={`${rec.relevance_score * 175.93} 175.93`}
                                    style={{
                                      transition: 'stroke-dasharray 0.5s ease-in-out'
                                    }}
                                  />
                                  <defs>
                                    <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                      <stop offset="0%" stopColor="#7C7CFF" />
                                      <stop offset="100%" stopColor="#B36BFF" />
                                    </linearGradient>
                                  </defs>
                                </svg>
                              </div>
                            </div>
                          </div>
                          
                          {/* Skills Section */}
                          <div>
                            <h4 className="text-sm font-medium mb-2" style={{ color: '#E8EBF3' }}>
                              Matched Skills
                            </h4>
                            <div className="flex flex-wrap gap-2">
                              {rec.skills_matched.map((skill, skillIndex) => (
                                <span
                                  key={skillIndex}
                                  className="inline-flex px-3 py-1 text-xs font-medium rounded-full transition-all duration-200"
                                  style={{
                                    backgroundColor: 'rgba(124, 124, 255, 0.1)',
                                    color: '#7C7CFF',
                                    border: '1px solid rgba(124, 124, 255, 0.2)'
                                  }}
                                  onMouseEnter={(e) => {
                                    e.currentTarget.style.backgroundColor = 'rgba(124, 124, 255, 0.2)';
                                    e.currentTarget.style.borderColor = 'rgba(124, 124, 255, 0.4)';
                                  }}
                                  onMouseLeave={(e) => {
                                    e.currentTarget.style.backgroundColor = 'rgba(124, 124, 255, 0.1)';
                                    e.currentTarget.style.borderColor = 'rgba(124, 124, 255, 0.2)';
                                  }}
                                >
                                  {skill}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {!loading && !error && recommendations.length === 0 && query && (
                  <div className="text-center py-16">
                    <div 
                      className="w-20 h-20 mx-auto mb-6 rounded-full flex items-center justify-center"
                      style={{
                        background: 'linear-gradient(135deg, rgba(124, 124, 255, 0.1) 0%, rgba(179, 107, 255, 0.1) 100%)',
                        border: '2px solid rgba(124, 124, 255, 0.2)'
                      }}
                    >
                      <div 
                        className="text-3xl"
                        style={{ color: '#7C7CFF' }}
                      >
                        🔍
                      </div>
                    </div>
                    <p className="text-xl font-medium mb-2" style={{ color: '#E8EBF3' }}>
                      No matching assessments found
                    </p>
                    <p className="text-sm" style={{ color: '#A6ADC8' }}>
                      Try adjusting your job description or requirements
                    </p>
                  </div>
                )}

                {!loading && !error && recommendations.length === 0 && !query && (
                  <div className="text-center py-16">
                    <div 
                      className="w-20 h-20 mx-auto mb-6 rounded-full flex items-center justify-center"
                      style={{
                        background: 'linear-gradient(135deg, rgba(124, 124, 255, 0.1) 0%, rgba(179, 107, 255, 0.1) 100%)',
                        border: '2px solid rgba(124, 124, 255, 0.2)'
                      }}
                    >
                      <div 
                        className="text-3xl"
                        style={{ color: '#7C7CFF' }}
                      >
                        ✨
                      </div>
                    </div>
                    <p className="text-xl font-medium mb-2" style={{ color: '#E8EBF3' }}>
                      Ready to find assessments
                    </p>
                    <p className="text-sm" style={{ color: '#A6ADC8' }}>
                      Enter a job description to get personalized recommendations
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer 
        className="mt-16 py-8 px-4"
        style={{ 
          backgroundColor: 'rgba(15, 19, 37, 0.8)', 
          backdropFilter: 'blur(10px)',
          borderTop: '1px solid rgba(255, 255, 255, 0.06)' 
        }}
      >
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="text-lg font-semibold mb-4" style={{ color: '#E8EBF3' }}>
                SHL Assessment Recommender
              </h3>
              <p className="text-sm" style={{ color: '#A6ADC8' }}>
                AI-powered assessment recommendations to help you find the perfect evaluation tools for your hiring needs.
              </p>
            </div>
            
            <div>
              <h4 className="text-sm font-semibold mb-3" style={{ color: '#E8EBF3' }}>
                Features
              </h4>
              <ul className="space-y-2 text-sm" style={{ color: '#A6ADC8' }}>
                <li>• Intelligent matching algorithm</li>
                <li>• Real-time recommendations</li>
                <li>• Advanced filtering options</li>
                <li>• Comprehensive skill analysis</li>
              </ul>
            </div>
            
            <div>
              <h4 className="text-sm font-semibold mb-3" style={{ color: '#E8EBF3' }}>
                System Status
              </h4>
              <div className="space-y-2 text-sm">
                <div className="flex items-center space-x-2">
                  <div 
                    className="w-2 h-2 rounded-full" 
                    style={{ backgroundColor: apiStatus === 'online' ? '#2EE59D' : '#DC2626' }}
                  ></div>
                  <span style={{ color: '#A6ADC8' }}>
                    API: {apiStatus === 'online' ? 'Online' : 'Offline'}
                  </span>
                </div>
                {healthData && (
                  <div style={{ color: '#7C84A3' }} className="text-xs">
                    {healthData.assessment_count} assessments available
                  </div>
                )}
              </div>
            </div>
          </div>
          
          <div 
            className="mt-8 pt-6 text-center text-sm"
            style={{ 
              borderTop: '1px solid rgba(255, 255, 255, 0.06)',
              color: '#7C84A3'
            }}
          >
            © 2024 SHL Assessment Recommender. Powered by AI technology.
          </div>
        </div>
      </footer>

      {/* Floating Action Button - Scroll to Top */}
      {recommendations.length > 0 && (
        <button
          onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          className="fixed bottom-8 right-8 w-14 h-14 rounded-full shadow-lg transition-all duration-300 z-50"
          style={{
            background: 'linear-gradient(135deg, #7C7CFF 0%, #B36BFF 100%)',
            color: '#FFFFFF',
            boxShadow: '0 8px 25px rgba(124, 124, 255, 0.4)'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'translateY(-2px) scale(1.05)';
            e.currentTarget.style.boxShadow = '0 12px 35px rgba(124, 124, 255, 0.6)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'translateY(0) scale(1)';
            e.currentTarget.style.boxShadow = '0 8px 25px rgba(124, 124, 255, 0.4)';
          }}
        >
          ↑
        </button>
      )}

      {/* Background Particles Effect */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 rounded-full animate-pulse"
            style={{
              backgroundColor: 'rgba(124, 124, 255, 0.1)',
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 3}s`,
              animationDuration: `${2 + Math.random() * 3}s`
            }}
          />
        ))}
      </div>
    </div>
  );
};

export default App;
