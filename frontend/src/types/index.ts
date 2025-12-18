export interface Assessment {
  assessment_name: string;
  url: string;
  relevance_score?: number;
  test_type?: string;
}

export interface RecommendationResponse {
  query: string;
  recommendations: Assessment[];
  total_results: number;
}

export interface ApiError {
  error_code: string;
  message: string;
  details?: Record<string, any>;
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  uptime: string;
  version: string;
  assessments_loaded: number;
}