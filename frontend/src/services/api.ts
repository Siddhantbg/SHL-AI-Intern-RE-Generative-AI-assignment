import axios from 'axios';
import { RecommendationResponse, HealthResponse } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiService = {
  async getRecommendations(query: string): Promise<RecommendationResponse> {
    const response = await api.post('/recommend', { query });
    return response.data;
  },

  async getHealth(): Promise<HealthResponse> {
    const response = await api.get('/health');
    return response.data;
  },
};

export default apiService;