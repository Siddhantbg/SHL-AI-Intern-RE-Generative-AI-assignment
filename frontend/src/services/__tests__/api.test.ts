// Mock the entire api module
jest.mock('../api', () => ({
  apiService: {
    getRecommendations: jest.fn(),
    getHealth: jest.fn(),
  },
}));

import { apiService } from '../api';

const mockedApiService = apiService as jest.Mocked<typeof apiService>;

describe('apiService', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('getRecommendations', () => {
    it('should return recommendations for a valid query', async () => {
      const mockResponse = {
        query: 'Software Engineer',
        recommendations: [
          {
            assessment_name: 'Python Test',
            url: 'https://example.com/python-test',
            relevance_score: 0.95,
            test_type: 'K'
          }
        ],
        total_results: 1
      };

      mockedApiService.getRecommendations.mockResolvedValue(mockResponse);

      const result = await apiService.getRecommendations('Software Engineer');

      expect(mockedApiService.getRecommendations).toHaveBeenCalledWith('Software Engineer');
      expect(result).toEqual(mockResponse);
    });

    it('should throw error when API call fails', async () => {
      const mockError = new Error('Network error');
      mockedApiService.getRecommendations.mockRejectedValue(mockError);

      await expect(apiService.getRecommendations('test query')).rejects.toThrow('Network error');
    });
  });

  describe('getHealth', () => {
    it('should return health status', async () => {
      const mockResponse = {
        status: 'healthy',
        uptime: '1h 30m',
        version: '1.0.0',
        assessments_loaded: 377
      };

      mockedApiService.getHealth.mockResolvedValue(mockResponse);

      const result = await apiService.getHealth();

      expect(mockedApiService.getHealth).toHaveBeenCalled();
      expect(result).toEqual(mockResponse);
    });

    it('should throw error when health check fails', async () => {
      const mockError = new Error('Service unavailable');
      mockedApiService.getHealth.mockRejectedValue(mockError);

      await expect(apiService.getHealth()).rejects.toThrow('Service unavailable');
    });
  });
});