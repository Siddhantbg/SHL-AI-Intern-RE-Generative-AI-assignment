import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';
import { apiService } from './services/api';

// Mock the API service
jest.mock('./services/api');
const mockedApiService = apiService as jest.Mocked<typeof apiService>;

describe('App', () => {
  beforeEach(() => {
    // Mock successful health check by default
    mockedApiService.getHealth.mockResolvedValue({
      status: 'healthy',
      uptime: '1h 30m',
      version: '1.0.0',
      assessments_loaded: 377
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('renders main components', async () => {
    render(<App />);
    
    expect(screen.getByText('SHL Assessment Recommender')).toBeInTheDocument();
    expect(screen.getByText('Find SHL Assessments')).toBeInTheDocument();
    expect(screen.getByText('Welcome to SHL Assessment Recommender')).toBeInTheDocument();
    
    // Wait for health check to complete
    await waitFor(() => {
      expect(screen.getByText('API Online')).toBeInTheDocument();
    });
  });

  it('shows API offline status when health check fails', async () => {
    mockedApiService.getHealth.mockRejectedValue(new Error('Connection failed'));
    
    render(<App />);
    
    await waitFor(() => {
      expect(screen.getByText('API Offline')).toBeInTheDocument();
    });
    
    expect(screen.getByText('API Connection Failed')).toBeInTheDocument();
  });

  it('handles successful recommendation request', async () => {
    
    mockedApiService.getRecommendations.mockResolvedValue({
      query: 'Software Engineer',
      recommendations: [
        {
          assessment_name: 'Python Programming Test',
          url: 'https://example.com/python-test',
          relevance_score: 0.95,
          test_type: 'K'
        }
      ],
      total_results: 1
    });

    render(<App />);
    
    // Wait for health check
    await waitFor(() => {
      expect(screen.getByText('API Online')).toBeInTheDocument();
    });
    
    // Enter query and submit
    const textarea = screen.getByLabelText('Job Description or Query');
    const submitButton = screen.getByRole('button', { name: 'Get Recommendations' });
    
    await userEvent.type(textarea, 'Software Engineer');
    await userEvent.click(submitButton);
    
    // Wait for results
    await waitFor(() => {
      expect(screen.getByText('Recommended Assessments')).toBeInTheDocument();
    });
    
    expect(screen.getByText('Python Programming Test')).toBeInTheDocument();
    expect(mockedApiService.getRecommendations).toHaveBeenCalledWith('Software Engineer');
  });

  it('handles API error during recommendation request', async () => {
    
    mockedApiService.getRecommendations.mockRejectedValue(new Error('API Error'));

    render(<App />);
    
    // Wait for health check
    await waitFor(() => {
      expect(screen.getByText('API Online')).toBeInTheDocument();
    });
    
    // Enter query and submit
    const textarea = screen.getByLabelText('Job Description or Query');
    const submitButton = screen.getByRole('button', { name: 'Get Recommendations' });
    
    await userEvent.type(textarea, 'Test query');
    await userEvent.click(submitButton);
    
    // Wait for error message
    await waitFor(() => {
      expect(screen.getByText('Error')).toBeInTheDocument();
    });
    
    expect(screen.getByText('API Error')).toBeInTheDocument();
  });

  it('refreshes API status when refresh button is clicked', async () => {
    
    render(<App />);
    
    // Wait for initial health check
    await waitFor(() => {
      expect(screen.getByText('API Online')).toBeInTheDocument();
    });
    
    // Click refresh button
    const refreshButton = screen.getByRole('button', { name: 'Refresh' });
    await userEvent.click(refreshButton);
    
    // Verify health check was called again
    expect(mockedApiService.getHealth).toHaveBeenCalledTimes(2);
  });
});
