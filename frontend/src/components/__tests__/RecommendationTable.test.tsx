import React from 'react';
import { render, screen } from '@testing-library/react';
import RecommendationTable from '../RecommendationTable';
import { Assessment } from '../../types';

describe('RecommendationTable', () => {
  const mockRecommendations: Assessment[] = [
    {
      assessment_name: 'Python Programming Test',
      url: 'https://example.com/python-test',
      relevance_score: 0.95,
      test_type: 'K'
    },
    {
      assessment_name: 'Leadership Assessment',
      url: 'https://example.com/leadership-test',
      relevance_score: 0.78,
      test_type: 'P'
    }
  ];

  it('renders recommendations table correctly', () => {
    render(
      <RecommendationTable 
        recommendations={mockRecommendations} 
        query="Software Engineer" 
      />
    );
    
    expect(screen.getByText('Recommended Assessments')).toBeInTheDocument();
    expect(screen.getByText('Found 2 assessments for: "Software Engineer"')).toBeInTheDocument();
    expect(screen.getByText('Python Programming Test')).toBeInTheDocument();
    expect(screen.getByText('Leadership Assessment')).toBeInTheDocument();
  });

  it('displays test type badges correctly', () => {
    render(
      <RecommendationTable 
        recommendations={mockRecommendations} 
        query="Test query" 
      />
    );
    
    expect(screen.getByText('Knowledge & Skills')).toBeInTheDocument();
    expect(screen.getByText('Personality & Behavior')).toBeInTheDocument();
  });

  it('renders view assessment links', () => {
    render(
      <RecommendationTable 
        recommendations={mockRecommendations} 
        query="Test query" 
      />
    );
    
    const links = screen.getAllByText('View Assessment');
    expect(links).toHaveLength(2);
    expect(links[0].closest('a')).toHaveAttribute('href', 'https://example.com/python-test');
    expect(links[1].closest('a')).toHaveAttribute('href', 'https://example.com/leadership-test');
  });

  it('displays relevance scores as percentages', () => {
    render(
      <RecommendationTable 
        recommendations={mockRecommendations} 
        query="Test query" 
      />
    );
    
    expect(screen.getByText('95%')).toBeInTheDocument();
    expect(screen.getByText('78%')).toBeInTheDocument();
  });

  it('shows empty state when no recommendations', () => {
    render(
      <RecommendationTable 
        recommendations={[]} 
        query="No results query" 
      />
    );
    
    expect(screen.getByText('No recommendations found')).toBeInTheDocument();
    expect(screen.getByText('Try adjusting your query or using different keywords.')).toBeInTheDocument();
  });

  it('handles recommendations without scores or test types', () => {
    const minimalRecommendations: Assessment[] = [
      {
        assessment_name: 'Basic Test',
        url: 'https://example.com/basic-test'
      }
    ];
    
    render(
      <RecommendationTable 
        recommendations={minimalRecommendations} 
        query="Basic query" 
      />
    );
    
    expect(screen.getByText('Basic Test')).toBeInTheDocument();
    expect(screen.getByText('View Assessment')).toBeInTheDocument();
  });
});