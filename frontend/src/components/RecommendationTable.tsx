import React from 'react';
import { Assessment } from '../types';

interface RecommendationTableProps {
  recommendations: Assessment[];
  query: string;
}

const RecommendationTable: React.FC<RecommendationTableProps> = ({ 
  recommendations, 
  query 
}) => {
  if (recommendations.length === 0) {
    return (
      <div className="bg-white shadow-sm rounded-lg p-6">
        <div className="text-center py-8">
          <svg
            className="mx-auto h-12 w-12 text-gray-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <h3 className="mt-2 text-sm font-medium text-gray-900">No recommendations found</h3>
          <p className="mt-1 text-sm text-gray-500">
            Try adjusting your query or using different keywords.
          </p>
        </div>
      </div>
    );
  }

  const getTestTypeBadge = (testType?: string) => {
    if (!testType) return null;
    
    const isKnowledgeSkills = testType === 'K';
    const badgeClass = isKnowledgeSkills 
      ? 'bg-blue-100 text-blue-800' 
      : 'bg-green-100 text-green-800';
    const label = isKnowledgeSkills ? 'Knowledge & Skills' : 'Personality & Behavior';
    
    return (
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${badgeClass}`}>
        {label}
      </span>
    );
  };

  const getRelevanceColor = (score?: number) => {
    if (!score) return 'text-gray-500';
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="bg-white shadow-sm rounded-lg overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-200">
        <h3 className="text-lg font-medium text-gray-900">
          Recommended Assessments
        </h3>
        <p className="mt-1 text-sm text-gray-500">
          Found {recommendations.length} assessment{recommendations.length !== 1 ? 's' : ''} for: "{query}"
        </p>
      </div>
      
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Assessment Name
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Type
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Relevance
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Action
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {recommendations.map((assessment, index) => (
              <tr key={index} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium text-gray-900">
                    {assessment.assessment_name}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {getTestTypeBadge(assessment.test_type)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {assessment.relevance_score && (
                    <div className="flex items-center">
                      <div className="flex-1 bg-gray-200 rounded-full h-2 mr-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full"
                          style={{ width: `${assessment.relevance_score * 100}%` }}
                        ></div>
                      </div>
                      <span className={`text-sm font-medium ${getRelevanceColor(assessment.relevance_score)}`}>
                        {Math.round(assessment.relevance_score * 100)}%
                      </span>
                    </div>
                  )}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                  <a
                    href={assessment.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-900 inline-flex items-center"
                  >
                    View Assessment
                    <svg
                      className="ml-1 h-4 w-4"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                      />
                    </svg>
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      <div className="px-6 py-3 bg-gray-50 border-t border-gray-200">
        <p className="text-xs text-gray-500">
          Click "View Assessment" to learn more about each assessment on the SHL website.
        </p>
      </div>
    </div>
  );
};

export default RecommendationTable;