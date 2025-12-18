import React, { useState } from 'react';

interface QueryInputProps {
  onSubmit: (query: string) => void;
  isLoading: boolean;
}

const QueryInput: React.FC<QueryInputProps> = ({ onSubmit, isLoading }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSubmit(query.trim());
    }
  };

  const exampleQueries = [
    "Software Engineer with Python and machine learning skills",
    "Sales Manager with leadership and communication abilities",
    "Data Analyst with SQL and statistical analysis experience",
    "Customer Service Representative with problem-solving skills"
  ];

  const handleExampleClick = (example: string) => {
    setQuery(example);
  };

  return (
    <div className="bg-white shadow-sm rounded-lg p-6 mb-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">
        Find SHL Assessments
      </h2>
      <p className="text-gray-600 mb-4">
        Enter a job description, role requirements, or skills to get personalized assessment recommendations.
      </p>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
            Job Description or Query
          </label>
          <textarea
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., Looking for a software developer with strong problem-solving skills, experience in Python, and ability to work in agile teams..."
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
            rows={4}
            disabled={isLoading}
          />
        </div>
        
        <div className="flex justify-between items-center">
          <button
            type="submit"
            disabled={!query.trim() || isLoading}
            className="bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? 'Getting Recommendations...' : 'Get Recommendations'}
          </button>
          
          <button
            type="button"
            onClick={() => setQuery('')}
            disabled={isLoading}
            className="text-gray-500 hover:text-gray-700 px-3 py-2 text-sm disabled:opacity-50"
          >
            Clear
          </button>
        </div>
      </form>

      <div className="mt-6">
        <h3 className="text-sm font-medium text-gray-700 mb-2">Try these examples:</h3>
        <div className="space-y-2">
          {exampleQueries.map((example, index) => (
            <button
              key={index}
              onClick={() => handleExampleClick(example)}
              disabled={isLoading}
              className="block w-full text-left text-sm text-blue-600 hover:text-blue-800 hover:bg-blue-50 px-3 py-2 rounded-md transition-colors disabled:opacity-50"
            >
              "{example}"
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default QueryInput;