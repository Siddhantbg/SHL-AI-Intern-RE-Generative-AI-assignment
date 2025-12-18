import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import QueryInput from '../QueryInput';

describe('QueryInput', () => {
  const mockOnSubmit = jest.fn();

  beforeEach(() => {
    mockOnSubmit.mockClear();
  });

  it('renders input form correctly', () => {
    render(<QueryInput onSubmit={mockOnSubmit} isLoading={false} />);
    
    expect(screen.getByText('Find SHL Assessments')).toBeInTheDocument();
    expect(screen.getByLabelText('Job Description or Query')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Get Recommendations' })).toBeInTheDocument();
  });

  it('calls onSubmit when form is submitted with valid query', async () => {
    render(<QueryInput onSubmit={mockOnSubmit} isLoading={false} />);
    
    const textarea = screen.getByLabelText('Job Description or Query');
    const submitButton = screen.getByRole('button', { name: 'Get Recommendations' });
    
    await userEvent.type(textarea, 'Software Engineer with Python skills');
    await userEvent.click(submitButton);
    
    expect(mockOnSubmit).toHaveBeenCalledWith('Software Engineer with Python skills');
  });

  it('does not submit empty query', async () => {
    render(<QueryInput onSubmit={mockOnSubmit} isLoading={false} />);
    
    const submitButton = screen.getByRole('button', { name: 'Get Recommendations' });
    await userEvent.click(submitButton);
    
    expect(mockOnSubmit).not.toHaveBeenCalled();
  });

  it('disables form when loading', () => {
    render(<QueryInput onSubmit={mockOnSubmit} isLoading={true} />);
    
    const textarea = screen.getByLabelText('Job Description or Query');
    const submitButton = screen.getByRole('button', { name: 'Getting Recommendations...' });
    
    expect(textarea).toBeDisabled();
    expect(submitButton).toBeDisabled();
  });

  it('clears input when clear button is clicked', async () => {
    render(<QueryInput onSubmit={mockOnSubmit} isLoading={false} />);
    
    const textarea = screen.getByLabelText('Job Description or Query');
    const clearButton = screen.getByRole('button', { name: 'Clear' });
    
    await userEvent.type(textarea, 'Test query');
    expect(textarea).toHaveValue('Test query');
    
    await userEvent.click(clearButton);
    expect(textarea).toHaveValue('');
  });

  it('fills input when example is clicked', async () => {
    render(<QueryInput onSubmit={mockOnSubmit} isLoading={false} />);
    
    const textarea = screen.getByLabelText('Job Description or Query');
    const exampleButton = screen.getByText('"Software Engineer with Python and machine learning skills"');
    
    await userEvent.click(exampleButton);
    expect(textarea).toHaveValue('Software Engineer with Python and machine learning skills');
  });
});