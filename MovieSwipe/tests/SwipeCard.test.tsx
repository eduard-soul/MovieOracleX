import React from 'react';
import { render } from '@testing-library/react-native';
import SwipeCard from '../src/components/SwipeCard';
import { Movie, SwipeDirection } from '../src/types';

// Film de test
const testMovie: Movie = {
  id: 'test-1',
  title: 'Test Movie',
  posterUrl: 'https://example.com/poster.jpg',
  year: 2023,
  director: 'Test Director',
  genre: ['Action', 'Drama'],
  rating: 8.5
};

// Mocker le comportement de onSwipe pour les tests
const mockOnSwipe = jest.fn();

describe('SwipeCard', () => {
  // Réinitialiser les mocks entre les tests
  beforeEach(() => {
    mockOnSwipe.mockClear();
  });

  // Test basique pour vérifier que le composant s'affiche
  it('can be rendered', () => {
    const { getByText } = render(
      <SwipeCard 
        movie={testMovie} 
        onSwipe={mockOnSwipe} 
        isTopCard={true} 
      />
    );
    
    // Vérifier que le titre du film est affiché
    expect(getByText('Test Movie')).toBeTruthy();
  });

  // Test pour vérifier que le genre est affiché
  it('shows the movie genre', () => {
    const { getByText } = render(
      <SwipeCard 
        movie={testMovie} 
        onSwipe={mockOnSwipe} 
        isTopCard={true} 
      />
    );
    
    expect(getByText('Action • Drama')).toBeTruthy();
  });

  // Test pour vérifier que l'année est affichée
  it('shows the movie year', () => {
    const { getByText } = render(
      <SwipeCard 
        movie={testMovie} 
        onSwipe={mockOnSwipe} 
        isTopCard={true} 
      />
    );
    
    expect(getByText('(2023)')).toBeTruthy();
  });
}); 