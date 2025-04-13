export interface Movie {
  id: string;
  title: string;
  posterUrl: string;
  year: number;
  director?: string;
  genre?: string[];
  rating?: number;
}

export type SwipeDirection = 'left' | 'right' | 'none';

export interface SwipeCardProps {
  movie: Movie;
  onSwipe: (direction: SwipeDirection, movie: Movie) => void;
  isTopCard?: boolean;
}

export interface MovieStackProps {
  movies: Movie[];
  onSwipeLeft: (movie: Movie) => void;
  onSwipeRight: (movie: Movie) => void;
} 