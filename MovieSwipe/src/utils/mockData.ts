import { Movie } from '../types';

export const MOCK_MOVIES: Movie[] = [
  {
    id: '1',
    title: 'Inception',
    posterUrl: 'https://image.tmdb.org/t/p/w500/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg',
    year: 2010,
    director: 'Christopher Nolan',
    genre: ['Action', 'Sci-Fi', 'Thriller'],
    rating: 8.8
  },
  {
    id: '2',
    title: 'The Shawshank Redemption',
    posterUrl: 'https://image.tmdb.org/t/p/w500/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg',
    year: 1994,
    director: 'Frank Darabont',
    genre: ['Drama'],
    rating: 9.3
  },
  {
    id: '3',
    title: 'Pulp Fiction',
    posterUrl: 'https://image.tmdb.org/t/p/w500/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg',
    year: 1994,
    director: 'Quentin Tarantino',
    genre: ['Crime', 'Drama'],
    rating: 8.9
  },
  {
    id: '4',
    title: 'The Dark Knight',
    posterUrl: 'https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg',
    year: 2008,
    director: 'Christopher Nolan',
    genre: ['Action', 'Crime', 'Drama'],
    rating: 9.0
  },
  {
    id: '5',
    title: 'La La Land',
    posterUrl: 'https://image.tmdb.org/t/p/w500/uDO8zWDhfWwoFdKS4fzkUJt0Rf0.jpg',
    year: 2016,
    director: 'Damien Chazelle',
    genre: ['Comedy', 'Drama', 'Music'],
    rating: 8.0
  }
]; 