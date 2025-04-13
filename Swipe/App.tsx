import React, { useCallback, useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, View, ImageBackground } from 'react-native';
import { BlurView } from 'expo-blur';
import MovieStack from './src/components/MovieStack';
import { MOCK_MOVIES } from './src/utils/mockData';
import { Movie } from './src/types';

export default function App() {
  const [currentMovie, setCurrentMovie] = useState<Movie | null>(MOCK_MOVIES[0] || null);

  // Gérer le swipe à gauche (dislike) - garder simple, juste pour le callback
  const handleSwipeLeft = useCallback((movie: Movie) => {
    console.log(`Disliked: ${movie.title}`);
    // Mettre à jour l'arrière-plan avec le prochain film
    const nextIndex = MOCK_MOVIES.findIndex(m => m.id === movie.id) + 1;
    if (nextIndex < MOCK_MOVIES.length) {
      setCurrentMovie(MOCK_MOVIES[nextIndex]);
    }
  }, []);

  // Gérer le swipe à droite (like) - garder simple
  const handleSwipeRight = useCallback((movie: Movie) => {
    console.log(`Liked: ${movie.title}`);
    // Mettre à jour l'arrière-plan avec le prochain film
    const nextIndex = MOCK_MOVIES.findIndex(m => m.id === movie.id) + 1;
    if (nextIndex < MOCK_MOVIES.length) {
      setCurrentMovie(MOCK_MOVIES[nextIndex]);
    }
  }, []);

  return (
    <View style={styles.container}>
      {/* Arrière-plan flouté avec l'image du film actuel */}
      {currentMovie && (
        <ImageBackground
          source={{ uri: currentMovie.posterUrl }}
          style={styles.backgroundImage}
          resizeMode="cover"
        >
          <BlurView intensity={90} style={styles.blurContainer} tint="dark" />
        </ImageBackground>
      )}
      
      <StatusBar style="light" />
      <MovieStack 
        movies={MOCK_MOVIES}
        onSwipeLeft={handleSwipeLeft}
        onSwipeRight={handleSwipeRight}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  backgroundImage: {
    ...StyleSheet.absoluteFillObject,
  },
  blurContainer: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.3)',
  }
});
