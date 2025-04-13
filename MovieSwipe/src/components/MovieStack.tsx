import React, { useState, useCallback, useRef, useEffect } from 'react';
import { View, StyleSheet, Text, Animated, Dimensions } from 'react-native';
import { Movie, MovieStackProps, SwipeDirection } from '../types';
import SwipeCard from './SwipeCard';
import { SCREEN_WIDTH, SCREEN_HEIGHT, getCardScale, getCardOffset } from '../utils/animations';

// Constantes pour le positionnement absolu
const { width, height } = Dimensions.get('window');
const CARD_WIDTH = width * 0.85;
const CARD_HEIGHT = height * 0.65;

const MovieStack: React.FC<MovieStackProps> = ({ movies, onSwipeLeft, onSwipeRight }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  
  // Animation value for card transition
  const transitionProgress = useRef(new Animated.Value(0)).current;
  
  // Reference to track if transition animation is in progress
  const isTransitioning = useRef(false);
  
  // Reset animation value when a new card appears
  useEffect(() => {
    transitionProgress.setValue(0);
  }, [currentIndex, transitionProgress]);
  
  // Fonction pour gérer le swipe d'une carte - version optimisée sans délai
  const handleSwipe = useCallback((direction: SwipeDirection, movie: Movie) => {
    // Empêcher les swipes multiples 
    if (isTransitioning.current) return;
    isTransitioning.current = true;
    
    // Passer immédiatement à la carte suivante sans attendre l'animation
    setCurrentIndex(prevIndex => prevIndex + 1);
    
    // Indiquer la fin de la transition immédiatement
    setTimeout(() => {
      isTransitioning.current = false;
    }, 10);
    
    // Exécuter les callbacks appropriés sans délai
    if (direction === 'left') {
      onSwipeLeft(movie);
    } else if (direction === 'right') {
      onSwipeRight(movie);
    }
  }, [onSwipeLeft, onSwipeRight]);
  
  // Rendu des cartes visibles (carte actuelle + quelques suivantes pour la performance)
  const renderCards = () => {
    if (currentIndex >= movies.length) {
      // Plus de cartes à afficher - écran vide simple
      return (
        <View style={styles.emptyStateContainer}>
          <Text style={styles.emptyStateText}>Plus de films disponibles</Text>
        </View>
      );
    }
    
    // Afficher jusqu'à 3 cartes pour l'effet d'empilement 3D (pour les performances)
    return movies
      .slice(currentIndex, currentIndex + 3)
      .map((movie, index) => {
        // Calculer les transformations en fonction de l'index dans la pile
        const isTopCard = index === 0;
        const scale = getCardScale(index);
        const translateY = getCardOffset(index);
        
        // Opacité réduite pour les cartes inférieures
        const opacity = Math.max(1 - (index * 0.15), 0.6);
        
        // Animation plus rapide pour la carte suivante
        const nextCardScale = transitionProgress.interpolate({
          inputRange: [0, 0.5], // Réduire l'intervalle pour une transition plus rapide
          outputRange: [scale, 1],
          extrapolate: 'clamp',
        });
        
        const nextCardTranslateY = transitionProgress.interpolate({
          inputRange: [0, 0.5], // Réduire l'intervalle pour une transition plus rapide
          outputRange: [translateY, 0],
          extrapolate: 'clamp',
        });
        
        // Styles pour la pile de cartes
        const cardStyle = {
          transform: [
            { scale: isTopCard ? scale : nextCardScale },
            { translateY: isTopCard ? translateY : nextCardTranslateY }
          ],
          opacity: isTopCard ? 1 : opacity,
          zIndex: movies.length - index,
        };
        
        return (
          <Animated.View 
            key={movie.id} 
            style={[styles.cardContainer, cardStyle]}
          >
            <SwipeCard 
              movie={movie}
              onSwipe={handleSwipe}
              isTopCard={isTopCard}
            />
          </Animated.View>
        );
      });
  };
  
  return (
    <View style={styles.container}>
      {renderCards()}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cardContainer: {
    position: 'absolute',
    width: CARD_WIDTH,
    height: CARD_HEIGHT,
    // Centrage absolu pour le positionnement
    top: (height - CARD_HEIGHT) / 2,
    left: (width - CARD_WIDTH) / 2,
  },
  emptyStateContainer: {
    width: CARD_WIDTH,
    height: CARD_HEIGHT,
    borderRadius: 16,
    backgroundColor: 'rgba(240,240,240,0.8)',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  emptyStateText: {
    fontSize: 18,
    fontWeight: '500',
    color: '#888',
    textAlign: 'center',
  }
});

export default MovieStack; 