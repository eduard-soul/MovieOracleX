import React, { useRef, useMemo, useEffect } from 'react';
import {
  StyleSheet,
  View,
  Text,
  Image,
  Animated,
  PanResponder,
  Platform,
} from 'react-native';
import { SwipeCardProps, SwipeDirection } from '../types';
import {
  SCREEN_WIDTH,
  SCREEN_HEIGHT,
  SWIPE_THRESHOLD,
  SWIPE_OUT_DURATION,
  getRotationDegrees,
  getLikeOpacity,
  getDislikeOpacity,
  shouldCompleteSwipe,
} from '../utils/animations';

// Import conditionnel pour le support de BlurView
let BlurView;
try {
  BlurView = require('@react-native-community/blur').BlurView;
} catch (e) {
  // Fallback si BlurView n'est pas disponible
  BlurView = View;
}

const SwipeCard: React.FC<SwipeCardProps> = ({ movie, onSwipe, isTopCard = true }) => {
  // Utiliser des refs pour les valeurs d'animation
  const position = useRef(new Animated.ValueXY()).current;
  const scale = useRef(new Animated.Value(1)).current;
  const opacity = useRef(new Animated.Value(1)).current;
  const shadowOpacity = useRef(new Animated.Value(isTopCard ? 0.3 : 0.1)).current;
  
  // Référence pour suivre si la carte a été swipée
  const swipedRef = useRef(false);
  const initialTouchRef = useRef({ x: 0, y: 0 });

  // Effet pour animer l'entrée des nouvelles cartes
  useEffect(() => {
    if (isTopCard) {
      // Animation d'entrée subtile pour la nouvelle carte du dessus
      Animated.parallel([
        Animated.timing(scale, {
          toValue: 1,
          duration: 300,
          useNativeDriver: true,
        }),
        Animated.timing(opacity, {
          toValue: 1,
          duration: 300,
          useNativeDriver: true,
        }),
        Animated.timing(shadowOpacity, {
          toValue: 0.3,
          duration: 300,
          useNativeDriver: true,
        })
      ]).start();
    }
  }, [isTopCard, scale, opacity, shadowOpacity]);

  // Configurer le gestionnaire de mouvements (PanResponder) avec des meilleures performances
  const panResponder = useMemo(() => 
    PanResponder.create({
      // Ne réagir que si c'est la carte du dessus et qu'elle n'a pas été swipée
      onStartShouldSetPanResponder: () => isTopCard && !swipedRef.current,
      
      // Capturer le point de départ du toucher
      onPanResponderGrant: (_, gesture) => {
        initialTouchRef.current = { x: gesture.x0, y: gesture.y0 };
        
        // Légère animation de feedback au toucher
        Animated.spring(shadowOpacity, {
          toValue: 0.5,
          useNativeDriver: true,
          friction: 8,
        }).start();
      },
      
      // Suivre le mouvement du doigt avec une petite atténuation pour plus de fluidité
      onPanResponderMove: (_, gesture) => {
        // Réduire légèrement le mouvement vertical pour un effet plus naturel
        const horizontalMovement = gesture.dx;
        const verticalMovement = gesture.dy * 0.5; // Atténuer le mouvement vertical
        
        position.setValue({ 
          x: horizontalMovement, 
          y: verticalMovement
        });
      },
      
      // Quand on relâche, déterminer si c'est un swipe ou un retour au centre
      onPanResponderRelease: (_, gesture) => {
        // Restaurer l'ombre normale
        Animated.spring(shadowOpacity, {
          toValue: 0.3,
          useNativeDriver: true,
          friction: 8,
        }).start();
        
        // Utiliser à la fois la distance et la vélocité pour décider
        if (shouldCompleteSwipe(gesture.dx, gesture.vx)) {
          const direction = gesture.dx > 0 ? 'right' : 'left';
          forceSwipe(direction, gesture.vx);
        } else {
          // Retourner au centre
          resetPosition();
        }
      },
      
      // Empêcher d'autres vues de devenir le responder pendant le swipe
      onPanResponderTerminationRequest: () => false,
    }), [isTopCard]); // Recréer seulement si isTopCard change

  // Animation pour forcer un swipe complet dans une direction avec physique naturelle
  const forceSwipe = (direction: SwipeDirection, velocity = 1) => {
    // Éviter les swipes multiples sur la même carte
    if (swipedRef.current) return;
    swipedRef.current = true;
    
    // Calculer l'emplacement final
    const x = direction === 'right' 
      ? SCREEN_WIDTH * 1.5 
      : -SCREEN_WIDTH * 1.5;
    
    // Vitesse ajustée en fonction de la vélocité de l'utilisateur
    const adjustedDuration = SWIPE_OUT_DURATION / Math.max(Math.abs(velocity), 1);
    
    // Animer la sortie de la carte avec un timing naturel
    Animated.parallel([
      Animated.timing(position, {
        toValue: { x, y: 0 },
        duration: Math.min(adjustedDuration, SWIPE_OUT_DURATION),
        useNativeDriver: true,
      }),
      Animated.timing(opacity, {
        toValue: 0.5, // Légère disparition pendant la sortie
        duration: SWIPE_OUT_DURATION,
        useNativeDriver: true,
      })
    ]).start(() => {
      // Callback quand l'animation est terminée
      onSwipe(direction, movie);
    });
  };

  // Animation pour retourner la carte à sa position centrale avec effet rebond
  const resetPosition = () => {
    Animated.spring(position, {
      toValue: { x: 0, y: 0 },
      friction: 7, // Plus d'amortissement pour un effet plus naturel
      tension: 80, // Tension modérée pour éviter un retour trop brusque
      useNativeDriver: true,
    }).start();
  };
  
  // Interpolations pour les effets visuels
  const rotateCard = position.x.interpolate({
    inputRange: [-SCREEN_WIDTH * 1.5, 0, SCREEN_WIDTH * 1.5],
    outputRange: ['-12deg', '0deg', '12deg'],
    extrapolate: 'clamp',
  });

  const cardScale = scale.interpolate({
    inputRange: [0, 1],
    outputRange: [0.96, 1],
    extrapolate: 'clamp',
  });

  // Combiner les styles animés
  const cardAnimatedStyle = {
    transform: [
      { translateX: position.x },
      { translateY: position.y },
      { rotate: rotateCard },
      { scale: cardScale },
    ],
    opacity,
  };

  // Style dynamique pour les ombres
  const cardShadowStyle = {
    shadowOpacity,
    elevation: isTopCard ? 8 : 3,
  };

  return (
    <Animated.View 
      style={[styles.container, cardAnimatedStyle, cardShadowStyle]}
      {...(isTopCard ? panResponder.panHandlers : {})}
    >
      <Image 
        source={{ uri: movie.posterUrl }} 
        style={styles.poster}
        resizeMode="cover"
      />
      
      {/* Simple bande d'information en bas */}
      <View style={styles.infoBar}>
        <Text style={styles.title}>
          {movie.title} <Text style={styles.year}>({movie.year})</Text>
        </Text>
        
        <View style={styles.metadataContainer}>
          {movie.genre && (
            <Text style={styles.genre}>{movie.genre.join(' • ')}</Text>
          )}
        </View>
      </View>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
    height: '100%',
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#fff',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 8,
  },
  poster: {
    width: '100%',
    height: '100%',
  },
  infoBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: 'rgba(255,255,255,0.85)',
    borderBottomLeftRadius: 12,
    borderBottomRightRadius: 12,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#222',
  },
  year: {
    fontSize: 14,
    fontWeight: 'normal',
    color: '#555',
  },
  metadataContainer: {
    flexDirection: 'row',
    marginTop: 2,
  },
  genre: {
    fontSize: 12,
    color: '#555',
  }
});

export default SwipeCard; 