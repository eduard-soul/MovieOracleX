import { Dimensions, Platform } from 'react-native';

// Obtenir les dimensions de l'écran
export const SCREEN_WIDTH = Dimensions.get('window').width;
export const SCREEN_HEIGHT = Dimensions.get('window').height;

// Ajustement du seuil de swipe pour être plus naturel (environ 35% de l'écran)
export const SWIPE_THRESHOLD = SCREEN_WIDTH * 0.35;

// Vitesse d'animation optimisée pour être fluide mais visible
export const SWIPE_OUT_DURATION = Platform.OS === 'ios' ? 200 : 230; // iOS est généralement plus fluide

// Constantes pour le comportement du swipe
export const ROTATION_MAGNITUDE = 12; // Moins extrême pour un effet plus naturel
export const ELASTIC_THRESHOLD = SCREEN_WIDTH * 0.15; // Où commence l'effet élastique
export const ELASTIC_FACTOR = 0.5; // Force de l'élasticité (plus bas = plus élastique)
export const VELOCITY_THRESHOLD = 0.4; // Seuil de vitesse pour considérer comme swipe

// Calculer la rotation plus douce et plus naturelle
export const getRotationDegrees = (x: number) => {
  // Fonction d'atténuation pour une rotation plus naturelle
  const sign = x < 0 ? -1 : 1;
  const absoluteX = Math.abs(x);
  
  // Effet d'élasticité: rotation ralentit progressivement plus on s'éloigne
  if (absoluteX > ELASTIC_THRESHOLD) {
    const elasticX = ELASTIC_THRESHOLD + (absoluteX - ELASTIC_THRESHOLD) * ELASTIC_FACTOR;
    return sign * (elasticX / SCREEN_WIDTH) * ROTATION_MAGNITUDE;
  }
  
  return sign * (absoluteX / SCREEN_WIDTH) * ROTATION_MAGNITUDE;
};

// Calculer l'opacité des icônes avec une progression plus douce
export const getLikeOpacity = (x: number) => {
  // Effet plus subtil et progressif, apparaît plus tôt
  return x > 0 ? Math.min(Math.pow(x / (SCREEN_WIDTH * 0.3), 1.5), 1) : 0;
};

export const getDislikeOpacity = (x: number) => {
  // Effet plus subtil et progressif, apparaît plus tôt
  return x < 0 ? Math.min(Math.pow(Math.abs(x) / (SCREEN_WIDTH * 0.3), 1.5), 1) : 0;
};

// Calcul de l'échelle pour l'effet 3D des cartes en pile
export const getCardScale = (index: number) => {
  // Effet de pile 3D plus subtil
  return Math.max(1 - (index * 0.04), 0.92);
};

// Fonction d'atténuation pour les translations des cartes
export const getCardOffset = (index: number) => {
  // Décalage plus naturel pour l'effet de pile
  return index * -8;
};

// Fonction pour déterminer si un swipe devrait être validé en fonction de la vitesse
export const shouldCompleteSwipe = (dx: number, vx: number) => {
  // Accepter le swipe si:
  // 1. Le déplacement dépasse le seuil OU
  // 2. La vitesse est suffisamment élevée dans la bonne direction
  return (
    Math.abs(dx) > SWIPE_THRESHOLD || 
    (Math.abs(vx) > VELOCITY_THRESHOLD && Math.sign(vx) === Math.sign(dx))
  );
}; 