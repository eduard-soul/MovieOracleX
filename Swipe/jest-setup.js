// Mock complet pour React Native sans référence aux modules natifs
jest.mock('react-native', () => ({
  // Composants de base
  View: 'View',
  Text: 'Text',
  Image: 'Image',
  ScrollView: 'ScrollView',
  TouchableOpacity: 'TouchableOpacity',
  StyleSheet: {
    create: (styles) => styles,
    flatten: jest.fn(),
    absoluteFill: {},
    hairlineWidth: 1,
  },
  Dimensions: {
    get: jest.fn().mockReturnValue({ width: 375, height: 812 }),
  },
  Platform: {
    OS: 'ios',
    select: jest.fn((obj) => obj.ios),
  },
  // Animated API
  Animated: {
    View: 'Animated.View',
    Text: 'Animated.Text',
    Image: 'Animated.Image',
    ValueXY: jest.fn(() => ({
      x: { interpolate: jest.fn(() => ({ 
        interpolate: jest.fn().mockReturnValue('interpolated-value') 
      })) },
      y: { interpolate: jest.fn() },
      setValue: jest.fn(),
    })),
    Value: jest.fn(() => ({
      setValue: jest.fn(),
      interpolate: jest.fn().mockReturnValue('interpolated-value'),
    })),
    timing: jest.fn(() => ({
      start: jest.fn((callback) => callback && callback()),
    })),
    spring: jest.fn(() => ({
      start: jest.fn((callback) => callback && callback()),
    })),
    parallel: jest.fn(() => ({
      start: jest.fn((callback) => callback && callback()),
    })),
  },
  // PanResponder
  PanResponder: {
    create: jest.fn(() => ({
      panHandlers: {},
    })),
  },
  // Autres APIs courantes
  Alert: {
    alert: jest.fn(),
  },
  PixelRatio: {
    get: jest.fn(() => 2),
    getPixelSizeForLayoutSize: jest.fn((size) => size * 2),
  },
}));

// Mock pour expo-blur
jest.mock('expo-blur', () => ({
  BlurView: 'BlurView',
}));

// Désactiver les erreurs et avertissements console pour les tests
global.console = {
  ...console,
  error: jest.fn(),
  warn: jest.fn(),
  log: jest.fn(),
};

// Pour éviter les warnings sur setNativeProps
const mockComponent = (name) => {
  return function (props) {
    return React.createElement(name, {
      ...props,
      setNativeProps: () => {},
    });
  };
}; 