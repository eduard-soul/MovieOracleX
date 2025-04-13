# MovieSwipe

Une application mobile React Native/Expo inspirée de Tinder pour swiper des films. Permet aux utilisateurs de faire glisser des cartes de films vers la gauche (dislike) ou vers la droite (like).

## Fonctionnalités

- Interface utilisateur fluide et réactive avec animations
- Design moderne et professionnel
- Swipe à gauche pour "dislike", à droite pour "like"
- Feedback visuel pendant le swipe (indicateurs "LIKE" et "NOPE")
- Pile de cartes avec effet 3D
- Boutons d'action pour like/dislike/undo
- Compteurs de likes et dislikes

## Prérequis

- Node.js (v18 ou supérieur)
- npm ou yarn
- Expo CLI (`npm install -g expo-cli`)
- Un émulateur iOS/Android ou l'application Expo Go sur un appareil physique

## Installation

1. Cloner le dépôt ou télécharger les fichiers
2. Installer les dépendances :

```bash
cd MovieSwipe
npm install
```

## Exécution de l'application

Pour démarrer l'application en mode développement :

```bash
npm start
```

Ou utilisez une des commandes suivantes pour cibler une plateforme spécifique :

```bash
npm run android  # Pour Android
npm run ios      # Pour iOS
npm run web      # Pour le web
```

## Tests

Pour exécuter les tests unitaires :

```bash
npm test
```

Pour exécuter les tests en mode watch (mise à jour automatique) :

```bash
npm run test:watch
```

Pour générer un rapport de couverture de tests :

```bash
npm run test:coverage
```

## Structure du projet

```
MovieSwipe/
├── src/
│   ├── components/        # Composants React
│   │   ├── SwipeCard.tsx  # Composant de carte individuelle
│   │   └── MovieStack.tsx # Gestionnaire de la pile de cartes
│   ├── types/             # Types TypeScript
│   ├── utils/             # Fonctions utilitaires
│   │   ├── animations.ts  # Utilitaires d'animation
│   │   └── mockData.ts    # Données de films fictives
│   └── assets/            # Images et ressources
├── tests/                 # Tests unitaires et e2e
├── App.tsx               # Point d'entrée de l'application
└── ...
```

## Personnalisation

### Ajouter de nouveaux films

Modifiez le fichier `src/utils/mockData.ts` pour ajouter, supprimer ou modifier les films.

### Modifier l'apparence des cartes

Les styles des cartes peuvent être personnalisés dans le fichier `src/components/SwipeCard.tsx`.

### Changer les actions de swipe

Les actions lors du swipe sont définies dans `App.tsx` dans les fonctions `handleSwipeLeft` et `handleSwipeRight`.

## Optimisations et améliorations futures

- Intégration d'une API de films réelle (comme TMDB)
- Persistance des likes/dislikes avec AsyncStorage
- Animations plus avancées et transitions
- Thème sombre/clair
- Support multi-langue
- Filtres par genre, année, etc.

## Licence

MIT 