import React, { useRef, useState } from "react";
import {
  View,
  Animated,
  PanResponder,
  StyleSheet,
  Platform,
  Dimensions,
  Text,
} from "react-native";

const ScalableMovableCard = () => {
  const screenHeight = Dimensions.get("window").height;
  const screenWidth = Dimensions.get("window").width;
  let cardHeight = screenHeight * 0.7;
  let cardWidth = (2 / 3) * cardHeight;

  if (cardWidth > screenWidth * 0.8) {
    cardWidth = 0.8 * screenWidth;
    cardHeight = (3 / 2) * cardWidth;
  }

  // Thresholds
  const swipeThreshold = cardWidth * 0.2;
  const dismissThreshold = cardWidth * 0.4;

  // Card state
  const [topCardIndex, setTopCardIndex] = useState(0);
  const [swipeDirection, setSwipeDirection] = useState(null);
  const isTransitioning = useRef(false);

  // Card properties
  const cardColors = ["#FFFFFF", "#F0F0F0"];

  // Animated values - create just once
  const positions = [
    useRef(new Animated.ValueXY({ x: 0, y: 0 })).current,
    useRef(new Animated.ValueXY({ x: 0, y: 0 })).current
  ];

  const scales = [
    useRef(new Animated.Value(1)).current,
    useRef(new Animated.Value(1)).current
  ];

  // Scale gesture handling
  const initialDistance = useRef(0);
  const initialScale = useRef(1);
  const isScaling = useRef(false);

  // Calculate distance between two touches for pinch scaling
  const getDistance = (touches) => {
    if (touches.length < 2) return 0;
    const [touch1, touch2] = touches;
    const dx = touch1.pageX - touch2.pageX;
    const dy = touch1.pageY - touch2.pageY;
    return Math.sqrt(dx * dx + dy * dy);
  };

  // Function to dismiss a card - sends it flying off screen
  const dismissCard = (direction) => {
    if (isTransitioning.current) return;

    const currentIndex = topCardIndex;
    isTransitioning.current = true;

    // Determine target position
    let targetX = 0;
    let targetY = 0;

    switch (direction) {
      case "left":
        targetX = -screenWidth * 1.5;
        break;
      case "right":
        targetX = screenWidth * 1.5;
        break;
      case "up":
        targetY = -screenHeight * 1.5;
        break;
      case "down":
        targetY = screenHeight * 1.5;
        break;
    }

    // Accelerate card off screen
    Animated.spring(positions[currentIndex], {
      toValue: { x: targetX, y: targetY },
      friction: 5, // Less friction = more bouncy/faster
      tension: 10,
      useNativeDriver: true,
    }).start(() => {
      // Switch cards
      const newTopIndex = currentIndex === 0 ? 1 : 0;

      // Reset the card that just flew off
      positions[currentIndex].setValue({ x: 0, y: 0 });
      scales[currentIndex].setValue(1);

      // Update state
      setTopCardIndex(newTopIndex);
      setSwipeDirection(null);
      isTransitioning.current = false;
    });
  };

  // Create a single panResponder for the top card
  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,

      onPanResponderGrant: (evt) => {
        if (isTransitioning.current) return false;

        const touches = evt.nativeEvent.touches;

        // Handle pinch to zoom
        if (touches.length === 2) {
          isScaling.current = true;
          initialDistance.current = getDistance(touches);
          initialScale.current = scales[topCardIndex]._value;
        } else {
          isScaling.current = false;
        }
        return true;
      },

      onPanResponderMove: (evt, gestureState) => {
        if (isTransitioning.current) return;

        const touches = evt.nativeEvent.touches;

        // Handle pinch to zoom
        if (touches.length === 2 && isScaling.current) {
          const currentDistance = getDistance(touches);
          if (initialDistance.current > 0) {
            const newScale = initialScale.current *
              (currentDistance / initialDistance.current);
            scales[topCardIndex].setValue(Math.max(0.5, Math.min(3, newScale)));
          }
        }
        // Handle dragging
        else if (touches.length === 1 && !isScaling.current) {
          // Update position based on gesture
          positions[topCardIndex].setValue({
            x: gestureState.dx,
            y: gestureState.dy
          });

          // Determine swipe direction for indicator
          const { dx, dy } = gestureState;

          if (Math.abs(dx) > Math.abs(dy)) {
            // Horizontal movement dominant
            if (dx > swipeThreshold) {
              setSwipeDirection("right");
            } else if (dx < -swipeThreshold) {
              setSwipeDirection("left");
            } else {
              setSwipeDirection(null);
            }
          } else {
            // Vertical movement dominant
            if (dy > swipeThreshold) {
              setSwipeDirection("down");
            } else if (dy < -swipeThreshold) {
              setSwipeDirection("up");
            } else {
              setSwipeDirection(null);
            }
          }
        }
      },

      onPanResponderRelease: (_, gestureState) => {
        if (isTransitioning.current) return;

        const { dx, dy } = gestureState;

        // Check if swipe was forceful enough to dismiss
        if (Math.abs(dx) > dismissThreshold) {
          dismissCard(dx > 0 ? "right" : "left");
        } else if (Math.abs(dy) > dismissThreshold) {
          dismissCard(dy > 0 ? "down" : "up");
        } else {
          // Spring back to center
          Animated.spring(positions[topCardIndex], {
            toValue: { x: 0, y: 0 },
            friction: 10,
            tension: 40,
            useNativeDriver: true,
          }).start(() => {
            setSwipeDirection(null);
          });
        }

        isScaling.current = false;
      },

      onPanResponderTerminate: () => {
        // Reset on termination
        if (!isTransitioning.current) {
          Animated.spring(positions[topCardIndex], {
            toValue: { x: 0, y: 0 },
            friction: 10,
            tension: 40,
            useNativeDriver: true,
          }).start();
        }
        isScaling.current = false;
      },

      onPanResponderTerminationRequest: () => false,
    })
  ).current;

  // Styling for direction indicator
  const getDirectionStyles = () => {
    switch (swipeDirection) {
      case "left":
        return { color: "red", transform: [{ rotate: "0deg" }] };
      case "right":
        return { color: "green", transform: [{ rotate: "180deg" }] };
      case "up":
        return { color: "blue", transform: [{ rotate: "90deg" }] };
      case "down":
        return { color: "orange", transform: [{ rotate: "-90deg" }] };
      default:
        return { opacity: 0 };
    }
  };

  return (
    <View style={styles.container}>
      {/* Render both cards */}
      {[0, 1].map((index) => {
        const isTopCard = index === topCardIndex;

        return (
          <Animated.View
            key={index}
            style={[
              styles.card,
              {
                backgroundColor: cardColors[index],
                height: cardHeight,
                width: cardWidth,
                transform: [
                  { translateX: positions[index].x },
                  { translateY: positions[index].y },
                  { scale: scales[index] }
                ],
                zIndex: isTopCard ? 1 : 0,
                position: "absolute",
              },
            ]}
            {...(isTopCard ? panResponder.panHandlers : {})}
          >
            {isTopCard && swipeDirection && (
              <View style={styles.directionContainer}>
                <Text style={[styles.directionText, getDirectionStyles()]}>
                  SWIPING {swipeDirection.toUpperCase()}
                </Text>
                <Text style={[styles.directionArrow, getDirectionStyles()]}>
                  ➜
                </Text>
              </View>
            )}
            <Text style={styles.cardNumber}>Card {index + 1}</Text>
          </Animated.View>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#f5f5f5",
  },
  card: {
    borderRadius: 10,
    justifyContent: "center",
    alignItems: "center",
    ...Platform.select({
      ios: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 4,
      },
      web: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 4,
      },
      android: {
        elevation: 4,
      },
    }),
  },
  directionContainer: {
    position: "absolute",
    alignItems: "center",
    justifyContent: "center",
  },
  directionText: {
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 10,
  },
  directionArrow: {
    fontSize: 40,
    fontWeight: "bold",
  },
  cardNumber: {
    position: "absolute",
    top: 20,
    fontSize: 20,
    fontWeight: "bold",
  },
});

export default ScalableMovableCard;

