import { Image } from 'expo-image';
import { BlurView } from 'expo-blur';
import { Platform, StyleSheet, TouchableOpacity, View } from 'react-native';

import { HelloWave } from '@/components/hello-wave';
import ParallaxScrollView from '@/components/parallax-scroll-view';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { Link, router } from 'expo-router';
import { IconSymbol } from '@/components/ui/icon-symbol';
import { Colors } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';

export default function HomeScreen() {
  const colorScheme = useColorScheme();

  return (
    <ParallaxScrollView
      headerBackgroundColor={{ light: '#0ea5e9', dark: '#0b1620' }}
      headerImage={
        <View style={styles.heroHeader}>
          <Image
            source={require('@/assets/images/partial-react-logo.png')}
            style={styles.reactLogo}
          />
          <BlurView intensity={40} tint={colorScheme === 'dark' ? 'dark' : 'light'} style={styles.heroBlur} />
          <View style={styles.heroContent}>
            <ThemedText type="title" style={styles.heroTitle}>BirdLens</ThemedText>
            <ThemedText style={styles.heroSubtitle}>拍一下，识鸟类</ThemedText>
            <TouchableOpacity
              style={[styles.primaryButton, { backgroundColor: Colors[colorScheme ?? 'light'].tint }]}
              onPress={() => router.push('/(tabs)/camera')}
            >
              <IconSymbol name="camera.fill" size={18} color="white" />
              <ThemedText style={styles.primaryButtonText}>开始识别</ThemedText>
            </TouchableOpacity>
          </View>
        </View>
      }>
      <View style={styles.grid}>
        <ThemedView style={styles.miniCard}>
          <IconSymbol name="sparkles" size={16} color={Colors[colorScheme ?? 'light'].tint} />
          <ThemedText style={styles.miniCardText}>ResNetBrid Model</ThemedText>
        </ThemedView>
        <ThemedView style={styles.miniCard}>
          <IconSymbol name="photo" size={16} color={Colors[colorScheme ?? 'light'].tint} />
          <ThemedText style={styles.miniCardText}>支持拍照</ThemedText>
        </ThemedView>
      </View>
      <BlurView intensity={30} tint={colorScheme === 'dark' ? 'dark' : 'light'} style={styles.glassCard}>
        <ThemedText style={styles.cardBodyText}>
          支持10类海鸟：黑脚信天翁、布鲁尔黑鸟、冠海雀、沟嘴阿尼鸟、莱桑信天翁等。
        </ThemedText>
      </BlurView>
    </ParallaxScrollView>
  );
}

const styles = StyleSheet.create({
  titleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  stepContainer: {
    gap: 8,
    marginBottom: 8,
  },
  reactLogo: {
    height: 178,
    width: 290,
    bottom: 0,
    left: 0,
    position: 'absolute',
  },
  heroHeader: {
    flex: 1,
    height: 230,
    justifyContent: 'flex-end',
  },
  heroBlur: {
    ...Platform.select({
      ios: { shadowColor: '#000', shadowOpacity: 0.2, shadowRadius: 12, shadowOffset: { width: 0, height: 6 } },
      android: { elevation: 2 },
      default: {},
    }),
    ...StyleSheet.absoluteFillObject,
    borderBottomLeftRadius: 18,
    borderBottomRightRadius: 18,
  },
  heroContent: {
    paddingHorizontal: 18,
    paddingBottom: 22,
    gap: 8,
  },
  heroTitle: {
    fontSize: 28,
    fontWeight: '800',
  },
  heroSubtitle: {
    opacity: 0.9,
  },
  card: {
    borderRadius: 14,
    padding: 14,
    marginBottom: 12,
    backgroundColor: 'rgba(0,0,0,0.04)',
  },
  cardPlain: {
    borderRadius: 14,
    padding: 14,
    marginTop: 8,
    marginBottom: 20,
    backgroundColor: 'transparent',
  },
  glassCard: {
    borderRadius: 14,
    padding: 14,
    marginTop: 8,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.25)',
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  cardHeaderText: {
    fontWeight: '600',
  },
  cardBodyText: {
    opacity: 0.9,
    lineHeight: 20,
  },
  ctaContainer: {
    marginTop: 6,
    marginBottom: 20,
    alignItems: 'center',
  },
  primaryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    paddingHorizontal: 26,
    borderRadius: 28,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 3,
    gap: 8,
  },
  primaryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  grid: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 12,
    marginBottom: 8,
  },
  miniCard: {
    flex: 1,
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 12,
    backgroundColor: 'rgba(0,0,0,0.04)',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  miniCardText: {
    fontWeight: '600',
  },
});
