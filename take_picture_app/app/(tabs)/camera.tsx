import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Dimensions,
  SafeAreaView,
} from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { Image } from 'expo-image';
import { BlurView } from 'expo-blur';
import { IconSymbol } from '@/components/ui/icon-symbol';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { Colors } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';

const { width, height } = Dimensions.get('window');

interface PredictionResult {
  success: boolean;
  predicted_class?: string;
  confidence?: number;
  top5_predictions?: Array<{
    class: string;
    confidence: number;
  }>;
  error?: string;
}

export default function CameraScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState<CameraType>('back');
  const [isCapturing, setIsCapturing] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const cameraRef = useRef<CameraView>(null);
  const colorScheme = useColorScheme();

  // FastAPI服务器地址 - 请根据实际情况修改
  const API_BASE_URL = 'http://192.168.162.192:8000'; // 替换为你的服务器IP

  useEffect(() => {
    if (!permission?.granted) {
      requestPermission();
    }
  }, [permission, requestPermission]);

  const toggleCameraFacing = () => {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  };

  const takePicture = async () => {
    if (!cameraRef.current) return;
    
    try {
      setIsCapturing(true);
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        base64: false,
      });
      
      if (photo?.uri) {
        setCapturedImage(photo.uri);
        setPrediction(null);
      }
    } catch (error) {
      console.error('拍照失败:', error);
      Alert.alert('错误', '拍照失败，请重试');
    } finally {
      setIsCapturing(false);
    }
  };

  const uploadAndPredict = async () => {
    if (!capturedImage) return;

    try {
      setIsUploading(true);
      
      // 创建FormData
      const formData = new FormData();
      formData.append('file', {
        uri: capturedImage,
        type: 'image/jpeg',
        name: 'bird_photo.jpg',
      } as any);

      // 发送请求到FastAPI服务器
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result: PredictionResult = await response.json();
      setPrediction(result);
      
      if (result.success) {
        Alert.alert(
          '识别成功！',
          `识别结果：${result.predicted_class}\n置信度：${(result.confidence! * 100).toFixed(2)}%`,
          [{ text: '确定' }]
        );
      } else {
        Alert.alert('识别失败', result.error || '未知错误');
      }
    } catch (error) {
      console.error('上传失败:', error);
      Alert.alert('上传失败', '无法连接到服务器，请检查网络连接');
    } finally {
      setIsUploading(false);
    }
  };

  const retakePicture = () => {
    setCapturedImage(null);
    setPrediction(null);
  };

  if (!permission) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.permissionContainer}>
          <ActivityIndicator size="large" color={Colors[colorScheme ?? 'light'].tint} />
          <ThemedText>正在请求相机权限...</ThemedText>
        </View>
      </SafeAreaView>
    );
  }

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.permissionContainer}>
          <IconSymbol name="camera.fill" size={80} color={Colors[colorScheme ?? 'light'].tint} />
          <ThemedText type="title" style={styles.permissionTitle}>
            需要相机权限
          </ThemedText>
          <ThemedText style={styles.permissionText}>
            请允许访问相机以拍摄鸟类照片进行识别
          </ThemedText>
          <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
            <ThemedText style={styles.permissionButtonText}>授权相机权限</ThemedText>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {!capturedImage ? (
        <View style={styles.cameraContainer}>
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing={facing}
            mode="picture"
          >
            <View style={styles.cameraOverlay}>
              {/* 顶部左上角小相机图标 */}
              <View style={styles.smallIconContainer}>
                <IconSymbol name="camera.fill" size={18} color="#ffffff" />
              </View>
              <BlurView intensity={25} tint={colorScheme === 'dark' ? 'dark' : 'light'} style={styles.topControls}>
                <TouchableOpacity style={styles.glassBtn} onPress={toggleCameraFacing}>
                  <IconSymbol name="arrow.triangle.2.circlepath.camera" size={22} color="white" />
                </TouchableOpacity>
              </BlurView>
              
              <View style={styles.bottomControls}>
                <View style={styles.captureArea}>
                  <TouchableOpacity
                    style={[styles.captureButton, isCapturing && styles.capturingButton]}
                    onPress={takePicture}
                    disabled={isCapturing}
                  >
                    {isCapturing ? (
                      <ActivityIndicator size="large" color="white" />
                    ) : (
                      <View style={styles.captureButtonInner} />
                    )}
                  </TouchableOpacity>
                </View>
              </View>
            </View>
          </CameraView>
        </View>
      ) : (
        <View style={styles.previewContainer}>
          <Image source={{ uri: capturedImage }} style={styles.previewImage} />
          
              <BlurView intensity={25} tint={colorScheme === 'dark' ? 'dark' : 'light'} style={styles.previewControls}>
                <TouchableOpacity style={[styles.controlButton, styles.glassBtn]} onPress={retakePicture}>
              <IconSymbol name="arrow.clockwise" size={24} color="white" />
              <ThemedText style={styles.controlButtonText}>重拍</ThemedText>
            </TouchableOpacity>
            
                <TouchableOpacity
                  style={[styles.controlButton, styles.uploadButton, styles.glassBtn]}
              onPress={uploadAndPredict}
              disabled={isUploading}
            >
              {isUploading ? (
                <ActivityIndicator size="small" color="white" />
              ) : (
                <IconSymbol name="arrow.up.circle.fill" size={24} color="white" />
              )}
              <ThemedText style={styles.controlButtonText}>
                {isUploading ? '识别中...' : '识别鸟类'}
              </ThemedText>
                </TouchableOpacity>
              </BlurView>
        </View>
      )}

      {prediction && (
        <View style={styles.resultContainer}>
          <ThemedText type="subtitle" style={styles.resultTitle}>
            识别结果
          </ThemedText>
          {prediction.success ? (
            <View>
              <ThemedText style={styles.resultText}>
                种类：{prediction.predicted_class}
              </ThemedText>
              <ThemedText style={styles.resultText}>
                置信度：{(prediction.confidence! * 100).toFixed(2)}%
              </ThemedText>
              {prediction.top5_predictions && (
                <View style={styles.top5Container}>
                  <ThemedText style={styles.top5Title}>前5个可能结果：</ThemedText>
                  {prediction.top5_predictions.map((item, index) => (
                    <ThemedText key={index} style={styles.top5Item}>
                      {index + 1}. {item.class} ({(item.confidence * 100).toFixed(2)}%)
                    </ThemedText>
                  ))}
                </View>
              )}
            </View>
          ) : (
            <ThemedText style={styles.errorText}>
              识别失败：{prediction.error}
            </ThemedText>
          )}
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  permissionTitle: {
    marginTop: 20,
    marginBottom: 10,
  },
  permissionText: {
    textAlign: 'center',
    marginBottom: 30,
    opacity: 0.8,
  },
  permissionButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
  },
  permissionButtonText: {
    color: 'white',
    fontWeight: 'bold',
  },
  cameraContainer: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  cameraOverlay: {
    flex: 1,
    backgroundColor: 'transparent',
  },
  topControls: {
    position: 'absolute',
    top: 50,
    left: 20,
    right: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
    zIndex: 1,
    borderRadius: 18,
    padding: 8,
  },
  smallIconContainer: {
    position: 'absolute',
    top: 12,
    left: 12,
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: 'rgba(0,0,0,0.4)',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 2,
  },
  galleryButton: {
    backgroundColor: 'rgba(0,0,0,0.5)',
    padding: 15,
    borderRadius: 25,
  },
  flipButton: {
    backgroundColor: 'rgba(0,0,0,0.5)',
    padding: 15,
    borderRadius: 25,
  },
  glassBtn: {
    backgroundColor: 'rgba(255,255,255,0.12)',
    borderRadius: 12,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
  },
  bottomControls: {
    position: 'absolute',
    bottom: 50,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  captureArea: {
    alignItems: 'center',
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255,255,255,0.3)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: 'white',
  },
  capturingButton: {
    backgroundColor: 'rgba(255,255,255,0.1)',
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: 'white',
  },
  previewContainer: {
    flex: 1,
  },
  previewImage: {
    flex: 1,
    width: '100%',
  },
  previewControls: {
    position: 'absolute',
    bottom: 50,
    left: 20,
    right: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 18,
  },
  controlButton: {
    backgroundColor: 'rgba(0,0,0,0.7)',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderRadius: 25,
    alignItems: 'center',
    minWidth: 100,
  },
  uploadButton: {
    backgroundColor: '#007AFF',
  },
  controlButtonText: {
    color: 'white',
    marginTop: 5,
    fontSize: 12,
    fontWeight: 'bold',
  },
  resultContainer: {
    position: 'absolute',
    top: 50,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(0,0,0,0.8)',
    padding: 20,
    borderRadius: 15,
  },
  resultTitle: {
    color: 'white',
    marginBottom: 10,
    textAlign: 'center',
  },
  resultText: {
    color: 'white',
    fontSize: 16,
    marginBottom: 5,
  },
  errorText: {
    color: '#FF3B30',
    fontSize: 16,
  },
  top5Container: {
    marginTop: 15,
  },
  top5Title: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  top5Item: {
    color: 'white',
    fontSize: 12,
    marginBottom: 3,
    opacity: 0.8,
  },
});
