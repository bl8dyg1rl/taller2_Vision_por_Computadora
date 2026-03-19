import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


def load_images(img_directorio_path:str, classes:list):
    '''Recibe una carpeta con subcarpetas de cada clase y retorna un dataframe'''
    images_color = []
    images_gray = []
    labels = []
    for clase in classes:
        class_dir = os.path.join(img_directorio_path, clase)
        for archivo in os.listdir(class_dir):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                ruta_img = os.path.join(class_dir, archivo)
                images_color.append(cv2.imread(ruta_img, cv2.IMREAD_COLOR_RGB))
                images_gray.append(cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE))
                labels.append(clase)

    df = pd.DataFrame({'imagen color':images_color, 
                       'imagen gris': images_gray,
                       'clase':labels})
    return df


def show_images(lista_imagenes: list, nombres: list, gray: bool)->None: 
    '''Recibe una lista de listas, los nombres de cada lista y los muestra en subplots (filas = clases, columnas = imágenes)'''
    fig, axes = plt.subplots(len(lista_imagenes[0]), len(lista_imagenes), figsize=(2 * len(lista_imagenes), 2 * len(lista_imagenes[0])))

    for i in range(len(lista_imagenes)):
        for j in range(len(lista_imagenes[0])):
            axes[j, i].imshow(lista_imagenes[i][j], cmap='gray' if gray else None)
            axes[j, i].set_title(nombres[i] if j == 0 else "")
            axes[j, i].axis('off')

    plt.tight_layout()
    plt.show()



class ImagePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, brightness=1.0, contrast=1.0, blur=(3, 0), sobel=3, canny=(100, 300), otsu=False, laplacian=3):
        self.brightness = brightness
        self.contrast = contrast
        self.blur = blur
        self.sobel = sobel
        self.canny = canny
        self.otsu = otsu
        self.laplacian = laplacian

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        processed = []
        for img in X:
            img = cv2.convertScaleAbs(img, alpha=self.brightness, beta=self.contrast)

            if self.blur != (0,0):
                img = cv2.GaussianBlur(img, (self.blur[0], self.blur[0]), self.blur[1])

            if self.sobel != 0:
                sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.sobel)
                sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.sobel)
                img = cv2.magnitude(sobelx, sobely).astype(np.uint8)

            if self.canny != (0,0):
                img = cv2.Canny(img, self.canny[0], self.canny[1])

            if self.otsu:
                _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if self.laplacian != 0:
                img = cv2.Laplacian(img, cv2.CV_64F, ksize=self.laplacian)

            processed.append(img.astype(np.uint8))
        return processed




class BagOfVisualWords(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=3, feature_detector='SIFT', max_descriptors=1000):
        self.n_clusters = n_clusters
        self.feature_detector = feature_detector
        self.max_descriptors = max_descriptors


    def _get_detector(self):
        if self.feature_detector == 'SIFT':
            return cv2.SIFT_create()
        elif self.feature_detector == 'ORB':
            return cv2.ORB_create()
        else:
            raise ValueError(f'Detector "{self.feature_detector}" no soportado')
        
    def fit(self, X, y=None):
        detector = self._get_detector()
        all_descriptors = []

        for img in X:
            keypoints, descriptors = detector.detectAndCompute(img, None)
            
            if descriptors is not None:
                keypoints_ordenados = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:self.max_descriptors]
                descriptors_ordenados = np.array([descriptors[keypoints.index(kp)] for kp in keypoints_ordenados])
                all_descriptors.append(descriptors_ordenados)

        # Concatenar todos los descriptores en una sola matriz
        all_descriptors = np.vstack(all_descriptors)
        
        # Clustering con Kmeans
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(all_descriptors)
        return self

    def transform(self, X, y=None):
        detector = self._get_detector()
        histograms = []

        for img in X:
            keypoints, descriptors = detector.detectAndCompute(img, None)

            if descriptors is not None:
                keypoints_ordenados = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:self.max_descriptors]
                descriptors_ordenados = np.array([descriptors[keypoints.index(kp)] for kp in keypoints_ordenados])
                words = self.kmeans.predict(descriptors_ordenados)
                hist, _ = np.histogram(words, bins=np.arange(self.n_clusters + 1))
            else:
                hist = np.zeros(self.n_clusters)

            histograms.append(hist)

        return np.array(histograms)
        


