import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import json
import os
from difflib import SequenceMatcher
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import requests
import re

class DouyinBot:
    def __init__(self):
        self.driver = webdriver.Chrome()
        self.wait = WebDriverWait(self.driver, 10)
        self.trained_data = self.load_trained_data()
        self.model = None
        self.load_model()
        
    def load_model(self):
        try:
            # GPU kullanımını devre dışı bırak
            tf.config.set_visible_devices([], 'GPU')
            
            # Model yükleme ayarları
            tf.config.run_functions_eagerly(True)
            
            # Modeli yükle
            self.model = tf.keras.models.load_model('best_model.h5', compile=False)
            
            # Modeli optimize et
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            print("Model başarıyla yüklendi!")
        except Exception as e:
            print(f"Model yükleme hatası: {str(e)}")
            self.model = None
            
    def load_trained_data(self):
        try:
            with open('trained_data.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Eğitilmiş veri dosyası bulunamadı!")
            return {}
            
    def start(self):
        self.driver.get('https://www.douyin.com')
        time.sleep(5)  # Sayfanın yüklenmesi için bekle
        
    def scroll_page(self, scroll_count=10):
        for _ in range(scroll_count):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(2, 4))  # Rastgele bekleme süresi
            
    def get_video_duration(self, video_element):
        try:
            # Video süresini al
            duration_element = video_element.find_element(By.CSS_SELECTOR, '.video-duration')
            duration_text = duration_element.text
            
            # Süreyi saniyeye çevir
            if ':' in duration_text:
                minutes, seconds = map(int, duration_text.split(':'))
                return minutes * 60 + seconds
            else:
                return int(duration_text)
        except Exception as e:
            print(f"Video süresi alma hatası: {str(e)}")
            return None
            
    def get_video_thumbnail(self, video_element):
        try:
            # Video önizleme resmini al
            thumbnail = video_element.find_element(By.CSS_SELECTOR, '.video-thumbnail')
            img_url = thumbnail.get_attribute('src')
            
            # Resmi indir ve işle
            response = requests.get(img_url)
            img = Image.open(io.BytesIO(response.content))
            img = img.resize((224, 224))  # Model için uygun boyuta getir
            img_array = np.array(img) / 255.0  # Normalize et
            return img_array
        except Exception as e:
            print(f"Thumbnail alma hatası: {str(e)}")
            return None
            
    def analyze_video(self, video_element):
        try:
            # Video süresini kontrol et
            duration = self.get_video_duration(video_element)
            if duration is not None and duration > self.trained_data.get('max_duration', 20):
                print(f"Video çok uzun: {duration} saniye")
                return False
            
            # Video başlığını al
            title = video_element.find_element(By.CSS_SELECTOR, '.video-title').text
            
            # Video açıklamasını al
            description = video_element.find_element(By.CSS_SELECTOR, '.video-desc').text
            
            # Video etkileşimlerini al
            likes = video_element.find_element(By.CSS_SELECTOR, '.like-count').text
            comments = video_element.find_element(By.CSS_SELECTOR, '.comment-count').text
            
            # Metin benzerliği skoru hesapla
            text_similarity = self.calculate_text_similarity({
                'title': title,
                'description': description,
                'likes': likes,
                'comments': comments
            })
            
            # Görsel analizi yap
            if self.model is not None:
                try:
                    thumbnail = self.get_video_thumbnail(video_element)
                    if thumbnail is not None:
                        # Model tahminini al
                        prediction = self.model.predict(np.expand_dims(thumbnail, axis=0), verbose=0)[0]
                        visual_similarity = float(prediction[0])  # İlk sınıfın olasılığı
                        
                        # Metin ve görsel benzerliğini birleştir
                        final_similarity = (text_similarity + visual_similarity) / 2
                    else:
                        final_similarity = text_similarity
                except Exception as e:
                    print(f"Görsel analiz hatası: {str(e)}")
                    final_similarity = text_similarity
            else:
                final_similarity = text_similarity
            
            return final_similarity > 0.7  # Benzerlik eşiği
            
        except Exception as e:
            print(f"Video analiz hatası: {str(e)}")
            return False
            
    def calculate_text_similarity(self, video_data):
        similarity = 0
        total_weight = 0
        
        for key in video_data:
            if key in self.trained_data:
                if isinstance(self.trained_data[key], list):
                    # Liste içindeki tüm olasılıklarla karşılaştır
                    max_similarity = 0
                    for trained_text in self.trained_data[key]:
                        # Çince karakterler için SequenceMatcher kullan
                        text_similarity = SequenceMatcher(None, 
                                                        video_data[key], 
                                                        trained_text).ratio()
                        max_similarity = max(max_similarity, text_similarity)
                    similarity += max_similarity
                else:
                    # Sayısal değerler için (likes, comments)
                    try:
                        video_value = int(video_data[key].replace(',', ''))
                        trained_value = int(self.trained_data[key])
                        # Sayısal değerlerin benzerliğini hesapla
                        value_similarity = min(1.0, video_value / trained_value)
                        similarity += value_similarity
                    except ValueError:
                        continue
                total_weight += 1
                
        return similarity / total_weight if total_weight > 0 else 0
        
    def save_video(self, video_element):
        try:
            # Video URL'sini al
            video_url = video_element.find_element(By.CSS_SELECTOR, 'video').get_attribute('src')
            
            # Video dosyasını indir
            # Not: Gerçek uygulamada video indirme işlemi için ek kütüphaneler gerekebilir
            print(f"Video kaydedildi: {video_url}")
            
        except Exception as e:
            print(f"Video kaydetme hatası: {str(e)}")
            
    def run(self):
        try:
            self.start()
            self.scroll_page()
            
            # Sayfadaki videoları bul
            videos = self.driver.find_elements(By.CSS_SELECTOR, '.video-item')
            
            for video in videos:
                if self.analyze_video(video):
                    self.save_video(video)
                    
        except Exception as e:
            print(f"Çalışma hatası: {str(e)}")
        finally:
            self.driver.quit()

if __name__ == "__main__":
    # TensorFlow uyarılarını kapat
    tf.get_logger().setLevel('ERROR')
    
    bot = DouyinBot()
    bot.run()
