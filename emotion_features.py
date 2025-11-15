import jieba
import numpy as np


class EmotionLexicon:
    def __init__(self):
        # 基础情感词典（在实际应用中应该使用更完整的情感词典）
        self.anger_words = {'愤怒', '生气', '气愤', '怒火', '发怒', '暴怒', '恼火', '气愤', '愤怒', '怒气'}
        self.fear_words = {'恐惧', '害怕', '恐怖', '惊吓', '恐慌', '畏惧', '胆怯', '可怕', '吓人'}
        self.sadness_words = {'悲伤', '伤心', '难过', '悲哀', '悲痛', '沮丧', '忧郁', '哀伤', '伤感'}
        self.joy_words = {'愉悦', '开心', '高兴', '快乐', '喜悦', '欢乐', '欣喜', '愉快', '兴奋'}
        self.disgust_words = {'恶心', '厌恶', '反感', '讨厌', '作呕', '嫌弃', '憎恶'}
        self.surprise_words = {'惊喜', '惊讶', '惊奇', '吃惊', '意外', '震惊', '诧异'}

        # 情感强度词
        self.intensity_words = {
            '非常': 2.0, '特别': 2.0, '极其': 2.5, '十分': 1.8, '相当': 1.5,
            '有点': 0.5, '稍微': 0.3, '略微': 0.3, '超级': 2.2, '极度': 2.5,
            '太': 2.0, '真': 1.5, '很': 1.2, '挺': 1.0
        }

    def extract_emotion_features(self, text):
        """提取基于情感词典的特征"""
        words = jieba.lcut(text)

        features = np.zeros(6)  # 6种情感

        current_intensity = 1.0

        for word in words:
            # 检查是否是强度词
            if word in self.intensity_words:
                current_intensity = self.intensity_words[word]
                continue

            # 检查情感词
            if word in self.anger_words:
                features[0] += current_intensity
            elif word in self.fear_words:
                features[1] += current_intensity
            elif word in self.sadness_words:
                features[2] += current_intensity
            elif word in self.joy_words:
                features[3] += current_intensity
            elif word in self.disgust_words:
                features[4] += current_intensity
            elif word in self.surprise_words:
                features[5] += current_intensity

            # 重置强度
            current_intensity = 1.0

        return features