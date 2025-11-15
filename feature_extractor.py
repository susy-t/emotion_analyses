import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import torch
from transformers import BertTokenizer, BertModel
import jieba


class EmotionLexicon:
    def __init__(self):
        # 基础情感词典（在实际应用中应该使用更完整的情感词典）
        self.anger_words = {'愤怒', '生气', '气愤', '怒火', '发怒', '暴怒', '恼火', '气愤', '愤怒', '怒气', '气死',
                            '火冒三丈', '大怒'}
        self.fear_words = {'恐惧', '害怕', '恐怖', '惊吓', '恐慌', '畏惧', '胆怯', '可怕', '吓人', '惊恐', '毛骨悚然',
                           '胆战心惊'}
        self.sadness_words = {'悲伤', '伤心', '难过', '悲哀', '悲痛', '沮丧', '忧郁', '哀伤', '伤感', '心碎', '低落',
                              '悲痛欲绝'}
        self.joy_words = {'愉悦', '开心', '高兴', '快乐', '喜悦', '欢乐', '欣喜', '愉快', '兴奋', '幸福', '满足',
                          '欣喜若狂'}
        self.disgust_words = {'恶心', '厌恶', '反感', '讨厌', '作呕', '嫌弃', '憎恶', '反胃', '反感', '受不了'}
        self.surprise_words = {'惊喜', '惊讶', '惊奇', '吃惊', '意外', '震惊', '诧异', '出乎意料', '大吃一惊',
                               '措手不及'}

        # 情感强度词
        self.intensity_words = {
            '非常': 2.0, '特别': 2.0, '极其': 2.5, '十分': 1.8, '相当': 1.5,
            '有点': 0.5, '稍微': 0.3, '略微': 0.3, '超级': 2.2, '极度': 2.5,
            '太': 2.0, '真': 1.5, '很': 1.2, '挺': 1.0, '极': 2.5, '异常': 2.0
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

        # 归一化特征
        if np.sum(features) > 0:
            features = features / np.sum(features)

        return features


class ChineseFeatureExtractor:
    def __init__(self, method='combined'):
        self.method = method
        self.vectorizer = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.emotion_lexicon = EmotionLexicon()

        if method == 'bert':
            self._load_bert_model()

    def _load_bert_model(self):
        """加载中文BERT模型"""
        try:
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            self.bert_model = BertModel.from_pretrained('bert-base-chinese')
            print("中文BERT模型加载成功")
        except Exception as e:
            print(f"加载BERT模型失败: {e}")
            print("将使用TF-IDF特征")
            self.method = 'tfidf'

    def extract_tfidf_features(self, texts, max_features=5000):
        """提取TF-IDF特征"""
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                token_pattern=r'(?u)\b\w+\b'
            )
            features = self.vectorizer.fit_transform(texts)
        else:
            features = self.vectorizer.transform(texts)
        return features

    def extract_bert_features(self, texts, batch_size=32):
        """提取BERT特征"""
        if self.bert_model is None:
            raise ValueError("BERT模型未加载")

        features = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # 编码文本
            encoded_input = self.bert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )

            # 获取BERT输出
            with torch.no_grad():
                outputs = self.bert_model(**encoded_input)
                # 使用[CLS] token的表示作为整个文本的表示
                batch_features = outputs.last_hidden_state[:, 0, :].numpy()
                features.append(batch_features)

        return np.vstack(features)

    def extract_lexicon_features(self, texts):
        """提取情感词典特征"""
        features = []
        for text in texts:
            lex_feat = self.emotion_lexicon.extract_emotion_features(text)
            features.append(lex_feat)
        return np.array(features)

    def extract_combined_features(self, texts):
        """结合TF-IDF和情感词典特征"""
        # TF-IDF特征
        tfidf_features = self.extract_tfidf_features(texts)

        # 情感词典特征
        lexicon_features = self.extract_lexicon_features(texts)

        # 合并特征
        if hasattr(tfidf_features, 'toarray'):
            tfidf_dense = tfidf_features.toarray()
        else:
            tfidf_dense = tfidf_features

        combined_features = np.hstack([tfidf_dense, lexicon_features])
        return combined_features

    def extract_features(self, texts):
        """根据配置的方法提取特征"""
        if self.method == 'tfidf':
            return self.extract_tfidf_features(texts)
        elif self.method == 'bert':
            return self.extract_bert_features(texts)
        elif self.method == 'lexicon':
            return self.extract_lexicon_features(texts)
        elif self.method == 'combined':
            return self.extract_combined_features(texts)
        else:
            raise ValueError(f"不支持的特征提取方法: {self.method}")