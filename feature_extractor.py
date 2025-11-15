import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import torch
from transformers import BertTokenizer, BertModel
import jieba


class EmotionLexicon:
    def __init__(self):
        # 基础情感词典
        self.anger_words = {"怒", "愤", "气", "恼", "恨", "怒火", "气愤", "怒斥", "狂怒", "震怒",
                            "嗔怒", "义愤", "愤懑", "恼羞成怒", "怒发冲冠", "火冒三丈", "七窍生烟", "咬牙切齿",
                            "怒不可遏", "勃然大怒",
                            "大发雷霆", "怒目而视", "愤愤不平", "怒形于色", "怒喝", "怒骂", "怒视", "怒容", "怒意",
                            "怒焰",
                            "怒潮", "怒涛", "怒号", "怒叱", "怒打", "怒砸", "怒烧", "怒杀", "怒怼", "怒诉",
                            "怒怨", "怒恨", "怒狂", "怒暴", "怒愤", "怒恼", "怒躁", "怒哮", "怒轰", "炸"}
        self.fear_words = {"惧", "怕", "畏", "吓", "惊慌", "恐怖", "恐慌", "惊惶", "胆怯", "心悸",
                           "毛骨悚然", "不寒而栗", "战战兢兢", "提心吊胆", "惊心动魄", "丧胆", "骇人", "惊骇", "惊恐",
                           "惊愕",
                           "惊惶失措", "惶恐", "惴惴不安", "畏缩", "畏懼", "怖畏", "惊栗", "惊颤", "惊逃", "惊魂",
                           "惊悚", "惊险", "惊疑", "惊异", "惊怕", "惊惧", "惊惶不安", "惊恐万状", "胆战心惊",
                           "闻风丧胆",
                           "畏罪", "畏难", "畏光", "畏寒", "畏缩不前", "畏首畏尾", "惊弓之鸟", "草木皆兵", "面如土色",
                           "魂飞魄散"}
        self.sadness_words = {"悲", "伤", "哀", "愁", "怨", "难过", "悲痛", "哭", "哀痛", "哀思",
                              "哀鸣", "哀悼", "哀怜", "哀婉", "哀叹", "哀号", "哀泣", "哀恸", "哀凄", "哀郁",
                              "哀默", "哀毁", "哀感", "哀念", "哀愤", "哀苦", "哀酸", "哀凉", "哀寂", "哀绝",
                              "哀沉", "哀痛欲绝", "悲痛欲绝", "伤心欲绝", "肝肠寸断", "心如刀割", "泪如雨下",
                              "痛哭流涕", "悲从中来",
                              "悲愤交加", "悲天悯人", "悲歌", "悲壮", "悲惨", "悲凉", "悲苦", "悲戚", "不想", "悲郁",
                              "悲鸣"}
        self.joy_words = {"愉悦", "快乐", "高兴", "开心", "喜悦", "欢欣", "欢喜", "欢乐", "欢快", "欢畅",
                          "欢腾", "欢跃", "欢欣鼓舞", "兴高采烈", "喜出望外", "心花怒放", "乐不可支", "其乐融融",
                          "怡然自得",
                          "得意洋洋", "欣慰", "满足", "幸福", "幸福感", "愉快", "畅快", "爽快", "舒心", "舒畅", "舒坦",
                          "轻松", "轻快", "欣喜", "笑", "欣欢", "欣幸", "欣忭", "欣豫", "欣乐", "欣怡", "喜欢", "美",
                          "好",
                          "欣然", "欣快", "欣悦不已", "喜滋滋", "喜洋洋", "喜上眉梢", "喜笑颜开", "喜气洋洋",
                          "喜不自胜", "喜极而泣"}
        self.disgust_words = {"恶心", "厌恶", "讨厌", "憎恶", "反感", "嫌弃", "憎恨", "痛恨", "抗拒", "深恶痛绝",
                              "作呕", "反胃", "吐", "不适", "难受", "不舒服", "鄙弃", "唾弃", "蔑视", "轻视",
                              "鄙视", "鄙夷", "不屑", "厌烦", "厌倦", "厌弃", "厌世", "厌食", "厌氧", "厌战",
                              "厌学", "厌工", "厌俗", "厌旧", "不想", "厌烦不已", "厌恶至极", "恶心巴拉", "令人作呕",
                              "不堪入目",
                              "不堪入耳", "臭不可闻", "脏乱差", "污秽", "肮脏", "龌龊", "猥琐", "下流", "低俗"}
        self.surprise_words = {"惊喜", "惊讶", "惊奇", "惊异", "诧异", "愕然", "吃惊", "惊呆", "震惊", "震撼",
                               "惊诧", "惊愕", "惊喜欢呼", "喜出望外", "喜从天降", "喜不自胜", "乐不可支", "兴高采烈",
                               "心花怒放",
                               "欢欣鼓舞",
                               "喜极而泣", "喜笑颜开", "喜气洋洋", "喜上眉梢", "喜洋洋", "喜滋滋", "欣然", "欣慰",
                               "满意", "满足",
                               "幸福感", "快乐", "高兴", "开心", "愉悦", "喜悦", "欢快", "欢畅", "欢腾", "欢跃",
                               "惊喜交加", "惊喜若狂", "惊喜万分", "惊喜不已", "惊喜欢悦", "惊喜连连", "惊喜不断",
                               "惊喜时刻", "惊喜礼物",
                               "惊喜派对"}

        # 否定词列表
        self.negation_words = {
            '不', '没', '无', '非', '未', '勿', '莫', '没有', '别', '未',
            '不要', '不用', '不必', '未能', '无法', '不会', '不可', '不能',
            '绝不', '从不', '毫无'
        }

        # 情感强度词
        self.intensity_words = {
            '非常': 2.0, '特别': 2.0, '极其': 2.5, '十分': 1.8, '相当': 1.5,
            '有点': 0.5, '稍微': 0.3, '略微': 0.3, '超级': 2.2, '极度': 2.5,
            '太': 2.0, '真': 1.5, '很': 1.2, '挺': 1.0, '极': 2.5, '异常': 2.0
        }

    def extract_emotion_features_with_negation(self, text, processed_text):
        """考虑否定词的情感特征提取"""
        tokens = processed_text.split()

        features = np.zeros(6)  # 6种情感
        current_intensity = 1.0
        negation_active = False

        for token in tokens:
            if token in self.intensity_words:
                current_intensity = self.intensity_words[token]
                continue

            if token == 'NOT' or token in self.negation_words:
                negation_active = True
                continue

            emotion_scores = np.zeros(6)
            if token in self.anger_words:
                emotion_scores[0] = current_intensity
            elif token in self.fear_words:
                emotion_scores[1] = current_intensity
            elif token in self.sadness_words:
                emotion_scores[2] = current_intensity
            elif token in self.joy_words:
                emotion_scores[3] = current_intensity
            elif token in self.disgust_words:
                emotion_scores[4] = current_intensity
            elif token in self.surprise_words:
                emotion_scores[5] = current_intensity

            if negation_active and np.sum(emotion_scores) > 0:
                reversed_scores = np.zeros(6)
                reversed_scores[0] = emotion_scores[3]
                reversed_scores[1] = emotion_scores[5]
                reversed_scores[2] = emotion_scores[4]
                reversed_scores[3] = emotion_scores[0]
                reversed_scores[4] = emotion_scores[2]
                reversed_scores[5] = emotion_scores[1]

                features += reversed_scores
                negation_active = False
            else:
                features += emotion_scores

            current_intensity = 1.0

        if np.sum(features) > 0:
            features = features / np.sum(features)

        return features

    def extract_emotion_features(self, text):
        """普通情感特征提取"""
        words = jieba.lcut(text)
        features = np.zeros(6)
        current_intensity = 1.0

        for word in words:
            if word in self.intensity_words:
                current_intensity = self.intensity_words[word]
                continue

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

            current_intensity = 1.0

        if np.sum(features) > 0:
            features = features / np.sum(features)

        return features


class ChineseFeatureExtractor:
    def __init__(self, method='combined', use_negation=True):
        self.method = method
        self.use_negation = use_negation
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
                min_df=1,
                max_df=0.9,
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

            encoded_input = self.bert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )

            with torch.no_grad():
                outputs = self.bert_model(**encoded_input)
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

    def extract_lexicon_features_with_negation(self, texts, processed_texts):
        """考虑否定词的情感词典特征提取"""
        features = []
        for i, text in enumerate(texts):
            processed_text = processed_texts[i] if i < len(processed_texts) else text
            lex_feat = self.emotion_lexicon.extract_emotion_features_with_negation(text, processed_text)
            features.append(lex_feat)
        return np.array(features)

    def extract_combined_features(self, texts):
        """结合TF-IDF和情感词典特征"""
        tfidf_features = self.extract_tfidf_features(texts)
        lexicon_features = self.extract_lexicon_features(texts)

        if hasattr(tfidf_features, 'toarray'):
            tfidf_dense = tfidf_features.toarray()
        else:
            tfidf_dense = tfidf_features

        combined_features = np.hstack([tfidf_dense, lexicon_features])
        return combined_features

    def extract_combined_features_with_negation(self, texts, processed_texts):
        """考虑否定词的组合特征提取"""
        tfidf_features = self.extract_tfidf_features(texts)
        lexicon_features = self.extract_lexicon_features_with_negation(texts, processed_texts)

        if hasattr(tfidf_features, 'toarray'):
            tfidf_dense = tfidf_features.toarray()
        else:
            tfidf_dense = tfidf_features

        combined_features = np.hstack([tfidf_dense, lexicon_features])
        return combined_features

    def extract_features(self, texts, processed_texts=None):
        """统一的特征提取方法"""
        if self.use_negation:
            return self.extract_features_with_negation(texts, processed_texts or texts)
        else:
            return self.extract_features_normal(texts)

    def extract_features_with_negation(self, texts, processed_texts):
        """带否定词处理的特征提取"""
        if self.method == 'tfidf':
            return self.extract_tfidf_features(texts)
        elif self.method == 'bert':
            return self.extract_bert_features(texts)
        elif self.method == 'lexicon':
            return self.extract_lexicon_features_with_negation(texts, processed_texts)
        elif self.method == 'combined':
            return self.extract_combined_features_with_negation(texts, processed_texts)
        else:
            raise ValueError(f"不支持的特征提取方法: {self.method}")

    def extract_features_normal(self, texts):
        """普通特征提取"""
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