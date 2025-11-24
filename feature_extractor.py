import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import torch
from transformers import BertTokenizer, BertModel
import jieba
import re


class EmotionLexicon:
    def __init__(self):
        # 扩展基础情感词典
        self.anger_words = {
            "怒", "愤", "气", "恼", "恨", "怒火", "气愤", "怒斥", "狂怒", "震怒",
            "嗔怒", "义愤", "愤懑", "恼羞成怒", "怒发冲冠", "火冒三丈", "七窍生烟", "咬牙切齿",
            "怒不可遏", "勃然大怒", "大发雷霆", "怒目而视", "愤愤不平", "怒形于色", "怒喝",
            "怒骂", "怒视", "怒容", "怒意", "怒焰", "怒潮", "怒涛", "怒号", "怒叱",
            "暴躁", "生气", "发火", "光火", "动怒", "激怒", "触怒", "迁怒", "泄愤",
            "气死", "气炸", "火大", "恼火", "发怒", "暴怒", "怒斥", "狂怒", "震怒",
            "愤怒", "发飙", "暴跳如雷", "怒不可遏", "义愤填膺", "怒目圆睁", "怒发冲冠"
        }

        self.fear_words = {
            "惧", "怕", "畏", "吓", "惊慌", "恐怖", "恐慌", "惊惶", "胆怯", "心悸",
            "毛骨悚然", "不寒而栗", "战战兢兢", "提心吊胆", "惊心动魄", "丧胆", "骇人", "惊骇",
            "惊恐", "惊愕", "惊惶失措", "惶恐", "惴惴不安", "畏缩", "畏懼", "怖畏", "惊栗",
            "惊颤", "惊逃", "惊魂", "惊悚", "惊险", "惊疑", "惊异", "惊怕", "惊惧",
            "惊惶不安", "惊恐万状", "胆战心惊", "闻风丧胆", "畏罪", "畏难", "畏光",
            "畏寒", "畏缩不前", "畏首畏尾", "惊弓之鸟", "草木皆兵", "面如土色", "魂飞魄散",
            "慌乱", "冷汗", "浸湿", "模糊不清", "无形", "攥住", "跳动", "冲破", "哨声",
            "尖锐", "哆嗦", "秒针", "空白", "杂乱无章", "血腥味", "哽咽", "考砸", "慌乱",
            "紧张", "不安", "焦虑", "担心", "惶恐", "害怕", "恐惧", "惊慌", "惊恐",
            "吓死", "吓坏", "吓尿", "胆战", "心惊", "害怕", "恐惧", "恐慌", "惊恐"
        }

        self.sadness_words = {
            "悲", "伤", "哀", "愁", "怨", "难过", "悲痛", "哭", "哀痛", "哀思",
            "哀鸣", "哀悼", "哀怜", "哀婉", "哀叹", "哀号", "哀泣", "哀恸", "哀凄",
            "哀郁", "哀默", "哀毁", "哀感", "哀念", "哀愤", "哀苦", "哀酸", "哀凉",
            "哀寂", "哀绝", "哀沉", "哀痛欲绝", "悲痛欲绝", "伤心欲绝", "肝肠寸断",
            "心如刀割", "泪如雨下", "痛哭流涕", "悲从中来", "悲愤交加", "悲天悯人",
            "悲歌", "悲壮", "悲惨", "悲凉", "悲苦", "悲戚", "悲郁", "悲鸣", "忧伤",
            "忧郁", "郁闷", "沮丧", "失落", "绝望", "心碎", "痛苦", "难受", "伤心",
            "难过", "悲哀", "哀伤", "忧伤", "忧郁", "郁闷", "沮丧", "悲伤"
        }

        self.joy_words = {
            "愉悦", "快乐", "高兴", "开心", "喜悦", "欢欣", "欢喜", "欢乐", "欢快",
            "欢畅", "欢腾", "欢跃", "欢欣鼓舞", "兴高采烈", "喜出望外", "心花怒放",
            "乐不可支", "其乐融融", "怡然自得", "得意洋洋", "欣慰", "满足", "幸福",
            "幸福感", "愉快", "畅快", "爽快", "舒心", "舒畅", "舒坦", "轻松", "轻快",
            "欣喜", "笑", "欣欢", "欣幸", "欣忭", "欣豫", "欣乐", "欣怡", "喜欢",
            "美", "好", "欣然", "欣快", "欣悦不已", "喜滋滋", "喜洋洋", "喜上眉梢",
            "喜笑颜开", "喜气洋洋", "喜不自胜", "喜极而泣", "激动", "兴奋", "激动不已",
            "激昂", "一等奖", "聚光灯", "雷鸣", "掌声", "激动", "通红", "挥舞", "校徽",
            "扬起", "笑意", "颤抖", "璀璨", "星光", "欢呼", "庆祝", "胜利", "成就",
            "开心", "高兴", "快乐", "喜悦", "欢乐", "欣喜", "愉快", "兴奋", "幸福"
        }

        self.disgust_words = {
            "恶心", "厌恶", "讨厌", "憎恶", "反感", "嫌弃", "憎恨", "痛恨", "抗拒", "深恶痛绝",
            "作呕", "反胃", "吐", "不适", "难受", "不舒服", "鄙弃", "唾弃", "蔑视", "轻视",
            "鄙视", "鄙夷", "不屑", "厌烦", "厌倦", "厌弃", "厌世", "厌食", "厌氧", "厌战",
            "厌学", "厌工", "厌俗", "厌旧", "厌烦不已", "厌恶至极", "恶心巴拉", "令人作呕",
            "不堪入目", "不堪入耳", "臭不可闻", "脏乱差", "污秽", "肮脏", "龌龊", "猥琐", "下流", "低俗"
        }

        self.surprise_words = {
            "惊喜", "惊讶", "惊奇", "惊异", "诧异", "愕然", "吃惊", "惊呆", "震惊", "震撼",
            "惊诧", "惊愕", "惊喜欢呼", "喜出望外", "喜从天降", "喜不自胜", "乐不可支", "兴高采烈",
            "心花怒放", "欢欣鼓舞", "喜极而泣", "喜笑颜开", "喜气洋洋", "喜上眉梢", "喜洋洋", "喜滋滋",
            "欣然", "欣慰", "满意", "满足", "幸福感", "快乐", "高兴", "开心", "愉悦", "喜悦",
            "欢快", "欢畅", "欢腾", "欢跃", "惊喜交加", "惊喜若狂", "惊喜万分", "惊喜不已",
            "惊喜欢悦", "惊喜连连", "惊喜不断", "惊喜时刻", "惊喜礼物", "惊喜派对"
        }

        # 否定词列表
        self.negation_words = {
            '不', '没', '无', '非', '未', '勿', '莫', '没有', '别', '未',
            '不要', '不用', '不必', '未能', '无法', '不会', '不可', '不能',
            '绝不', '从不', '毫无', '毫无意义', '别想', '休想', '毋', '毋须',
            '无须', '无需', '何必', '何须', '岂', '岂能', '岂可', '绝非',
            '并非', '并无', '毫无', '毫无道理', '毫无意义', '没意思', '不必要',
            '不需要', '不应该', '不可以', '不可能', '不至于', '不至于', '不至于'
        }

        # 修复强度词权重 - 确保合理
        self.intensity_words = {
            '非常': 2.0, '特别': 2.0, '极其': 2.5, '十分': 1.8, '相当': 1.5,
            '有点': 0.5, '稍微': 0.3, '略微': 0.3, '超级': 2.2, '极度': 2.5,
            '太': 2.0, '真': 1.5, '很': 1.5, '挺': 1.2, '极': 2.5, '异常': 2.0,
            '死': 2.5, '炸': 2.5, '超': 1.8, '巨': 1.8, '特': 1.5, '蛮': 1.2,
            '好': 1.5, '超极': 2.2, '超级': 2.2, '贼': 1.8, '超级无敌': 2.5,
            '比较': 1.2, '较为': 1.2, '颇为': 1.5, '颇为': 1.5, '格外': 1.8
        }

        # 添加模式匹配规则
        self.pattern_rules = self._create_pattern_rules()

    def _create_pattern_rules(self):
        """创建基于模式的规则"""
        rules = {
            'anger': [
                r'.*气死.*',
                r'.*烦死了.*',
                r'.*讨厌.*',
                r'.*恨.*',
                r'.*怒火.*',
                r'.*愤怒.*',
                r'.*生气.*',
                r'.*发火.*',
                r'.*暴躁.*'
            ],
            'fear': [
                r'.*吓死.*',
                r'.*害怕.*',
                r'.*恐惧.*',
                r'.*恐怖.*',
                r'.*惊慌.*',
                r'.*紧张.*',
                r'.*不安.*',
                r'.*担心.*',
                r'.*恐慌.*'
            ],
            'sadness': [
                r'.*伤心.*',
                r'.*难过.*',
                r'.*悲哀.*',
                r'.*痛苦.*',
                r'.*泪.*',
                r'.*哭.*',
                r'.*抑郁.*',
                r'.*失落.*',
                r'.*绝望.*',
                r'.*悲伤.*'
            ],
            'joy': [
                r'.*开心.*',
                r'.*高兴.*',
                r'.*快乐.*',
                r'.*喜悦.*',
                r'.*幸福.*',
                r'.*兴奋.*',
                r'.*激动.*',
                r'.*欢笑.*',
                r'.*庆祝.*'
            ],
            'disgust': [
                r'.*恶心.*',
                r'.*厌恶.*',
                r'.*讨厌.*',
                r'.*反感.*',
                r'.*嫌弃.*',
                r'.*憎恶.*',
                r'.*作呕.*'
            ],
            'surprise': [
                r'.*惊喜.*',
                r'.*惊讶.*',
                r'.*惊奇.*',
                r'.*意外.*',
                r'.*震惊.*',
                r'.*诧异.*',
                r'.*居然.*',
                r'.*竟然.*'
            ]
        }
        return rules

    def extract_emotion_features_with_negation(self, text, processed_text):
        """考虑多个否定词的情感特征提取"""
        tokens = processed_text.split()

        features = np.zeros(6)  # 6种情感
        current_intensity = 1.0
        negation_active = False
        negation_count = 0
        last_was_intensity = False
        has_emotion_word = False

        for token in tokens:
            # 处理强度词
            if token in self.intensity_words:
                current_intensity = self.intensity_words[token]
                last_was_intensity = True
                continue

            # 处理否定词标记
            if token.startswith('NOT_'):
                try:
                    # 提取否定词数量
                    count = int(token.split('_')[1])
                    negation_count += count
                    negation_active = (negation_count % 2 == 1)  # 奇数个否定词才反转
                except (IndexError, ValueError):
                    # 如果解析失败，默认当作单个否定词
                    negation_count += 1
                    negation_active = True
                last_was_intensity = False
                continue

            # 处理否定标记开始
            if token == 'NEGATION_START':
                negation_active = True
                last_was_intensity = False
                continue

            # 处理双重否定标记
            if token == 'DOUBLE_NEGATION':
                negation_active = False
                last_was_intensity = False
                continue

            # 处理情感词
            emotion_scores = np.zeros(6)
            if token in self.anger_words:
                emotion_scores[0] = current_intensity
                has_emotion_word = True
            elif token in self.fear_words:
                emotion_scores[1] = current_intensity
                has_emotion_word = True
            elif token in self.sadness_words:
                emotion_scores[2] = current_intensity
                has_emotion_word = True
            elif token in self.joy_words:
                emotion_scores[3] = current_intensity
                has_emotion_word = True
            elif token in self.disgust_words:
                emotion_scores[4] = current_intensity
                has_emotion_word = True
            elif token in self.surprise_words:
                emotion_scores[5] = current_intensity
                has_emotion_word = True

            # 如果有情感分数，处理否定逻辑
            if np.sum(emotion_scores) > 0:
                if negation_active:
                    # 反转情感
                    reversed_scores = np.zeros(6)
                    reversed_scores[0] = emotion_scores[3]  # 愤怒 <-> 愉悦
                    reversed_scores[1] = emotion_scores[5]  # 恐惧 <-> 惊喜
                    reversed_scores[2] = emotion_scores[4]  # 悲伤 <-> 恶心
                    reversed_scores[3] = emotion_scores[0]  # 愉悦 <-> 愤怒
                    reversed_scores[4] = emotion_scores[2]  # 恶心 <-> 悲伤
                    reversed_scores[5] = emotion_scores[1]  # 惊喜 <-> 恐惧

                    features += reversed_scores
                    # 重置否定状态
                    negation_active = False
                    negation_count = 0
                else:
                    features += emotion_scores

                current_intensity = 1.0
                last_was_intensity = False
            else:
                # 如果不是情感词，但有强度词在前面，降低强度
                if last_was_intensity:
                    current_intensity = max(1.0, current_intensity * 0.7)
                last_was_intensity = False

        # 应用模式匹配规则
        pattern_features = self._apply_pattern_rules(text)
        features = features + pattern_features * 0.5  # 给模式匹配规则一定的权重

        # 如果没有检测到情感词，返回均匀分布的低概率
        if not has_emotion_word and np.sum(pattern_features) == 0:
            return np.ones(6) * (1.0 / 6)

        # 归一化
        total = np.sum(features)
        if total > 0:
            features = features / total
        else:
            features = np.ones(6) * (1.0 / 6)

        return features

    def _apply_pattern_rules(self, text):
        """应用模式匹配规则"""
        features = np.zeros(6)

        # 检查每个情感类别的模式
        emotion_mapping = {
            'anger': 0,
            'fear': 1,
            'sadness': 2,
            'joy': 3,
            'disgust': 4,
            'surprise': 5
        }

        for emotion, patterns in self.pattern_rules.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    features[emotion_mapping[emotion]] += 1.0

        return features

    def extract_emotion_features(self, text):
        """普通情感特征提取 - 平衡权重"""
        words = jieba.lcut(text)
        features = np.zeros(6)
        current_intensity = 1.0
        last_was_intensity = False
        has_emotion_word = False

        emotion_counts = np.zeros(6)  # 统计每种情感词出现的次数

        for word in words:
            if word in self.intensity_words:
                current_intensity = self.intensity_words[word]
                last_was_intensity = True
                continue

            emotion_idx = -1
            if word in self.anger_words:
                emotion_idx = 0
                has_emotion_word = True
            elif word in self.fear_words:
                emotion_idx = 1
                has_emotion_word = True
            elif word in self.sadness_words:
                emotion_idx = 2
                has_emotion_word = True
            elif word in self.joy_words:
                emotion_idx = 3
                has_emotion_word = True
            elif word in self.disgust_words:
                emotion_idx = 4
                has_emotion_word = True
            elif word in self.surprise_words:
                emotion_idx = 5
                has_emotion_word = True

            if emotion_idx != -1:
                features[emotion_idx] += current_intensity
                emotion_counts[emotion_idx] += 1
                current_intensity = 1.0
                last_was_intensity = False
            else:
                # 如果不是情感词，但有强度词在前面，降低强度
                if last_was_intensity:
                    current_intensity = max(1.0, current_intensity * 0.7)
                last_was_intensity = False

        # 应用模式匹配规则
        pattern_features = self._apply_pattern_rules(text)
        features = features + pattern_features * 0.5

        # 如果没有检测到情感词，返回均匀分布的低概率
        if not has_emotion_word and np.sum(pattern_features) == 0:
            return np.ones(6) * (1.0 / 6)

        # 应用平衡策略：如果某种情感词出现过多，适当降低其权重
        total_emotion_words = np.sum(emotion_counts)
        if total_emotion_words > 0:
            # 计算每种情感的比例
            emotion_ratios = emotion_counts / total_emotion_words
            # 如果某种情感占比过高（超过50%），适当降低其权重
            for i in range(6):
                if emotion_ratios[i] > 0.5:
                    features[i] *= 0.7  # 降低30%的权重

        # 归一化
        total = np.sum(features)
        if total > 0:
            features = features / total
        else:
            features = np.ones(6) * (1.0 / 6)

        return features

    def rule_based_analysis(self, text):
        """基于规则的情感分析，用于模型不确定时的后备方案"""
        features = self.extract_emotion_features(text)

        # 找到最大概率的情感
        max_idx = np.argmax(features)
        max_prob = features[max_idx]

        emotions = ['愤怒', '恐惧', '悲伤', '愉悦', '恶心', '惊喜']

        # 提高阈值，确保只有明确的情感才被识别
        if max_prob > 0.6:  # 提高阈值到0.6
            return {
                'emotion': emotions[max_idx],
                'probability': max_prob,
                'confidence': 'rule_based',
                'all_probabilities': features
            }
        else:
            return {
                'emotion': None,
                'probability': 0.0,
                'confidence': 'uncertain',
                'all_probabilities': features
            }


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

    def rule_based_analysis(self, texts):
        """基于规则的情感分析"""
        results = []
        for text in texts:
            result = self.emotion_lexicon.rule_based_analysis(text)
            results.append(result)
        return results

    def extract_combined_features(self, texts):
        """结合TF-IDF和情感词典特征 - 调整权重"""
        tfidf_features = self.extract_tfidf_features(texts)
        lexicon_features = self.extract_lexicon_features(texts)

        if hasattr(tfidf_features, 'toarray'):
            tfidf_dense = tfidf_features.toarray()
        else:
            tfidf_dense = tfidf_features

        # 给情感特征更高的权重
        weighted_lexicon_features = lexicon_features * 3.0  # 增加情感特征的权重

        combined_features = np.hstack([tfidf_dense, weighted_lexicon_features])
        return combined_features

    def extract_combined_features_with_negation(self, texts, processed_texts):
        """考虑否定词的组合特征提取"""
        tfidf_features = self.extract_tfidf_features(texts)
        lexicon_features = self.extract_lexicon_features_with_negation(texts, processed_texts)

        if hasattr(tfidf_features, 'toarray'):
            tfidf_dense = tfidf_features.toarray()
        else:
            tfidf_dense = tfidf_features

        # 给情感特征更高的权重
        weighted_lexicon_features = lexicon_features * 3.0

        combined_features = np.hstack([tfidf_dense, weighted_lexicon_features])
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