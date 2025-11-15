import re
import jieba
import jieba.posseg as pseg
from zhon.hanzi import punctuation
import os


class ChinesePreprocessor:
    def __init__(self, stopwords_dir='stopwords'):
        self.stopwords = self._load_stopwords(stopwords_dir)
        # 初始化情感白名单
        self.emotion_whitelist = self._create_emotion_whitelist()

        # 初始化jieba分词器
        # 添加否定词列表
        self.negation_words = {
            '不', '没', '无', '非', '未', '勿', '莫', '没有', '别', '未',
            '不要', '不用', '不必', '未能', '无法', '不会', '不可', '不能',
            '绝不', '从不', '毫无', '毫无', '毫无', '毫无意义'
        }

        jieba.initialize()

    def segment_text_with_negation(self, text, use_pos=False):
        """分词并标记否定词"""
        if use_pos:
            words = pseg.cut(text)
            allowed_pos = {'n', 'v', 'a', 'd', 'ad', 'i', 'l'}
            words = [(word, pos) for word, pos in words if pos[0] in allowed_pos]
        else:
            words = [(word, 'x') for word in jieba.cut(text)]  # 'x'表示未知词性

        # 过滤停用词，但保留否定词和情感词
        filtered_words = []
        for word, pos in words:
            # 保留否定词
            if word in self.negation_words:
                filtered_words.append(('NOT', 'negation'))  # 标记为否定
            # 保留情感白名单中的词
            elif word in self.emotion_whitelist:
                filtered_words.append((word, pos))
            # 其他词：不在停用词列表中且长度大于0
            elif word not in self.stopwords and len(word.strip()) > 0:
                filtered_words.append((word, pos))

        return filtered_words

    def _create_emotion_whitelist(self):
        """创建情感词白名单"""
        whitelist = set()

        # 愤怒相关
        anger_words = ['怒', '愤', '气', '恼', '恨', '火', '暴躁', '生气', '愤怒', '气愤', '怒火']
        # 恐惧相关
        fear_words = ['惧', '怕', '畏', '吓', '惊', '恐', '害怕', '恐怖', '恐惧', '惊吓']
        # 悲伤相关
        sadness_words = ['悲', '伤', '哀', '愁', '怨', '哭', '泪', '伤心', '悲伤', '难过']
        # 愉悦相关
        joy_words = ['喜', '乐', '欢', '笑', '欣', '悦', '开心', '高兴', '快乐', '喜欢']
        # 恶心相关
        disgust_words = ['恶', '厌', '吐', '呕', '嫌', '憎', '讨厌', '厌恶', '反感', '恶心']
        # 惊喜相关
        surprise_words = ['惊', '讶', '奇', '异', '诧', '愕', '惊喜', '惊讶', '惊奇']

        # 合并所有情感词
        all_emotion_words = anger_words + fear_words + sadness_words + joy_words + disgust_words + surprise_words
        whitelist.update(all_emotion_words)

        return whitelist

    def preprocess_with_negation(self, text, use_pos=False):
        """带否定词处理的预处理"""
        cleaned_text = self.clean_text(text)
        words_with_negation = self.segment_text_with_negation(cleaned_text, use_pos)

        # 如果过滤后为空，返回原始文本的前几个字符
        if not words_with_negation:
            fallback = cleaned_text[:3] if len(cleaned_text) >= 3 else cleaned_text
            return fallback if fallback else text

        # 将处理后的词连接成字符串，否定词标记为NOT
        processed_tokens = []
        for word, pos in words_with_negation:
            if word == 'NOT':
                processed_tokens.append('NOT')
            else:
                processed_tokens.append(word)

        return ' '.join(processed_tokens)

    def _load_stopwords(self, stopwords_dir):
        """加载中文停用词表"""
        stopwords = set()

        # 常见的中文停用词文件
        stopword_files = [
            'cn_stopwords.txt',
            'hit_stopwords.txt'
        ]

        for filename in stopword_files:
            filepath = os.path.join(stopwords_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        words = f.read().splitlines()
                        stopwords.update(words)
                    print(f"已加载停用词文件: {filename}")
                except Exception as e:
                    print(f"加载停用词文件 {filename} 时出错: {e}")

        # 如果没有找到停用词文件，使用内置的基础停用词
        if not stopwords:
            stopwords = self._get_basic_stopwords()
            print("使用内置基础停用词")

        return stopwords

    def _get_basic_stopwords(self):
        """获取基础中文停用词"""
        basic_stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到',
            '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '他', '她', '它', '我们',
            '你们', '他们', '这个', '那个', '这些', '那些', '这样', '那样', '这里', '那里', '这时', '那时', '什么',
            '为什么', '怎么', '哪里', '谁', '几', '多少', '很', '非常', '特别', '更', '最', '太', '极', '挺', '相当',
            '有点', '一些', '一点', '啊', '呀', '呢', '吧', '吗', '啦', '唉', '哦', '嗯', '哼', '哇', '喔', '喂', '嘛',
            '喽', '咚', '咦', '哟', '呃'
        }
        return basic_stopwords

    def clean_text(self, text):
        """清理中文文本"""
        if not isinstance(text, str):
            return ""

        # 移除URL
        text = re.sub(r'http\S+', '', text)
        # 移除HTML标签
        text = re.sub(r'<.*?>', '', text)
        # 移除数字
        text = re.sub(r'\d+', '', text)
        # 移除英文
        text = re.sub(r'[a-zA-Z]', '', text)
        # 移除多余空格和换行符
        text = re.sub(r'\s+', ' ', text)
        # 移除标点符号（保留中文常用标点用于情感分析）
        text = re.sub(r'[{}]'.format(punctuation), ' ', text)
        text = text.strip()
        return text

    def segment_text(self, text, use_pos=False):
        """中文分词"""
        if not isinstance(text, str) or not text.strip():
            return []

        if use_pos:
            words = pseg.cut(text)
            allowed_pos = {'n', 'v', 'a', 'd', 'ad', 'i', 'l'}
            words = [word for word, pos in words if pos[0] in allowed_pos]
        else:
            words = jieba.cut(text)

        # 不过滤情感白名单中的词，其他词按正常规则过滤
        filtered_words = []
        for word in words:
            if word in self.emotion_whitelist:
                filtered_words.append(word)
            elif word not in self.stopwords and len(word.strip()) > 0:
                filtered_words.append(word)

        return filtered_words

    def preprocess(self, text, use_pos=False, use_negation=True):
        """
        统一的预处理流程
        Args:
            text: 输入文本
            use_pos: 是否使用词性标注
            use_negation: 是否使用否定词处理（默认开启）
        """
        if not isinstance(text, str):
            return ""

        if use_negation:
            return self.preprocess_with_negation(text, use_pos)

        # 原有的预处理逻辑
        cleaned_text = self.clean_text(text)
        words = self.segment_text(cleaned_text, use_pos)

        if not words:
            fallback = cleaned_text[:3] if len(cleaned_text) >= 3 else cleaned_text
            return fallback if fallback else text

        return ' '.join(words)

    def add_stopwords(self, words):
        """添加自定义停用词"""
        if isinstance(words, str):
            self.stopwords.add(words)
        else:
            self.stopwords.update(words)

    def remove_stopwords(self, words):
        """移除停用词"""
        if isinstance(words, str):
            self.stopwords.discard(words)
        else:
            self.stopwords.difference_update(words)