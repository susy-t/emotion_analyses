import re
import jieba
import jieba.posseg as pseg
from zhon.hanzi import punctuation
import os


class ChinesePreprocessor:
    def __init__(self, stopwords_dir='stopwords'):
        self.stopwords = self._load_stopwords(stopwords_dir)
        # 初始化jieba分词器
        jieba.initialize()

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
        if use_pos:
            # 使用词性标注的分词
            words = pseg.cut(text)
            # 只保留名词、动词、形容词、副词等实词
            allowed_pos = {'n', 'v', 'a', 'd', 'ad'}  # 名词、动词、形容词、副词
            words = [word for word, pos in words if pos[0] in allowed_pos]
        else:
            # 普通分词
            words = jieba.cut(text)

        # 移除停用词和单字词
        words = [word for word in words if word not in self.stopwords and len(word) > 1]
        return list(words)

    def preprocess(self, text, use_pos=False):
        """完整的预处理流程"""
        cleaned_text = self.clean_text(text)
        words = self.segment_text(cleaned_text, use_pos)
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