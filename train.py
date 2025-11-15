import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from chinese_preprocessor import ChinesePreprocessor
from feature_extractor import ChineseFeatureExtractor
from model import ChineseEmotionClassifier


class ChineseEmotionTrainer:
    def __init__(self, feature_method='combined', model_type='random_forest'):
        self.preprocessor = ChinesePreprocessor()
        self.feature_extractor = ChineseFeatureExtractor(method=feature_method)
        self.classifier = ChineseEmotionClassifier(model_type=model_type)

    def load_sample_data(self):
        """
        加载更丰富的中文示例数据
        """
        # 增强版中文示例数据 - 每个情感类别都有更多样本
        data = {
            'text': [
                # 愤怒类 - 12个样本
                "我对此情况感到非常愤怒！简直气死我了！",
                "我恨透了这个事情，怒火中烧",
                "气得我直跺脚，简直无法忍受",
                "这让我火冒三丈，太生气了",
                "真是让人气愤不已",
                "我对此极为不满，愤怒至极",
                "这简直是在挑战我的底线",
                "气得我浑身发抖",
                "这种不公平让我愤怒",
                "我怒火冲天，无法平静",
                "这行为让我义愤填膺",
                "气得我咬牙切齿",

                # 恐惧类 - 12个样本
                "这太可怕了，我真的被吓到了，心里很恐惧",
                "这让我从心底感到害怕，恐惧万分",
                "吓得我浑身发抖，太恐怖了",
                "这个恐怖片让我毛骨悚然",
                "我感到极度恐惧，不敢独自在家",
                "这种危险的情况让我害怕",
                "听到那个消息我胆战心惊",
                "吓得我魂飞魄散",
                "这个鬼故事太吓人了",
                "黑暗中我感到非常恐惧",
                "这种未知让我感到害怕",
                "惊悚的场景让我恐惧不已",

                # 悲伤类 - 12个样本
                "今天我感到非常伤心和沮丧，心情低落",
                "我的心都碎了，一直在哭泣",
                "悲伤的情绪笼罩着我，很难过",
                "听到这个消息我悲痛欲绝",
                "我感到十分难过，眼泪止不住",
                "这种失落感让我很伤心",
                "离别总是让人伤感",
                "看到这种情景我很难过",
                "失去的痛苦让我悲伤",
                "忧郁的情绪挥之不去",
                "这件事让我心如刀割",
                "悲伤让我无法振作",

                # 愉悦类 - 12个样本
                "多么美好和愉快的经历啊！太开心了！",
                "我太高兴了，兴奋不已！",
                "喜悦的心情难以言表，太棒了",
                "今天真是快乐的一天",
                "我感到无比幸福和满足",
                "这个好消息让我欣喜若狂",
                "心情特别好，充满喜悦",
                "开心的不得了",
                "这让我心情愉悦，笑容满面",
                "快乐的时光总是短暂",
                "内心充满欢喜和满足",
                "幸福的感觉真美好",

                # 恶心类 - 12个样本
                "这真令人作呕，太恶心了",
                "这让我恶心想吐，太反感了",
                "看到这个就觉得恶心，真受不了",
                "这种行为令人作呕",
                "这种味道让我反胃",
                "看到这种场面我觉得很恶心",
                "这种卑劣的行为真让人反感",
                "恶心的我吃不下饭",
                "这种肮脏的东西真恶心",
                "令人厌恶到极点",
                "看到就让我反胃",
                "这种气味真让人作呕",

                # 惊喜类 - 12个样本
                "哇，这真是个惊喜！完全出乎意料！",
                "真不敢相信，太让人震惊了！",
                "意外的惊喜让我措手不及",
                "这完全出乎我的意料",
                "真是个意外的惊喜",
                "没想到会这样，太惊喜了",
                "这结果让我大吃一惊",
                "突如其来的好消息",
                "这意外的发现让我惊喜",
                "完全没想到会是这样的结果",
                "惊喜来得太突然了",
                "这真是个美妙的意外"
            ],
            # 对应的情感标签 - 每个情感12个1，其余为0
            '愤怒': [1] * 12 + [0] * 60,
            '恐惧': [0] * 12 + [1] * 12 + [0] * 48,
            '悲伤': [0] * 24 + [1] * 12 + [0] * 36,
            '愉悦': [0] * 36 + [1] * 12 + [0] * 24,
            '恶心': [0] * 48 + [1] * 12 + [0] * 12,
            '惊喜': [0] * 60 + [1] * 12
        }
        return pd.DataFrame(data)

    def prepare_data(self, df):
        """准备训练数据"""
        print("正在预处理文本数据...")
        # 预处理文本
        df['processed_text'] = df['text'].apply(self.preprocessor.preprocess)

        print("正在提取特征...")
        # 提取特征
        X = self.feature_extractor.extract_features(df['processed_text'].tolist())

        # 准备标签
        emotion_columns = ['愤怒', '恐惧', '悲伤', '愉悦', '恶心', '惊喜']
        y = df[emotion_columns].values

        return X, y, emotion_columns

    def train(self, test_size=0.2, save_model=True):
        """改进的训练方法"""
        # 加载数据
        print("正在加载数据...")
        df = self.load_sample_data()

        # 准备特征和标签
        X, y, emotion_columns = self.prepare_data(df)

        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # 训练模型
        print("开始训练模型...")
        self.classifier.fit(X_train, y_train)

        # 评估模型
        print("评估模型性能...")
        accuracy = self.classifier.evaluate(X_test, y_test)

        # 如果准确率太低，尝试使用更复杂的模型
        if accuracy < 0.7:
            print("模型性能不佳，尝试使用SVM模型...")
            self.classifier = ChineseEmotionClassifier(model_type='svm')
            self.classifier.fit(X_train, y_train)
            accuracy = self.classifier.evaluate(X_test, y_test)

        if save_model:
            self.save_model('chinese_emotion_model.pkl')

        return self.classifier

    def save_model(self, filepath):
        """保存训练好的模型"""
        model_data = {
            'classifier': self.classifier,
            'feature_extractor': self.feature_extractor,
            'preprocessor': self.preprocessor
        }
        joblib.dump(model_data, filepath)
        print(f"模型已保存到: {filepath}")


if __name__ == "__main__":
    # 训练模型示例
    print("开始训练中文情感分析模型...")
    trainer = ChineseEmotionTrainer(feature_method='combined', model_type='random_forest')
    trainer.train()
    print("训练完成!")