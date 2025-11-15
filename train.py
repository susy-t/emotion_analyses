import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from chinese_preprocessor import ChinesePreprocessor
from feature_extractor import ChineseFeatureExtractor
from model import ChineseEmotionClassifier


class ChineseEmotionTrainer:
    def __init__(self, feature_method='combined', model_type='random_forest', use_negation=True):
        self.preprocessor = ChinesePreprocessor()
        self.feature_extractor = ChineseFeatureExtractor(
            method=feature_method,
            use_negation=use_negation
        )
        self.classifier = ChineseEmotionClassifier(model_type=model_type)
        self.use_negation = use_negation

    def load_sample_data(self):
        """
        加载更丰富的中文示例数据（使用前面代码的数据集）
        """
        # 定义每个情感的文本（使用您原来的丰富数据集）
        anger_texts = [
            "他气得一拳砸在桌子上，震得茶杯嗡嗡作响。",
            "她双眉倒竖，眼中几乎要喷出火来。",
            "你凭什么这样对我！他怒吼道。",
            "他紧握的双拳青筋暴起，浑身都在发抖。",
            "我受够了你的谎言！她咬牙切齿地说。",
            "他愤怒地撕碎了合同，纸屑漫天飞舞。",
            "你们这是在欺负人！他涨红着脸喊道。",
            "她气得浑身发抖，话都说不连贯。",
            "他怒目圆睁，像一头被激怒的狮子。",
            "这简直是在侮辱我的智商！他愤然离席。",
        ]

        fear_texts = [
            "她惊恐地睁大眼睛，一步步往后退。",
            "黑暗中传来奇怪的声响，他吓得屏住了呼吸。",
            "我...我好害怕...她颤抖着说。",
            "他浑身冷汗直冒，心跳快得要蹦出胸口。",
            "别过来！她尖叫着缩到墙角。",
            "他两腿发软，几乎要跪倒在地。",
            "那是什么声音？她紧张地抓住同伴的胳膊。",
            "他吓得魂飞魄散，头也不回地狂奔。",
            "求求你别伤害我...她带着哭腔哀求。",
            "他躲在衣柜里，大气都不敢出。",
        ]

        sadness_texts = [
            "她独自坐在窗前，泪水无声滑落。",
            "他抱着相框，久久不语。",
            "为什么偏偏是我...她哽咽着说。",
            "他望着远方，眼中盛满哀愁。",
            "雨滴敲打着窗户，就像她破碎的心。",
            "他轻轻抚摸着那张泛黄的照片。",
            "一切都结束了...她低声啜泣。",
            "他把自己关在房间里，一整天都不出门。",
            "她红着眼眶，强忍着不让眼泪掉下来。",
            "那首老歌让他想起了逝去的亲人。",
        ]

        joy_texts = [
            "她哼着歌，脚步轻快地走在路上。",
            "太棒了！他兴奋地跳了起来。",
            "阳光真好，她眯着眼享受这美好时刻。",
            "他忍不住嘴角上扬，心里甜滋滋的。",
            "今天真是个好日子！她开心地说。",
            "他收到礼物时，脸上绽放出灿烂的笑容。",
            "她和小狗在草地上嬉戏，笑声不断。",
            "这次成功让他感到前所未有的满足。",
            "他惬意地靠在躺椅上，享受着午后时光。",
            "她看到老朋友时，惊喜地叫出声来。",
        ]

        disgust_texts = [
            "他闻到那股味道，立即捂住了鼻子。",
            "看着蠕动的蛆虫，她感到一阵反胃。",
            "这食物都发霉了！他嫌弃地推开盘子。",
            "她看到蟑螂爬过，恶心得浑身起鸡皮疙瘩。",
            "别让我再看到这种东西！他厌恶地转过头。",
            "那股酸臭味让她差点吐出来。",
            "他小心翼翼地用纸巾捏起那只死老鼠。",
            "太恶心了！她看着污秽的厕所直皱眉。",
            "他喝到变质的牛奶，连忙吐了出来。",
            "她不敢碰那件沾满污渍的衣服。",
        ]

        surprise_texts = [
            "打开门的那一刻，她惊喜地叫出声来。",
            "这是给我的吗？他不敢相信自己的眼睛。",
            "你们怎么都来了！她感动得热泪盈眶。",
            "他拆开礼物，脸上写满了惊喜。",
            "这个结果完全出乎她的意料！",
            "他以为自己看错了，揉了揉眼睛。",
            "生日快乐！众人齐声喊道，她愣住了。",
            "他没想到会在这里遇见老朋友。",
            "这份礼物太贴心了！她激动地说。",
            "他收到录取通知时，欣喜若狂。",
        ]

        # 合并所有文本
        all_texts = anger_texts + fear_texts + sadness_texts + joy_texts + disgust_texts + surprise_texts

        # 创建标签数组
        anger_labels = [1] * len(anger_texts) + [0] * (len(all_texts) - len(anger_texts))
        fear_labels = [0] * len(anger_texts) + [1] * len(fear_texts) + [0] * (len(all_texts) - len(anger_texts) - len(fear_texts))
        sadness_labels = [0] * (len(anger_texts) + len(fear_texts)) + [1] * len(sadness_texts) + [0] * (len(all_texts) - len(anger_texts) - len(fear_texts) - len(sadness_texts))
        joy_labels = [0] * (len(anger_texts) + len(fear_texts) + len(sadness_texts)) + [1] * len(joy_texts) + [0] * (len(all_texts) - len(anger_texts) - len(fear_texts) - len(sadness_texts) - len(joy_texts))
        disgust_labels = [0] * (len(anger_texts) + len(fear_texts) + len(sadness_texts) + len(joy_texts)) + [1] * len(disgust_texts) + [0] * (len(all_texts) - len(anger_texts) - len(fear_texts) - len(sadness_texts) - len(joy_texts) - len(disgust_texts))
        surprise_labels = [0] * (len(anger_texts) + len(fear_texts) + len(sadness_texts) + len(joy_texts) + len(disgust_texts)) + [1] * len(surprise_texts)

        # 验证数据长度
        print(f"总文本数量: {len(all_texts)}")
        print(f"愤怒标签数量: {len(anger_labels)}")
        print(f"恐惧标签数量: {len(fear_labels)}")
        print(f"悲伤标签数量: {len(sadness_labels)}")
        print(f"愉悦标签数量: {len(joy_labels)}")
        print(f"恶心标签数量: {len(disgust_labels)}")
        print(f"惊喜标签数量: {len(surprise_labels)}")

        # 创建数据字典
        data = {
            'text': all_texts,
            '愤怒': anger_labels,
            '恐惧': fear_labels,
            '悲伤': sadness_labels,
            '愉悦': joy_labels,
            '恶心': disgust_labels,
            '惊喜': surprise_labels
        }

        return pd.DataFrame(data)

    def prepare_data(self, df):
        """准备训练数据"""
        print("正在预处理文本数据...")

        # 根据use_negation参数决定预处理方式
        if self.use_negation:
            df['processed_text'] = df['text'].apply(
                lambda x: self.preprocessor.preprocess(x, use_negation=True)
            )
        else:
            df['processed_text'] = df['text'].apply(self.preprocessor.preprocess)

        print("正在提取特征...")

        # 根据use_negation参数决定特征提取方式
        try:
            if self.use_negation:
                X = self.feature_extractor.extract_features(
                    df['text'].tolist(),
                    df['processed_text'].tolist()
                )
            else:
                X = self.feature_extractor.extract_features(df['processed_text'].tolist())
        except Exception as e:
            print(f"特征提取失败: {e}")
            # 使用备用方法
            if self.use_negation:
                X = self.feature_extractor.extract_features_normal(df['text'].tolist())
            else:
                X = self.feature_extractor.extract_features_normal(df['processed_text'].tolist())

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

        # 如果准确率太低，尝试使用更简单的模型
        if accuracy < 0.5:
            print("模型性能不佳，尝试使用逻辑回归模型...")
            self.classifier = ChineseEmotionClassifier(model_type='logistic_regression')
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
    trainer = ChineseEmotionTrainer(
        feature_method='combined',
        model_type='random_forest',
        use_negation=True
    )
    trainer.train()
    print("训练完成!")