import joblib
import numpy as np
import pandas as pd
from chinese_preprocessor import ChinesePreprocessor
from feature_extractor import ChineseFeatureExtractor
from model import ChineseEmotionClassifier


class ChineseEmotionPredictor:
    def __init__(self, model_path=None):
        if model_path:
            self.load_model(model_path)
        else:
            self.preprocessor = ChinesePreprocessor()
            self.feature_extractor = ChineseFeatureExtractor()
            self.classifier = ChineseEmotionClassifier()

    def load_model(self, model_path):
        """加载训练好的模型"""
        try:
            model_data = joblib.load(model_path)
            self.classifier = model_data['classifier']
            self.feature_extractor = model_data['feature_extractor']
            self.preprocessor = model_data['preprocessor']
            print("中文情感分析模型加载成功!")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("将使用默认模型...")
            self.preprocessor = ChinesePreprocessor()
            self.feature_extractor = ChineseFeatureExtractor()
            self.classifier = ChineseEmotionClassifier()

    def predict_emotion(self, text):
        """预测文本情感"""
        # 预处理文本
        processed_text = self.preprocessor.preprocess(text)

        # 提取特征
        features = self.feature_extractor.extract_features([processed_text])

        # 预测情感
        prediction = self.classifier.predict(features)[0]
        probabilities = self.classifier.predict_proba(features)

        # 处理概率输出 - 确保是一维数组
        if len(probabilities.shape) > 1:
            emotion_probs = probabilities[0]
        else:
            emotion_probs = probabilities

        emotions = ['愤怒', '恐惧', '悲伤', '愉悦', '恶心', '惊喜']

        # 构建结果
        result = {
            'text': text,
            'processed_text': processed_text,
            'emotions': {}
        }

        for i, emotion in enumerate(emotions):
            result['emotions'][emotion] = {
                'predicted': bool(prediction[i]),
                'probability': float(emotion_probs[i])
            }

        return result

    def analyze_multiple_texts(self, texts):
        """分析多个文本的情感"""
        results = []
        for text in texts:
            results.append(self.predict_emotion(text))
        return results

    def print_result(self, result):
        """格式化打印结果"""
        print(f"原文: {result['text']}")
        print(f"处理后: {result['processed_text']}")
        print("\n情感分析结果:")
        print("-" * 50)

        # 按置信度排序
        sorted_emotions = sorted(
            result['emotions'].items(),
            key=lambda x: x[1]['probability'],
            reverse=True
        )

        for emotion, info in sorted_emotions:
            status = "✓" if info['predicted'] else "✗"
            prob_percent = info['probability'] * 100
            print(f"{emotion:5} [{status}] 置信度: {prob_percent:5.1f}%")

        # 显示主要情感
        main_emotions = [emotion for emotion, info in result['emotions'].items()
                         if info['predicted']]
        if main_emotions:
            print(f"\n主要情感: {', '.join(main_emotions)}")
        else:
            print("\n无明显情感倾向")

        # 显示最高置信度的情感
        if sorted_emotions:
            top_emotion, top_info = sorted_emotions[0]
            top_prob = top_info['probability'] * 100
            print(f"最强烈情感: {top_emotion} ({top_prob:.1f}%)")


if __name__ == "__main__":
    # 使用示例
    predictor = ChineseEmotionPredictor('chinese_emotion_model.pkl')

    # 如果没有训练好的模型，先训练一个
    if predictor.classifier.model is None:
        from train import ChineseEmotionTrainer

        print("未找到训练好的模型，开始训练...")
        trainer = ChineseEmotionTrainer()
        classifier = trainer.train()
        predictor.classifier = classifier
        predictor.feature_extractor = trainer.feature_extractor
        predictor.preprocessor = trainer.preprocessor

    # 中文测试文本
    test_texts = [
        "我对此感到非常愤怒，简直无法忍受！",
        "这个情况让我感到恐惧和不安",
        "今天心情特别好，感到非常愉悦！",
        "这种行为真让人恶心，看不下去",
        "完全出乎意料，真是个惊喜！",
        "听到这个消息我伤心极了",
        "这个电影太恐怖了，吓死我了"
    ]

    print("中文情感分析演示:")
    print("=" * 60)

    for text in test_texts:
        result = predictor.predict_emotion(text)
        predictor.print_result(result)
        print("=" * 60)