import joblib
import numpy as np
import pandas as pd
from chinese_preprocessor import ChinesePreprocessor
from feature_extractor import ChineseFeatureExtractor
from model import ChineseEmotionClassifier


class ChineseEmotionPredictor:
    def __init__(self, model_path=None, use_negation=True):
        self.use_negation = use_negation
        self.preprocessor = ChinesePreprocessor()
        self.feature_extractor = ChineseFeatureExtractor(use_negation=use_negation)

        if model_path:
            self.load_model(model_path)
        else:
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
            self.feature_extractor = ChineseFeatureExtractor(use_negation=self.use_negation)
            self.classifier = ChineseEmotionClassifier()

    def predict_emotion(self, text):
        """预测文本情感"""
        try:
            if not isinstance(text, str) or not text.strip():
                return self._create_empty_result("输入文本为空")

            if self.use_negation:
                processed_text = self.preprocessor.preprocess(text, use_negation=True)
            else:
                processed_text = self.preprocessor.preprocess(text)

            features = self.feature_extractor.extract_features(
                [text],
                [processed_text] if self.use_negation else None
            )

            if features is None or len(features) == 0:
                return self._create_empty_result("特征提取失败")

            prediction = self.classifier.predict(features)[0]
            probabilities = self.classifier.predict_proba(features)

            # 处理概率格式
            if isinstance(probabilities, list):
                emotion_probs = []
                for prob in probabilities:
                    if len(prob) > 0 and len(prob[0]) > 1:
                        emotion_probs.append(prob[0][1])
                    elif len(prob) > 0:
                        emotion_probs.append(prob[0][0])
                    else:
                        emotion_probs.append(0.0)
            else:
                if probabilities.shape[1] > 1:
                    emotion_probs = probabilities[0][:, 1] if probabilities[0].ndim > 1 else probabilities[0]
                else:
                    emotion_probs = probabilities[0]

            emotions = ['愤怒', '恐惧', '悲伤', '愉悦', '恶心', '惊喜']

            result = {
                'text': text,
                'processed_text': processed_text,
                'emotions': {}
            }

            for i, emotion in enumerate(emotions):
                if i < len(emotion_probs):
                    result['emotions'][emotion] = {
                        'predicted': bool(prediction[i]) if i < len(prediction) else False,
                        'probability': float(emotion_probs[i]) if i < len(emotion_probs) else 0.0
                    }
                else:
                    result['emotions'][emotion] = {
                        'predicted': False,
                        'probability': 0.0
                    }

            return result
        except Exception as e:
            print(f"分析过程中出现错误: {e}")
            return self._create_empty_result(f"分析错误: {str(e)}")

    def _create_empty_result(self, error_msg):
        """创建空结果"""
        emotions = ['愤怒', '恐惧', '悲伤', '愉悦', '恶心', '惊喜']
        result = {
            'text': '',
            'processed_text': '',
            'error': error_msg,
            'emotions': {}
        }
        for emotion in emotions:
            result['emotions'][emotion] = {
                'predicted': False,
                'probability': 0.0
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
        if 'error' in result:
            print(f"错误: {result['error']}")
            return

        print(f"原文: {result['text']}")
        print(f"处理后: {result['processed_text']}")
        print("\n情感分析结果:")
        print("-" * 50)

        sorted_emotions = sorted(
            result['emotions'].items(),
            key=lambda x: x[1]['probability'],
            reverse=True
        )

        for emotion, info in sorted_emotions:
            status = "✓" if info['predicted'] else "✗"
            prob_percent = info['probability'] * 100
            print(f"{emotion:5} [{status}] 置信度: {prob_percent:5.1f}%")

        main_emotions = [emotion for emotion, info in result['emotions'].items()
                         if info['predicted']]
        if main_emotions:
            print(f"\n主要情感: {', '.join(main_emotions)}")
        else:
            print("\n无明显情感倾向")

        if sorted_emotions:
            top_emotion, top_info = sorted_emotions[0]
            top_prob = top_info['probability'] * 100
            print(f"最强烈情感: {top_emotion} ({top_prob:.1f}%)")


if __name__ == "__main__":
    predictor = ChineseEmotionPredictor('chinese_emotion_model.pkl', use_negation=True)

    if predictor.classifier.model is None:
        from train import ChineseEmotionTrainer

        print("未找到训练好的模型，开始训练...")
        trainer = ChineseEmotionTrainer()
        classifier = trainer.train()
        predictor.classifier = classifier
        predictor.feature_extractor = trainer.feature_extractor
        predictor.preprocessor = trainer.preprocessor

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