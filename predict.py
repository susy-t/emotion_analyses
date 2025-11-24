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
            # 确保默认分类器有基本模型
            try:
                self.classifier.build_model()
            except:
                pass

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
            # 确保默认分类器有基本模型
            try:
                self.classifier.build_model()
            except:
                pass

    def hybrid_predict(self, text):
        """混合预测：结合模型预测和规则分析"""
        # 首先尝试模型预测
        model_result = self.predict_emotion(text)

        # 检查是否有明确的情感预测
        has_clear_prediction = any(info['predicted'] for info in model_result['emotions'].values())

        # 如果模型没有明确预测或置信度低，尝试规则分析
        if not has_clear_prediction or model_result.get('confidence') in ['low', 'unknown']:

            rule_results = self.feature_extractor.rule_based_analysis([text])
            rule_result = rule_results[0]

            if rule_result['emotion'] is not None:
                # 使用规则分析结果
                emotions = ['愤怒', '恐惧', '悲伤', '愉悦', '恶心', '惊喜']
                emotion_idx = emotions.index(rule_result['emotion'])

                # 更新结果
                for i, emotion in enumerate(emotions):
                    model_result['emotions'][emotion]['predicted'] = (i == emotion_idx)
                    model_result['emotions'][emotion]['probability'] = rule_result['all_probabilities'][i]

                model_result['confidence'] = 'rule_based'
                model_result['primary_emotion'] = rule_result['emotion']
            else:
                # 规则分析也无法判断
                model_result['confidence'] = 'uncertain'
                model_result['primary_emotion'] = None
                # 将所有情感标记为未预测
                for emotion in model_result['emotions']:
                    model_result['emotions'][emotion]['predicted'] = False

        return model_result

    def predict_emotion(self, text):
        """预测文本情感 - 改进版本，处理不确定性"""
        try:
            if not isinstance(text, str) or not text.strip():
                return self._create_empty_result("输入文本为空", uncertain=True)

            if self.use_negation:
                processed_text = self.preprocessor.preprocess(text, use_negation=True)
            else:
                processed_text = self.preprocessor.preprocess(text)

            features = self.feature_extractor.extract_features(
                [text],
                [processed_text] if self.use_negation else None
            )

            if features is None or len(features) == 0:
                return self._create_empty_result("特征提取失败", uncertain=True)

            # 检查分类器是否有模型
            if self.classifier.model is None:
                return self._create_empty_result("模型未训练", uncertain=True)

            # 使用改进的预测方法
            if hasattr(self.classifier, 'predict_emotion_with_confidence'):
                try:
                    results = self.classifier.predict_emotion_with_confidence(features)
                    result_data = results[0]  # 取第一个结果
                except Exception as e:
                    print(f"使用置信度预测失败: {e}，回退到普通预测")
                    # 回退到普通预测
                    prediction = self.classifier.predict(features)[0]
                    probabilities = self.classifier.predict_proba(features)[0]
                    result_data = {
                        'predictions': prediction,
                        'probabilities': probabilities,
                        'confidence': 'unknown',
                        'primary_emotion': None
                    }
            else:
                # 回退到原来的方法
                prediction = self.classifier.predict(features)[0]
                probabilities = self.classifier.predict_proba(features)[0]
                result_data = {
                    'predictions': prediction,
                    'probabilities': probabilities,
                    'confidence': 'unknown',
                    'primary_emotion': None
                }

            # 处理概率格式
            emotion_probs = result_data['probabilities']
            if len(emotion_probs) != 6:
                fixed_probs = np.zeros(6)
                min_len = min(len(emotion_probs), 6)
                fixed_probs[:min_len] = emotion_probs[:min_len]
                emotion_probs = fixed_probs

            emotions = ['愤怒', '恐惧', '悲伤', '愉悦', '恶心', '惊喜']

            result = {
                'text': text,
                'processed_text': processed_text,
                'emotions': {},
                'confidence': result_data['confidence'],
                'primary_emotion': result_data['primary_emotion']
            }

            # 检查是否有明确的情感预测
            has_prediction = False
            max_prob = 0
            max_emotion = None

            for i, emotion in enumerate(emotions):
                predicted = (result_data['predictions'][i] == 1) if i < len(result_data['predictions']) else False

                # 对于中等置信度，如果该情感概率最高，也标记为预测
                if result_data['confidence'] == 'medium' and i == np.argmax(emotion_probs):
                    predicted = True

                result['emotions'][emotion] = {
                    'predicted': predicted,
                    'probability': float(emotion_probs[i]) if i < len(emotion_probs) else 0.0
                }

                if predicted:
                    has_prediction = True
                    if emotion_probs[i] > max_prob:
                        max_prob = emotion_probs[i]
                        max_emotion = emotion

            # 如果没有明确预测，标记为不确定
            if not has_prediction:
                result['confidence'] = 'uncertain'
                result['primary_emotion'] = None

            return result

        except Exception as e:
            print(f"分析过程中出现错误: {e}")
            return self._create_empty_result(f"分析错误: {str(e)}", uncertain=True)

    def _create_empty_result(self, error_msg, uncertain=False):
        """创建空结果"""
        emotions = ['愤怒', '恐惧', '悲伤', '愉悦', '恶心', '惊喜']
        result = {
            'text': '',
            'processed_text': '',
            'error': error_msg,
            'emotions': {},
            'confidence': 'uncertain' if uncertain else 'unknown',
            'primary_emotion': None
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
            results.append(self.hybrid_predict(text))
        return results

    def print_result(self, result):
        """格式化打印结果 - 明确区分无法判断的情况"""
        if 'error' in result:
            print(f"错误: {result['error']}")
            return

        print(f"原文: {result['text']}")
        print(f"处理后: {result['processed_text']}")

        # 显示置信度
        confidence = result.get('confidence', 'unknown')
        confidence_text = {
            'high': '高置信度',
            'medium': '中等置信度',
            'low': '低置信度',
            'rule_based': '规则分析',
            'uncertain': '无法判断',
            'unknown': '未知置信度'
        }.get(confidence, '未知置信度')

        print(f"置信度: {confidence_text}")
        print("\n情感分析结果:")
        print("-" * 50)

        # 如果无法判断，直接显示并返回
        if confidence == 'uncertain':
            print("无法判断文本的情感倾向")
            print("\n各情感概率分布（仅供参考）:")
            sorted_emotions = sorted(
                result['emotions'].items(),
                key=lambda x: x[1]['probability'],
                reverse=True
            )

            for emotion, info in sorted_emotions:
                prob_percent = info['probability'] * 100
                print(f"{emotion:5}: {prob_percent:5.1f}%")
            return

        sorted_emotions = sorted(
            result['emotions'].items(),
            key=lambda x: x[1]['probability'],
            reverse=True
        )

        for emotion, info in sorted_emotions:
            status = "✓" if info['predicted'] else "✗"
            prob_percent = info['probability'] * 100

            # 根据置信度使用不同的标记
            if confidence == 'low':
                status = "?"
                print(f"{emotion:5} [{status}] 置信度: {prob_percent:5.1f}% (不确定)")
            elif confidence == 'medium':
                status = "~"
                print(f"{emotion:5} [{status}] 置信度: {prob_percent:5.1f}% (可能)")
            elif confidence == 'rule_based':
                status = "R"
                print(f"{emotion:5} [{status}] 置信度: {prob_percent:5.1f}% (规则)")
            else:
                print(f"{emotion:5} [{status}] 置信度: {prob_percent:5.1f}%")

        # 处理主要情感显示
        if confidence == 'low':
            print(f"\n情感倾向不明显，模型对此文本不确定")
        elif confidence == 'medium':
            main_emotions = [emotion for emotion, info in result['emotions'].items()
                             if info['predicted']]
            if main_emotions:
                print(f"\n可能的情感: {', '.join(main_emotions)}")
            else:
                print("\n无明显情感倾向")
        elif confidence == 'rule_based':
            main_emotions = [emotion for emotion, info in result['emotions'].items()
                             if info['predicted']]
            if main_emotions:
                print(f"\n基于规则分析的情感: {', '.join(main_emotions)}")
            else:
                print("\n无明显情感倾向")
        else:
            main_emotions = [emotion for emotion, info in result['emotions'].items()
                             if info['predicted']]
            if main_emotions:
                print(f"\n主要情感: {', '.join(main_emotions)}")
            else:
                print("\n无明显情感倾向")

        if sorted_emotions:
            top_emotion, top_info = sorted_emotions[0]
            top_prob = top_info['probability'] * 100
            if confidence in ['high', 'medium', 'rule_based'] and top_info['predicted']:
                if confidence == 'high':
                    confidence_level = "高度可能"
                elif confidence == 'medium':
                    confidence_level = "可能"
                else:
                    confidence_level = "基于规则"
                print(f"最强烈情感: {top_emotion} ({top_prob:.1f}%) - {confidence_level}")


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
        "我很愤怒",
        "我很悲伤",
        "有点生气",
        "气死我了",
        "这个电影太恐怖了",
        "今天心情特别好",
        "这种行为真让人恶心",
        "完全出乎意料",
        "听到这个消息我伤心极了",
        "这个游戏真好玩",
        "老板今天又骂我了",
        "收到礼物太开心了",
        "今天天气不错",  # 中性文本，应该无法判断
        "我吃饭了"  # 中性文本，应该无法判断
    ]

    print("中文情感分析演示:")
    print("=" * 60)

    for text in test_texts:
        result = predictor.hybrid_predict(text)
        predictor.print_result(result)
        print("=" * 60)