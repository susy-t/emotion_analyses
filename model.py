import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier


class ChineseEmotionClassifier:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.emotions = ['愤怒', '恐惧', '悲伤', '愉悦', '恶心', '惊喜']
        self.uncertainty_threshold = 0.25  # 调整阈值

    def build_model(self, input_dim=None):
        """构建分类模型 - 修改为支持多标签分类"""
        if self.model_type == 'random_forest':
            self.model = MultiOutputClassifier(RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            ))
        elif self.model_type == 'svm':
            # 使用OneVsRestClassifier包装SVM以支持多标签分类
            self.model = OneVsRestClassifier(SVC(
                probability=True,
                random_state=42,
                class_weight='balanced'
            ))
        elif self.model_type == 'logistic_regression':
            self.model = MultiOutputClassifier(LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            ))
        elif self.model_type == 'mlp':
            self.model = MultiOutputClassifier(MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=42,
                max_iter=1000
            ))
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def fit(self, X, y):
        """训练模型"""
        if self.model is None:
            self.build_model(X.shape[1] if hasattr(X, 'shape') else None)
        self.model.fit(X, y)

    def predict(self, X):
        """预测情感"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """预测情感概率 - 修复概率计算"""
        if hasattr(self.model, 'predict_proba'):
            try:
                probabilities = self.model.predict_proba(X)

                # 处理不同模型返回的概率格式
                if isinstance(probabilities, list):
                    # MultiOutputClassifier返回列表
                    emotion_probs = []
                    for prob in probabilities:
                        if prob.shape[1] > 1:  # 有多个类别
                            emotion_probs.append(prob[:, 1])  # 取正例概率
                        else:
                            emotion_probs.append(prob[:, 0])
                    emotion_probs = np.array(emotion_probs).T
                else:
                    # OneVsRestClassifier或其他模型
                    if probabilities.ndim == 3:
                        emotion_probs = probabilities[:, :, 1]
                    else:
                        emotion_probs = probabilities

                # 确保概率在合理范围内
                emotion_probs = np.clip(emotion_probs, 0.0, 1.0)

                # 如果所有概率都为0，均匀分布
                if np.sum(emotion_probs) == 0:
                    emotion_probs = np.ones(emotion_probs.shape) * (1.0 / emotion_probs.shape[1])

                return emotion_probs

            except Exception as e:
                print(f"概率预测错误: {e}")
                # 返回均匀分布作为后备
                return np.ones((X.shape[0], 6)) * (1.0 / 6)
        else:
            # 对于不支持概率预测的模型，返回二进制预测的软版本
            predictions = self.predict(X)
            return predictions.astype(float)

    def predict_with_uncertainty(self, X):
        """带不确定性检测的预测"""
        probabilities = self.predict_proba(X)
        predictions = self.predict(X)

        # 检测不确定性
        uncertain_indices = []
        for i, prob_row in enumerate(probabilities):
            max_prob = np.max(prob_row)
            second_max_prob = np.sort(prob_row)[-2] if len(prob_row) > 1 else 0

            # 调整不确定性检测的严格程度
            if max_prob < self.uncertainty_threshold or (max_prob - second_max_prob) < 0.1:
                uncertain_indices.append(i)
                # 对于不确定的样本，将预测结果设为全0
                predictions[i] = np.zeros_like(predictions[i])

        return predictions, probabilities, uncertain_indices

    def predict_emotion_with_confidence(self, X):
        """带置信度的情感预测"""
        predictions, probabilities, uncertain_indices = self.predict_with_uncertainty(X)

        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if i in uncertain_indices:
                # 对于不确定的样本，返回均匀分布的低概率
                uniform_prob = np.ones(6) * 0.1  # 每个情感10%的基础概率
                results.append({
                    'predictions': np.zeros(6, dtype=int),
                    'probabilities': uniform_prob,
                    'confidence': 'uncertain',  # 直接标记为不确定
                    'primary_emotion': None
                })
            else:
                primary_idx = np.argmax(prob)
                max_prob = prob[primary_idx]

                # 根据概率值确定置信度
                if max_prob > 0.7:
                    confidence = 'high'
                elif max_prob > 0.5:
                    confidence = 'medium'
                else:
                    confidence = 'low'

                results.append({
                    'predictions': pred,
                    'probabilities': prob,
                    'confidence': confidence,
                    'primary_emotion': self.emotions[primary_idx] if max_prob > 0.5 else None
                })

        return results

    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        y_pred = self.predict(X_test)

        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"模型准确率: {accuracy:.4f}")

        # 详细分类报告
        print("\n详细分类报告:")
        for i, emotion in enumerate(self.emotions):
            print(f"\n{emotion}:")
            emotion_true = y_test[:, i]
            emotion_pred = y_pred[:, i]
            print(classification_report(emotion_true, emotion_pred, zero_division=0))

        return accuracy