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
        """预测情感概率"""
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)

            # 处理不同模型返回的概率格式
            if isinstance(probabilities, list):
                # MultiOutputClassifier返回列表
                emotion_probs = np.array([prob[:, 1] if prob.shape[1] > 1 else prob[:, 0] for prob in probabilities]).T
            else:
                # OneVsRestClassifier返回数组
                if probabilities.ndim == 3:
                    # 某些模型返回3D数组
                    emotion_probs = probabilities[:, :, 1] if probabilities.shape[2] > 1 else probabilities[:, :, 0]
                else:
                    # 2D数组
                    emotion_probs = probabilities

            return emotion_probs
        else:
            # 对于不支持概率预测的模型，返回二进制预测
            predictions = self.predict(X)
            return predictions.astype(float)

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