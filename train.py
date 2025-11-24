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
            # 扩展愤怒文本
            "老板的无理要求让他火冒三丈，差点把手机摔在地上。",
            "看到有人插队，他立即上前理论，声音因愤怒而颤抖。",
            "被最信任的朋友背叛，她感到前所未有的愤怒和失望。",
            "他握紧拳头，指甲深深陷入掌心，强忍着心中的怒火。",
            "面对不公平的待遇，他决定不再忍气吞声，要讨回公道。"
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
            # 扩展恐惧文本 - 特别是考试焦虑类
            "距离考试开始只有十分钟了，她抱着厚厚的复习资料，在考场外的走廊里来回踱步，脚步慌乱得几乎要踩空。",
            "手心全是冷汗，将纸张浸湿了一片，连带着字迹都变得模糊不清，她用力攥了攥拳头，试图让自己冷静下来。",
            "心脏却像被一只无形的手紧紧攥住，疯狂地跳动着，几乎要冲破胸膛，远处传来监考老师的哨声。",
            "大脑一片空白，之前背得滚瓜烂熟的知识点此刻全变成了杂乱无章的碎片，她咬着下唇直到尝到血腥味。",
            "面试官严肃的表情让她心里发毛，回答问题时的声音都在颤抖，生怕说错一个字就被淘汰。",
            "深夜独自回家，身后传来的脚步声让她心跳加速，不由得加快步伐几乎要跑起来。",
            "医生拿着检查报告神色凝重，她的心一下子提到了嗓子眼，等待宣判般紧张。",
            "站在百米高空的玻璃栈道上，她双腿发软，紧紧抓住栏杆不敢向前迈步。",
            "突然的停电让整个房间陷入黑暗，她蜷缩在角落，听着自己砰砰的心跳声。",
            "面对台下数百名观众，她突然忘词，冷汗瞬间湿透了后背的衣服。"
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
            # 扩展悲伤文本
            "看着空荡荡的房间，想起曾经的热闹，她的眼泪止不住地流下来。",
            "得知宠物去世的消息，他一整天都提不起精神，心里空落落的。",
            "分手后，她删除了所有照片，却删不掉心中的回忆和伤痛。",
            "他默默收拾着父亲的遗物，每一件都承载着深深的思念。",
            "梦想破灭的那一刻，他感到前所未有的失落和绝望。"
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
            # 扩展愉悦文本 - 特别是成功喜悦类
            "当主持人用激昂的声音念出一等奖获得者时，聚光灯瞬间聚焦在她身上，台下雷鸣般的掌声震得她耳膜发麻。",
            "手中的话筒因为过度用力而微微颤抖，眼眶里涌出的热流模糊了视线，她看着父母激动得通红的脸颊。",
            "嘴角不受控制地向上扬起，连带着肩膀都因抑制不住的笑意而轻轻颤抖，过去的努力在这一刻都值得了。",
            "收到心仪大学的录取通知书，她高兴得在房间里转圈，迫不及待地打电话告诉每一个亲人。",
            "项目成功的消息传来，整个团队欢呼雀跃，互相击掌庆祝这来之不易的胜利。",
            "久别重逢的挚友紧紧相拥，笑声和泪水交织在一起，诉说着彼此的思念。",
            "宝宝第一次开口叫妈妈，她的心瞬间被幸福填满，所有的疲惫都烟消云散。",
            "经过数月努力终于完成的作品获得认可，他长长舒了口气，脸上露出欣慰的笑容。",
            "在异国他乡听到熟悉的乡音，她激动地跑过去，仿佛找到了失散多年的亲人。",
            "看到自己帮助过的人走出困境，他的心里涌起一股暖流，觉得一切付出都值得。"
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
            # 扩展恶心文本
            "看到有人随地吐痰，她立即绕道而行，脸上写满了嫌弃。",
            "揭开垃圾桶的瞬间，腐烂的气味扑面而来，他差点当场呕吐。",
            "菜里发现头发丝，她顿时食欲全无，再也没有动过那盘菜。",
            "听到那些阿谀奉承的虚伪话语，他从心底感到厌恶和不屑。",
            "触摸到黏糊糊的不明物体，她立即冲到洗手间反复清洗双手。"
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
            # 扩展惊喜文本
            "推开家门看到满地的玫瑰和烛光，她惊讶地捂住嘴巴，眼泪在眼眶打转。",
            "原本以为落选的项目突然被告知通过，他愣在原地好几秒才反应过来。",
            "在陌生的城市偶然遇见多年未见的老同学，两人都惊讶得说不出话来。",
            "孩子悄悄准备了生日惊喜，当她看到时感动得热泪盈眶，紧紧抱住孩子。",
            "抽奖时随手一抽竟然中了大奖，他反复确认了好几遍才相信这是真的。"
        ]

        # 合并所有文本
        all_texts = anger_texts + fear_texts + sadness_texts + joy_texts + disgust_texts + surprise_texts

        # 创建标签数组
        anger_labels = [1] * len(anger_texts) + [0] * (len(all_texts) - len(anger_texts))
        fear_labels = [0] * len(anger_texts) + [1] * len(fear_texts) + [0] * (
                    len(all_texts) - len(anger_texts) - len(fear_texts))
        sadness_labels = [0] * (len(anger_texts) + len(fear_texts)) + [1] * len(sadness_texts) + [0] * (
                    len(all_texts) - len(anger_texts) - len(fear_texts) - len(sadness_texts))
        joy_labels = [0] * (len(anger_texts) + len(fear_texts) + len(sadness_texts)) + [1] * len(joy_texts) + [0] * (
                    len(all_texts) - len(anger_texts) - len(fear_texts) - len(sadness_texts) - len(joy_texts))
        disgust_labels = [0] * (len(anger_texts) + len(fear_texts) + len(sadness_texts) + len(joy_texts)) + [1] * len(
            disgust_texts) + [0] * (len(all_texts) - len(anger_texts) - len(fear_texts) - len(sadness_texts) - len(
            joy_texts) - len(disgust_texts))
        surprise_labels = [0] * (
                    len(anger_texts) + len(fear_texts) + len(sadness_texts) + len(joy_texts) + len(disgust_texts)) + [
                              1] * len(surprise_texts)

        # 验证数据长度
        print(f"总文本数量: {len(all_texts)}")
        print(f"愤怒标签数量: {sum(anger_labels)}")
        print(f"恐惧标签数量: {sum(fear_labels)}")
        print(f"悲伤标签数量: {sum(sadness_labels)}")
        print(f"愉悦标签数量: {sum(joy_labels)}")
        print(f"恶心标签数量: {sum(disgust_labels)}")
        print(f"惊喜标签数量: {sum(surprise_labels)}")

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