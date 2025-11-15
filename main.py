import argparse
import sys
import os
from predict import ChineseEmotionPredictor
from train import ChineseEmotionTrainer


def ensure_stopwords_dir():
    """确保停用词目录存在"""
    if not os.path.exists('stopwords'):
        os.makedirs('stopwords')
        print("创建停用词目录: stopwords")

        # 创建基础停用词文件
        basic_stopwords = """的
了
在
是
我
有
和
就
不
人
都
一
一个
上
也
很
到
说
要
去
你
会
着
没有
看
好
自己
这
那
他
她
它
我们
你们
他们
这个
那个
这些
那些
这样
那样
这里
那里
这时
那时
什么
为什么
怎么
哪里
谁
几
多少
很
非常
特别
更
最
太
极
挺
相当
有点
一些
一点
啊
呀
呢
吧
吗
啦
唉
哦
嗯
哼
哇
喔
喂
嘛
喽
咚
咦
哟
呃"""

        with open('stopwords/cn_stopwords.txt', 'w', encoding='utf-8') as f:
            f.write(basic_stopwords)
        print("已创建基础停用词文件")


def main():
    parser = argparse.ArgumentParser(description='中文文本情感分析器')
    parser.add_argument('--mode', choices=['train', 'predict', 'interactive'],
                        default='interactive', help='运行模式')
    parser.add_argument('--model', type=str, default='chinese_emotion_model.pkl',
                        help='模型文件路径')
    parser.add_argument('--text', type=str, help='要分析的文本')

    args = parser.parse_args()

    # 确保停用词目录存在
    ensure_stopwords_dir()

    if args.mode == 'train':
        print("开始训练中文情感分析模型...")
        trainer = ChineseEmotionTrainer()
        trainer.train()
        print("训练完成!")

    elif args.mode == 'predict':
        if not args.text:
            print("请使用 --text 参数提供要分析的文本")
            return

        predictor = ChineseEmotionPredictor(args.model)
        result = predictor.predict_emotion(args.text)
        predictor.print_result(result)

    elif args.mode == 'interactive':
        print("欢迎使用中文文本情感分析器!")
        print("输入 'quit' 或 'exit' 退出程序")
        print("-" * 50)

        # 加载模型
        predictor = ChineseEmotionPredictor(args.model)

        while True:
            text = input("\n请输入要分析的中文文本: ").strip()

            if text.lower() in ['quit', 'exit', '退出']:
                print("感谢使用!")
                break

            if not text:
                print("请输入有效文本")
                continue

            try:
                result = predictor.predict_emotion(text)
                predictor.print_result(result)
            except Exception as e:
                print(f"分析过程中出现错误: {e}")


if __name__ == "__main__":
    main()