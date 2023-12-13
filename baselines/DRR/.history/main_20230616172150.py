# coding=utf-8
import sys
import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# 添加src目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
sys.path.append(os.path.dirname(BASE_DIR))              # 将src目录添加到环境

from model import EEModel
from model import EEPredictor
import util

util.set_random_seed(20230606)
# os.environ["TOKENIZERS_PARALLELISM"] = "True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def gen_args():
    WORKING_DIR = "."

    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=1, choices=[1, 0])
    parser.add_argument("--debug", type=int, default=1, choices=[1, 0])
    parser.add_argument("--is_train", type=util.str2bool, default=True, help="train the EE model or not (default: False)")
    parser.add_argument("--batch_size", type=int, default=16, help="input batch size for training and test (default: 8)")
    parser.add_argument("--max_epochs", type=int, default=20, help="the max epochs for training and test (default: 5)")
    parser.add_argument("--lr", type=float, default=2e-6, help="learning rate (default: 2e-5)")
    parser.add_argument("--crf_lr", type=float, default=0.1, help="crf learning rate (default: 0.1)")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout (default: 0.2)")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"], help="optimizer")
    parser.add_argument("--data_type", type=str, default="paragraph", help="paragraph or sentence")
    parser.add_argument("--grained", type=str, default="coarse", help="coarse or fine")

    parser.add_argument("--use_bert", type=util.str2bool, default=True,
                        help="whether to use bert training or not (default: True)")
    parser.add_argument("--use_crf", type=util.str2bool, default=True,
                        help="whether to use crf layer training or not (default: True)")

    parser.add_argument("--use-bilstm", type=util.str2bool, default=True,
                        help="whether to use bilstm training or not (default: True)")
    parser.add_argument("--lstm_dropout_prob", type=float, default=0.5, help="lstm dropout probability")
    parser.add_argument("--lstm_embedding_size", type=int, default=512, help="lstm embedding size")
    parser.add_argument("--hidden_size", type=int, default=1024, help="bilstm hidden size")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="hidden dropout probability")

    # 下面参数基本默认
    # parser.add_argument("--train_path", type=str, default="{}/data/en/conll16st-en-01-12-16-train/connective.csv".format(WORKING_DIR),
    #                     help="train_path")
    parser.add_argument("--train_path", type=str, default="{}/data/paragraph/train.json".format(WORKING_DIR),
                        help="train_path")
    # parser.add_argument("--dev_path", type=str, default="{}/data/en/conll16st-en-01-12-16-dev/connective.csv".format(WORKING_DIR),
    #                     help="dev_path")
    parser.add_argument("--dev_path", type=str, default="{}/data/paragraph/test.json".format(WORKING_DIR),
                        help="dev_path")
    parser.add_argument("--train_num", type=int, default=-1,help="train data number")
    parser.add_argument("--dev_num", type=int, default=-1,help="train data number")
    parser.add_argument("--test_path", type=str, default="{}/data/paragraph/test.json".format(WORKING_DIR),
                        help="test_path")
    parser.add_argument("--relation_type", type=str, default="Explicit".format(WORKING_DIR),
                    help="test_path")

    
    parser.add_argument("--ee_result_path", type=str, default="{}/result".format(WORKING_DIR),
                        help="ee_result_path")
    parser.add_argument("--ckpt_save_path", type=str,
                        default="{}/weights".format(WORKING_DIR), help="ckpt_save_path")
    parser.add_argument("--resume_ckpt", type=str,
                        default=None, help="checkpoint file name for resume")
    parser.add_argument("--pretrained_path", type=str,
                        default="hfl/chinese-bert-wwm-ext", help="pretrained_path")
    # parser.add_argument("--pretrained_path", type=str,
    #                     default="/home/lawson/program/pretrain_bert/user_data/pretrain_model/checkpoint-21800", help="pretrained_path")

    parser.add_argument("--ckpt_name",  type=str, default="base", help="ckpt save name")
    parser.add_argument("--ner_save_path",type=str, default="weights", help="ner save path")
    parser.add_argument("--test_ckpt_name",  type=str, default="base_epoch=20_val_f1=67.4.ckpt", help="ckpt name for test")

    args = parser.parse_args()
    # args.ckpt_save_path = "{}/{}_weights".format(WORKING_DIR,args.data_type)
    return args

if __name__ == '__main__':

    args1 = gen_args()

    print('--------config----------')
    print(args1)
    print('--------config----------')

    if args1.is_train == True:
        # ============= train 训练模型==============
        print("start train model ...")
        model = EEModel(args1)

        # 设置保存模型的路径及参数
        ckpt_callback = ModelCheckpoint(
            dirpath=args1.ckpt_save_path,                           # 模型保存路径
            filename=args1.pretrained_path + args1.ckpt_name + "_{epoch}_{val_f1:.1f}",   # 模型保存名称，参数ckpt_name后加入epoch信息以及验证集分数
            monitor='val_f1',                                      # 根据验证集上的准确率评估模型优劣
            mode='max',
            save_top_k=2,                                           # 保存得分最高的前两个模型
            verbose=True,
        )

        resume_checkpoint=None
        if args1.resume_ckpt:
            resume_checkpoint=os.path.join(args1.ckpt_save_path ,args1.resume_ckpt)   # 加载已保存的模型继续训练

        # 设置训练器
        trainer = pl.Trainer(
            progress_bar_refresh_rate=1,
            resume_from_checkpoint = resume_checkpoint,
            max_epochs=args1.max_epochs,
            callbacks=[ckpt_callback],
            checkpoint_callback=True,
            gpus=1
        )

        # 开始训练模型
        trainer.fit(model)

        # 只训练CRF的时候，保存最后的模型
        # if config.use_crf and config.first_train_crf == 1:
        #     trainer.save_checkpoint(os.path.join(config.ner_save_path, 'crf_%d.ckpt' % (config.max_epochs)))
    else:
        # ============= test 测试模型==============
        print("\n\nstart test model...")

        outfile_txt = os.path.join(args1.ee_result_path, args1.test_ckpt_name[:-5] + ".csv")

        # 开始测试，将结果保存至输出文件
        checkpoint_path = os.path.join(args1.ner_save_path, args1.test_ckpt_name)
        predictor = EEPredictor(checkpoint_path, args1)
        predictor.predict("显式")
