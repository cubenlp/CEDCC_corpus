alphas=(bert-base-chinese xlm-roberta-base hfl/chinese-xlnet-base)
ts=(4 5)


for a in ${alphas[@]};
do
    for t in ${ts[@]};
    do
        echo $a $t
        CUDA_VISIBLE_DEVICES=2 python MLM_connective_distillation/EE_main.py --template_version 777 --alpha ${a} --temperature_rate ${t} --version true_KLDLoss_roberta_base_masked_model+ls=0.05+alpha=${a}_tr=${t}_maxlen=250_bsz=16_agb=2_teacher+temp_true_test_epoch=8_20221026_lr=1e-5_template1_trueS --batch_size 16 --data_type pdtb3_top --pretrained_path roberta-base --teacher_pretrained_path roberta-base --teacher_checkpoint_path mlm_test_all_weights/basepdtb3_top_template_777_epoch=2_val_acc=63.77_val_f1=51.29.ckpt --train_path data/pdtb3/train.tsv --dev_path data/pdtb3/test.tsv --lr 1e-5  > base_log/pdtb3_log_template777_trueS_true_KLDLoss/pdtb3_alpha=${a}+tr=${t}.log
    done
done
