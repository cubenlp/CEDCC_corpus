alphas=(yechen/bert-large-chinese xlm-roberta-large hfl/chinese-xlnet-mid)


for a in ${alphas[@]};
do
    echo $a $t
    CUDA_VISIBLE_DEVICES=2 python main.py  --pretrained_path ${a} --grained fine --data_type sentence --train_path /home/hongyi/EMNLP_2023/DRR/data/sentence/train.json --dev_path /home/hongyi/EMNLP_2023/DRR/data/sentence/test.json --test_path /home/hongyi/EMNLP_2023/DRR/data/sentence/test.json --batch_size 4
done
