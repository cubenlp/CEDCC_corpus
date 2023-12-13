alphas=(yechen/bert-large-chinese xlm-roberta-large hfl/chinese-xlnet-mid)


for a in ${alphas[@]};
do
    echo $a $t
    CUDA_VISIBLE_DEVICES=3 python main.py  --pretrained_path ${a} --grained fine --batch_size 4
done
