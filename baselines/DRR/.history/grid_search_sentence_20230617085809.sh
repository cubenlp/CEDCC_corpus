alphas=(bert-base-chinese xlm-roberta-base hfl/chinese-xlnet-base)


for a in ${alphas[@]};
do
    echo $a $t
    CUDA_VISIBLE_DEVICES=2 python main.py  --pretrained_path ${a} --grained fine
done
