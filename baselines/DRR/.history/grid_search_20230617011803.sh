alphas=(bert-base-chinese xlm-roberta-base hfl/chinese-xlnet-base)
ts=(4 5)


for a in ${alphas[@]};
do
    for t in ${ts[@]};
    do
        echo $a $t
        CUDA_VISIBLE_DEVICES=2 python main.py  --pretrained_path ${a} --grained fine
    done
done
