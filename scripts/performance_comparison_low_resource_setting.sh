CUDA_VISIBLE_DEVICES=$1 python Finetuning_SpamFiltering.py --epoch 7 --resource low --test_company y --method CM
CUDA_VISIBLE_DEVICES=$1 python Finetuning_SpamFiltering.py --epoch 7 --resource low --test_company y --method SM
CUDA_VISIBLE_DEVICES=$1 python Finetuning_SpamFiltering.py --epoch 7 --resource low --test_company y --method WWM
CUDA_VISIBLE_DEVICES=$1 python Finetuning_SpamFiltering.py --epoch 7 --resource low --test_company y --method NoPT
CUDA_VISIBLE_DEVICES=$1 python Finetuning_SpamFiltering.py --epoch 7 --resource low --test_company n --method CM
CUDA_VISIBLE_DEVICES=$1 python Finetuning_SpamFiltering.py --epoch 7 --resource low --test_company n --method SM
CUDA_VISIBLE_DEVICES=$1 python Finetuning_SpamFiltering.py --epoch 7 --resource low --test_company n --method WWM
CUDA_VISIBLE_DEVICES=$1 python Finetuning_SpamFiltering.py --epoch 7 --resource low --test_company n --method NoPT