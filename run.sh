#linear-large
python train_lifting.py --dataset 3dhp --batch_size 1024 --num_epoch 50 --lr 2e-4 --task_net linear-large \
    --type baseline --no_logging

python train_lifting.py --dataset 3dhp --batch_size 1024 --num_epoch 50 --lr 2e-4 --task_net linear-large \
    --type dynamic --em_loss_type margin \
    --energy_weight 1e-3 --lr_loss 5e-4 --em_loss mpjpe --no_logging
#semgcn
python train_lifting.py --dataset 3dhp --batch_size 1024 --num_epoch 50 --lr 1e-2 --task_net semgcn --task_dropout 0 \
    --type baseline --no_logging

python train_lifting.py --dataset 3dhp --batch_size 1024 --num_epoch 50 --lr 1e-2 --task_net semgcn --task_dropout 0 \
    --type dynamic --em_loss_type margin --centering hip --num_samples 1 \
    --energy_weight 1e-4 --lr_loss 1e-3 --em_loss mpjpe --no_logging
#videopose
#직접 찾아보기

 