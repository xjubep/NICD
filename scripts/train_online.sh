echo "[FIVR] - [Online]"
python train_online.py --model mobilenet -d FIVR --save_dir /mldisk/nfs_shared_/sy/FIVR/checkpoints;
python train_online.py --model efficientnet -d FIVR --save_dir /mldisk/nfs_shared_/sy/FIVR/checkpoints;
python train_online.py --model hybrid_vit -d FIVR --save_dir /mldisk/nfs_shared_/sy/FIVR/checkpoints;
python train_online.py --model mobilenet_dolg -d FIVR --save_dir /mldisk/nfs_shared_/sy/FIVR/checkpoints;
python train_online.py --model mobilenet_dolg_df -d FIVR --save_dir /mldisk/nfs_shared_/sy/FIVR/checkpoints;
python train_online.py --model mobilenet_dolg_ff -d FIVR --save_dir /mldisk/nfs_shared_/sy/FIVR/checkpoints;

echo "[DISC] - [Online]"
python train_online.py --model mobilenet -d DISC --save_dir /mldisk/nfs_shared_/sy/DISC/checkpoints;
python train_online.py --model efficientnet -d DISC --save_dir /mldisk/nfs_shared_/sy/DISC/checkpoints;
python train_online.py --model hybrid_vit -d DISC --save_dir /mldisk/nfs_shared_/sy/DISC/checkpoints;
python train_online.py --model mobilenet_dolg -d DISC --save_dir /mldisk/nfs_shared_/sy/DISC/checkpoints;
python train_online.py --model mobilenet_dolg_df -d DISC --save_dir /mldisk/nfs_shared_/sy/DISC/checkpoints;
python train_online.py --model mobilenet_dolg_ff -d DISC --save_dir /mldisk/nfs_shared_/sy/DISC/checkpoints;