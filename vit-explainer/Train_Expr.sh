python Train_Expr.py --timm_model_name 'swin_tiny_patch4_window7_224' \
                     --pretrained 1 \
                     --num_classes 5 \
                     --device 'cuda' \
                     --train_path './data/cassava_sample_data/train' \
                     --valid_path './data/cassava_sample_data/valid' \
                     --test_path './data/cassava_sample_data/test' \
                     --batch_size 16 \
                     --num_epochs 20 \
                     --image_size 224 \
                     --loss_type 'cross_entropy' \
                     --apply_cutmix 0 \
                     --model_path './models/swin_tiny_patch4_window7_224.pth' \
                     --log_path './outputs/swin_tiny_patch4_window7_224_log.json' \
                     --conf_path './outputs/swin_tiny_patch4_window7_224_conf.csv' \
                     --metric_path './outputs/swin_tiny_patch4_window7_224_metric.csv' \
                     