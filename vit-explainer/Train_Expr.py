from Loss import FocalLoss, BiTemperedLogisticLoss
from torch.optim.lr_scheduler import ExponentialLR
from Trainer import Trainer
from torch.optim import Adam
from Data import DataGenerator
import torch.nn as nn
import timm
import os


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--timm_model_name', type=str,
                        default='vit_tiny_patch16_224')
    parser.add_argument('--pretrained', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--train_path', type=str,
                        default='./data/cassava_sample_data/train')
    parser.add_argument('--valid_path', type=str,
                        default='./data/cassava_sample_data/valid')
    parser.add_argument('--test_path', type=str,
                        default='./data/cassava_sample_data/test')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--loss_type', type=str, default='cross_entropy')
    parser.add_argument('--apply_cutmix', type=int, default=0)

    parser.add_argument('--model_path', type=str,
                        default='./models/vit_tiny_patch16_224.pth')
    parser.add_argument('--log_path', type=str,
                        default='./outputs/vit_tiny_patch16_224_log.json')
    parser.add_argument('--conf_path', type=str,
                        default='./outputs/vit_tiny_patch16_224_conf.csv')
    parser.add_argument('--metric_path', type=str,
                        default='./outputs/vit_tiny_patch16_224_metric.csv')

    args = parser.parse_args()

    print(args)

    timm_model_name = args.timm_model_name
    pretrained = args.pretrained
    num_classes = args.num_classes
    device = args.device
    model = timm.create_model(timm_model_name, pretrained=pretrained,
                              num_classes=num_classes)
    model.to(device)

    train_path = args.train_path
    valid_path = args.valid_path
    test_path = args.test_path
    batch_size = args.batch_size
    image_size = args.image_size
    apply_cutmix = args.apply_cutmix

    generator = DataGenerator(train_path, valid_path, test_path,
                              batch_size, image_size,
                              apply_cutmix)
    dataloaders_dict = generator.generate_dataloader()
    train_loader = dataloaders_dict['train_dataloader']
    valid_loader = dataloaders_dict['valid_dataloader']
    test_loader = dataloaders_dict['test_dataloader']

    loss_type = args.loss_type
    if loss_type == 'focal_loss':
        criterion = FocalLoss()
    if loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(
            reduction="sum", label_smoothing=0.1)
    if loss_type == 'bitemper_log':
        criterion = BiTemperedLogisticLoss(
            reduction='sum', t1=0.7, t2=1.3, label_smoothing=0.3)

    optimizer = Adam(model.parameters(), lr=1.5e-5)
    scheduler = ExponentialLR(optimizer, gamma=0.97)

    num_epochs = args.num_epochs
    device = args.device
    log_path = args.log_path
    model_path = args.model_path
    conf_path = args.conf_path
    metric_path = args.metric_path

    if not os.path.exists(model_path):
        trainer = Trainer(model, num_epochs, device, train_loader, valid_loader,
                          test_loader, optimizer, criterion,
                          log_path, model_path, conf_path,
                          metric_path, scheduler)
        trainer.finetune_model()
        trainer.evaluate_model()
