from Data import DataGenerator
from CKA2 import plot_cka_graph
from CKA2 import CKA
import torch
import timm
import json


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--timm_model1_name', type=str,
                        default='vit_tiny_patch16_224')
    parser.add_argument('--timm_model2_name', type=str,
                        default='swin_tiny_patch4_window7_224')
    parser.add_argument('--pretrained', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=5)

    parser.add_argument('--model1_name', type=str, default='vit tiny')
    parser.add_argument('--model2_name', type=str, default='swin tiny')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cka_save_path', type=str,
                        default='./outputs/cka2_info.json')
    parser.add_argument('--fig_save_path', type=str,
                        default='./outputs/cka2_info.png')

    parser.add_argument('--train_path', type=str,
                        default='./data/cassava_sample_data/train')
    parser.add_argument('--valid_path', type=str,
                        default='./data/cassava_sample_data/valid')
    parser.add_argument('--test_path', type=str,
                        default='./data/cassava_sample_data/test')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--apply_cutmix', type=int, default=0)

    parser.add_argument('--use_fine_tune', type=int, default=0)
    parser.add_argument('--model1_path', type=str,
                        default='./models/vit_tiny_patch16_224.pth')
    parser.add_argument('--model2_path', type=str,
                        default='./models/swin_tiny_patch4_window7_224.pth')

    args = parser.parse_args()

    print(args)

    timm_model1_name = args.timm_model1_name
    timm_model2_name = args.timm_model2_name
    pretrained = args.pretrained
    num_classes = args.num_classes
    device = args.device
    model1 = timm.create_model(timm_model1_name, pretrained=pretrained,
                               num_classes=num_classes)
    model2 = timm.create_model(timm_model2_name, pretrained=pretrained,
                               num_classes=num_classes)

    use_fine_tune = args.use_fine_tune
    model1_path = args.model1_path
    model2_path = args.model2_path
    if use_fine_tune:
        model1.load_state_dict(torch.load(model1_path))
        model2.load_state_dict(torch.load(model2_path))

    model1.to(device)
    model2.to(device)

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
    valid_loader = dataloaders_dict['valid_dataloader']

    model1_name = args.model1_name
    model2_name = args.model2_name
    device = args.device
    cka_save_path = args.cka_save_path
    fig_save_path = args.fig_save_path

    cka = CKA(model1, model2, model1_name, model2_name, None, None, device)

    cka.compare(valid_loader)

    cka_info = cka.export(cka_save_path)

    with open(cka_save_path, 'r', encoding='utf-8') as fp:
        info = json.load(fp)

    model1_name = info["model1_name"]
    model2_name = info["model2_name"]
    col1 = info["col1"]
    col2 = info["col2"]
    hsicScoreList = info["hsicScoreList"]
    title = f"{model1_name} vs {model2_name}"
    plot_cka_graph(col1, col2, hsicScoreList, title, model1_name,
                   model2_name, fig_save_path)
