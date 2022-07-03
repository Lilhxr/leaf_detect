from Data import DataGenerator
from CKA import save_results
from CKA import CKA
import timm
import json


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--timm_model1_name', type=str,
                        default='vit_tiny_patch16_224')
    parser.add_argument('--timm_model2_name', type=str,
                        default='swin_tiny_patch4_window7_224')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--num_classes', type=int, default=5)

    parser.add_argument('--model1_name', type=str, default='vit tiny')
    parser.add_argument('--model2_name', type=str, default='swin tiny')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cka_save_path', type=str, default='./cka_info.json')
    parser.add_argument('--fig_save_path', type=str, default='./cka_info.png')

    parser.add_argument('--train_path', type=str,
                        default='./data/cassava_sample_data/train')
    parser.add_argument('--valid_path', type=str,
                        default='./data/cassava_sample_data/valid')
    parser.add_argument('--test_path', type=str,
                        default='./data/cassava_sample_data/test')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=224)

    args = parser.parse_args()

    print(args)

    timm_model1_name = args.timm_model1_name
    timm_model2_name = args.timm_model2_name
    pretrained = args.pretrained
    num_classes = args.num_classes
    model1 = timm.create_model(timm_model1_name, pretrained=pretrained,
                               num_classes=num_classes)
    model2 = timm.create_model(timm_model2_name, pretrained=pretrained,
                               num_classes=num_classes)

    train_path = args.train_path
    valid_path = args.valid_path
    test_path = args.test_path
    batch_size = args.batch_size
    image_size = args.image_size

    generator = DataGenerator(train_path, valid_path, test_path,
                              batch_size, image_size)
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
    hsic_matrix = info["CKA"]
    title = f"{model1_name} vs {model2_name}"
    save_results(fig_save_path, title,
                 model1_name, model2_name,
                 hsic_matrix)
