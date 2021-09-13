import os
import argparse
import torch
import onnx
from metro.modeling.bert import BertConfig, METRO
from metro.modeling._smpl import SMPL, Mesh
from metro.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net
from metro.modeling.hrnet.config import config as hrnet_config
from metro.modeling.hrnet.config import update_config as hrnet_update_config
import metro.modeling.data.config as cfg

from metro.utils.renderer import Renderer, visualize_reconstruction, visualize_reconstruction_test, visualize_reconstruction_no_text, visualize_reconstruction_and_att_local
from metro.utils.geometric_layers import orthographic_projection
from metro.utils.logger import setup_logger
from metro.utils.miscellaneous import mkdir, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for inference.")
    parser.add_argument("--model_name_or_path", default='metro/modeling/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False,
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False,
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False,
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False,
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str,
                        help="The Image Feature Dimension.")
    parser.add_argument("--hidden_feat_dim", default='1024,256,128', type=str,
                        help="The Image Feature Dimension.")
    parser.add_argument("--legacy_setting", default=True, action='store_true',)
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--device", type=str, default='cuda',
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88,
                        help="random seed for initialization.")

    args = parser.parse_args()
    return args

def main():
    global logger
    logger = setup_logger('MeshTransformer', '.', 0)
    os.environ['NO_APEX'] = 'true'
    from metro.modeling.bert import METRO_Body_Network_ONNX as METRO_Network
    args = parse_args()
    mesh_smpl = SMPL().to(args.device)
    mesh_sampler = Mesh()
    # Build model from scratch, and load weights from state_dict.bin
    trans_encoder = []
    input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
    output_feat_dim = input_feat_dim[1:] + [3]
    # init three transformer encoders in a loop
    for i in range(len(output_feat_dim)):
        config_class, model_class = BertConfig, METRO
        config = config_class.from_pretrained(args.model_name_or_path)

        config.output_attentions = False
        config.img_feature_dim = input_feat_dim[i]
        config.output_feature_dim = output_feat_dim[i]
        args.hidden_size = hidden_feat_dim[i]

        if args.legacy_setting:
            # During our paper submission, we were using the original intermediate size, which is 3072 fixed
            # We keep our legacy setting here
            args.intermediate_size = -1
        else:
            # We have recently tried to use an updated intermediate size, which is 4*hidden-size.
            # But we didn't find significant performance changes on Human3.6M (~36.7 PA-MPJPE)
            args.intermediate_size = int(args.hidden_size * 4)

        # update model structure if specified in arguments
        update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

        for idx, param in enumerate(update_params):
            arg_param = getattr(args, param)
            config_param = getattr(config, param)
            if arg_param > 0 and arg_param != config_param:
                logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                setattr(config, param, arg_param)

        # init a transformer encoder and append it to a list
        assert config.hidden_size % config.num_attention_heads == 0
        model = model_class(config=config)
        logger.info("Init model from scratch.")
        trans_encoder.append(model)
        if args.arch=='hrnet':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w40 model')
        elif args.arch=='hrnet-w64':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w64 model')
        else:
            print("=> using pre-trained model '{}'".format(args.arch))
            backbone = models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

    trans_encoder = torch.nn.Sequential(*trans_encoder)
    total_params = sum(p.numel() for p in trans_encoder.parameters())
    logger.info('Transformers total parameters: {}'.format(total_params))
    backbone_total_params = sum(p.numel() for p in backbone.parameters())
    logger.info('Backbone total parameters: {}'.format(backbone_total_params))

    # build end-to-end METRO network (CNN backbone + multi-layer transformer encoder)
    _metro_network = METRO_Network(args, config, backbone, trans_encoder, mesh_sampler, mesh_smpl)

    logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
    cpu_device = torch.device('cpu')
    state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
    _metro_network.load_state_dict(state_dict, strict=False)
    del state_dict
    setattr(_metro_network.trans_encoder[-1].config,'output_attentions', True)
    setattr(_metro_network.trans_encoder[-1].config,'output_hidden_states', True)
    _metro_network.trans_encoder[-1].bert.encoder.output_attentions = True
    _metro_network.trans_encoder[-1].bert.encoder.output_hidden_states =  True
    for iter_layer in range(4):
        _metro_network.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True
    for inter_block in range(3):
        setattr(_metro_network.trans_encoder[-1].config,'device', args.device)

    _metro_network.to(args.device)
    _metro_network.eval()
    batch_imgs = torch.zeros((1, 3, 224, 224)).to(args.device)
    logger.info('Run model export')
    with torch.no_grad():
        torch.onnx.export(_metro_network, batch_imgs, 'mesh_transformer.onnx', opset_version=11, output_names=[
            'pred_camera', 'pred_3d_joints', 'pred_vertices_sub2', 'pred_vertices_sub', 'pred_vertices', 'hidden_states', 'att'
        ])

    logger.info('Check exported model')
    onnx_model = onnx.load("mesh_transformer.onnx")
    onnx.checker.check_model(onnx_model)


if __name__ == "__main__":
    main()
