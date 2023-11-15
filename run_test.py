import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from collections import OrderedDict

import torch
import torchvision.utils as vutils

import utils
from models.ir import IR

sys.path.append("./")


# Arguments for demo
def add_example_parser(parser):
    parser.add_argument(
        '--image_path', type=str, default='',
        help='Path of the image to be processed (default: %(default)s)')
    parser.add_argument(
        '--class', type=int, default=-1,
        help='class index of the image (default: %(default)s)')
    parser.add_argument(
        '--image_path2', type=str, default='',
        help='Path of the 2nd image to be processed, used in "morphing" mode (default: %(default)s)')
    parser.add_argument(
        '--class2', type=int, default=-1,
        help='class index of the 2nd image, used in "morphing" mode (default: %(default)s)')
    return parser


# prepare arguments and save in config
parser = utils.prepare_parser()
parser = utils.add_ir_parser(parser)
parser = add_example_parser(parser)
config = vars(parser.parse_args())
utils.ir_update_config(config)

# set random seed
utils.seed_rng(config['seed'])

if not os.path.exists('{}/images'.format(config['exp_path'])):
    os.makedirs('{}/images'.format(config['exp_path']))
if not os.path.exists('{}/images_sheet'.format(config['exp_path'])):
    os.makedirs('{}/images_sheet'.format(config['exp_path']))

ir = IR(config)

img = utils.get_img(config['image_path'], config['resolution']).cuda()
category = torch.Tensor([config['class']]).long().cuda()
ir.set_target(img, category, config['image_path'])

ir.select_z(select_y=True if config['class'] < 0 else False)

loss_dict = ir.run()

if config['ir_mode'] == 'category_transfer':
    save_imgs = img.clone().cpu()
    for i in range(151, 294):  # dog & cat
    # for i in range(7, 25):  # bird
        with torch.no_grad():
            x = ir.G(ir.z, ir.G.shared(ir.y.fill_(i)))
            utils.save_img(
                x[0],
                '%s/images/%s_class%d.jpg' % (config['exp_path'], ir.img_name, i))
            save_imgs = torch.cat((save_imgs, x.cpu()), dim=0)
    vutils.save_image(
        save_imgs,
        '%s/images_sheet/%s_categories.jpg' % (config['exp_path'], ir.img_name),
        nrow=int(save_imgs.size(0)**0.5),
        normalize=True)

elif config['ir_mode'] == 'morphing':
    ir2 = IR(config)
    ir_interp = IR(config)

    img2 = utils.get_img(config['image_path2'], config['resolution']).cuda()
    category2 = torch.Tensor([config['class2']]).long().cuda()

    ir2.set_target(img2, category2, config['image_path2'])
    ir2.select_z(select_y=True if config['class2'] < 0 else False)
    loss_dict = ir2.run()

    weight1 = ir.G.state_dict()
    weight2 = ir2.G.state_dict()
    weight_interp = OrderedDict()
    save_imgs = []
    with torch.no_grad():
        for i in range(11):
            alpha = i / 10
            z_interp = alpha * ir.z + (1 - alpha) * ir2.z
            y_interp = alpha * ir.G.shared(ir.y) + (1 - alpha) * ir2.G.shared(ir2.y)
            for k, w1 in weight1.items():
                w2 = weight2[k]
                weight_interp[k] = alpha * w1 + (1 - alpha) * w2
            ir_interp.G.load_state_dict(weight_interp)
            x_interp = ir_interp.G(z_interp, y_interp)
            save_imgs.append(x_interp.cpu())
            # save images
            save_path = '%s/images/%s_%s' % (config['exp_path'], ir.img_name, ir2.img_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            utils.save_img(x_interp[0], '%s/%03d.jpg' % (save_path, i + 1))
        save_imgs = torch.cat(save_imgs, 0)
    vutils.save_image(
        save_imgs,
        '%s/images_sheet/morphing_%s_%s.jpg' % (config['exp_path'], ir.img_name, ir2.img_name),
        nrow=int(save_imgs.size(0)**0.5),
        normalize=True)
