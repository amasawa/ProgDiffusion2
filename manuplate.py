from templates import *
from templates_cls import *
from experiment_classifier import ClsModel
from torchvision.utils import save_image

device = 'cuda:3'
conf = ffhq128_autoenc_130M()
# print(conf.name)
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)



cls_conf = ffhq128_autoenc_cls()
cls_model = ClsModel(cls_conf)
state = torch.load(f'checkpoints/{cls_conf.name}/last.ckpt',
                    map_location='cpu')
print('latent step:', state['global_step'])
cls_model.load_state_dict(state['state_dict'], strict=False);
cls_model.to(device)

# ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
#  'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
#  'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
#  'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
#  'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

data = ImageDataset('imgs/time', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)

#batch = data[0]['img'][None]
batch = torch.stack([
    data[0]['img'],
])


cond = model.encode(batch.to(device))
xT = model.encode_stochastic(batch.to(device), cond, T=250)

xT = xT.repeat(10, 1, 1,1)
cls_id = CelebAttrDataset.cls_to_id['Bags_Under_Eyes']

conList = []
for item in torch.linspace(0,0.5,10):
    cond2 = cls_model.normalize(cond)
    cond2 = cond2 + item* math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
    cond2 = cls_model.denormalize(cond2)
    conList.append(cond2)


img = model.render(xT, torch.squeeze(torch.stack(conList)), T=100)
# grid_img = save_image(img.cpu(), 'dis_rebuttal/Mustache.png', nrow=10, normalize=True)
for idx, single_img in enumerate(img):
    save_image(single_img.cpu(), f'dis_rebuttal/42_{idx}.png', normalize=True)