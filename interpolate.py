from templates import *
from torchvision.utils import save_image

device = 'cuda:2'
conf = horse128_autoenc()
#conf = bedroom128_autoenc()
#conf = ffhq128_autoenc_130M()
# print(conf.name)
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/epoch=63-step=612000.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

data = ImageDataset('imgs_interpolate/horse2', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
batch = torch.stack([
    data[0]['img'],
    data[1]['img'],
])

cond = model.encode(batch.to(device),torch.zeros([1]).to(device))
#cond = model.encode(batch.to(device))
xT = model.encode_stochastic(batch.to(device), cond, T=40)


import numpy as np
alpha = torch.tensor(np.linspace(-0.08, 1.1, 10, dtype=np.float32)).to(cond.device)
intp = cond[0][None] * (1 - alpha[:, None]) + cond[1][None] * alpha[:, None]

def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()

theta = torch.arccos(cos(xT[0], xT[1]))
x_shape = xT[0].shape
intp_x = (torch.sin((1 - alpha[:, None]) * theta) * xT[0].flatten(0, 2)[None] + torch.sin(alpha[:, None] * theta) * xT[1].flatten(0, 2)[None]) / torch.sin(theta)
intp_x = intp_x.view(-1, *x_shape)

pred = model.render(intp_x, intp, T=100)
grid_img = save_image(pred.cpu(), 'horse-diffae.png', nrow=10, normalize=True)