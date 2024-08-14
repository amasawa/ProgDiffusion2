from templates import *
from templates_latent import *
import sys

if __name__ == '__main__':
    # train the autoenc moodel
    # this requires V100s.
    gpus = [2,3]
    conf = ffhq128_autoenc_130M()
    train(conf, gpus=gpus, mode='train')
    # print("OK")

    # train the latent DPM
    # NOTE: only need a single gpu
    # gpus = [0]
    # conf = ffhq128_autoenc_latent()
    # train(conf, gpus=gpus)

    # unconditional sampling score
    # NOTE: a lot of gpus can speed up this process
    # gpus = [2,3]
    #conf.eval_programs = ['fid10']
    # train(conf, gpus=gpus, mode='eval')
    # if len(sys.argv) > 1:
    #     eval_program = sys.argv[1]
    #     #conf.eval_programs = [eval_program]
    #     conf.wzk_ckpt = eval_program
    #
    #train(conf, gpus=gpus, mode='eval')
