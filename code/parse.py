import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Go VAGNN")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of VAGNN")
    parser.add_argument('--bpr_batch', type=int, default=4096,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--layer', type=int,default=2,
                        help="the layer num of VAGNN")
    parser.add_argument('--lr', type=float,default=1e-3,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--testbatch', type=int,default=128,
                        help="the batch size of users for testing")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[10,20,50]",
                        help="@k test list")
    parser.add_argument('--load', type=int,default=1)
    parser.add_argument('--epochs', type=int,default=200)
    parser.add_argument('--multicore', type=int, default=1, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')

    parser.add_argument('--neg', type=int, default=1, help='')

    parser.add_argument('--projection_dim', type=int, default=16, help='')
    parser.add_argument('--projection_dropout', type=int, default=0.5, help='')

    parser.add_argument('--gpu_id', type=str, default='0', help='')
    parser.add_argument('--dataset', type=str, default='takatak',
                        help="available datasets: [wechat, takatak]")
    # wechat 5 takatak 5
    parser.add_argument('--vlogger_reg', type=int, default=5, help='')
    # wechat 0.0005 takatak 0.05
    parser.add_argument('--cl_reg', type=int, default=0.05, help='')
    #wechat 0.5 takatak 0.05
    parser.add_argument('--cl_temp', type=int, default=0.05, help='')

    return parser.parse_args()