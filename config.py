import  argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', type=str, default='YAGO')
args.add_argument('--time-stamp', type=int, default=1)
args.add_argument('-lr', type=float, default=0.001)
args.add_argument('--n-epochs', type=int, default=30)
args.add_argument('--hidden-dim', type=int, default=200)
args.add_argument("-gpu", type=int, default=0,
                  help="gpu")
args.add_argument("--regularization", type=float, default=0.01,
                    help="regularization weight")
args.add_argument('--valid-epoch', type=int, default=5)
args.add_argument('-alpha', type=float, default=0.5)
args.add_argument('--batch-size', type=int, default=1024)
args.add_argument('--counts', type=int, default=4)
args.add_argument('--entity', type=str, default='subject')

# args.add_argument('--raw', action='store_true', default=False) # no longer used, modified by eval_paper_authors- now used: setting
# added by eval_paper_authors: for logging
args.add_argument('--setting', type=str, default='static', choices=['raw', 'static', 'time'],
                    help="filtering setting")
args.add_argument("--runnr", default=0, type=int) 
args.add_argument("--multi_step", default=True, type=bool, help="do multi-steps inference without ground truth")
args.add_argument("--feedvalid", default=True, type=bool, help="do we feed the validation set for testing?")
### end added eval_paper_authors

args = args.parse_args()
print(args)


