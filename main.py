import argparse
import pdb
import experiments as exp
from utils import configs


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('conf_path', type=str, metavar='conf_path')
    parser.add_argument('--exp', type=str, default="ae")
    args = parser.parse_args()
    K = 5
    for k in range(K):
        if args.exp == "ae":
            config = configs.ExperimentConfig(args.conf_path)
            experiment = exp.ae.AEExp(config, k)
        elif args.exp == "betavae":
            config = configs.BetaVAEConfig(args.conf_path)
            experiment = exp.betavae.BetaVAEExp(config, k)
        elif args.exp == "lssl":
            config = configs.ExperimentConfig(args.conf_path)
            experiment = exp.lssl.LSSLExp(config, k)
        elif args.exp == "simclr":
            config = configs.ExperimentConfig(args.conf_path)
            experiment = exp.simclr.SimCLRExp(config, k)
        elif args.exp == "loca":
            config = configs.LoCAConfig(args.conf_path)
            experiment = exp.loca.LoCAExp(config, k)
        else:
            raise NotImplementedError
        experiment.run()
    pdb.set_trace()

if __name__ == '__main__':
    main()