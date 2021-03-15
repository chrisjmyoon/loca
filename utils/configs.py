import pdb
from pyhocon import ConfigFactory
import os
"""
Loads configurations set in a .hocon file to a Python object
Refer to the hocon file for description of variables
"""

class BaseConfig:
    def __init__(self, conf_path):
        super().__init__()
        self.conf = ConfigFactory.parse_file(conf_path)
        # ------------- general options ----------------------------------------

        self.save_path = os.path.expanduser(self.conf['save_path'])
        self.gpu = self.conf['gpu'] 
        self.print_freq = self.conf['print_freq']
        self.num_workers = self.conf['num_workers']  

        # ------------- experiment options -------------------------------------
        self.test_mode = self.conf['test_mode']
        self.mode = self.conf['mode']

        # ------------- data options -------------------------------------------
        self.split_dir = os.path.expanduser(self.conf['split_dir'])
        self.shuffle = self.conf['shuffle']
        self.pin_memory = self.conf['pin_memory']
        self.rand_seed = self.conf['rand_seed']
        # ------------- common optimization options ----------------------------
        self.batch_size = self.conf['batch_size']  
        self.desired_bs = self.conf['desired_bs']
        self.momentum = self.conf['momentum'] 
        self.weight_decay = self.conf['weight_decay'] 
        self.lr = self.conf['lr']   
        self.epochs = self.conf['epochs']   
        self.scheduler = self.conf['scheduler']  

        # ------------- logging options ----------------------------------------
        self.visdom = self.conf['visdom']
       
        # ---------- resume or retrain options ---------------------------------
        self.retrain = None if len(self.conf['retrain']) == 0 else os.path.expanduser(self.conf['retrain'])
        self.resume = None if len(self.conf['resume']) == 0 else os.path.expanduser(self.conf['resume'])
        self.notes = self.conf['notes']

        # load from remote configs if remote is True
        self.remote_conf = ConfigFactory.parse_file("configs/remote.hocon")
        if self.remote_conf["remote"]:
            self.remote = self.remote_conf["remote"]
            self.test_mode = self.remote_conf["test_mode"]
            self.num_workers = self.remote_conf["num_workers"]
            self.print_freq = self.remote_conf["print_freq"]
            self.batch_size = self.remote_conf["batch_size"]


class DatasetConfig(BaseConfig):
    def __init__(self, conf_path):
        super().__init__("configs/base.hocon")
        self.datasets = dict()

    def add_dataset(self, dataset_path, name):
        data_conf_obj = dict()
        data_conf = ConfigFactory.parse_file(dataset_path)
        data_conf_obj['data_path'] = os.path.expanduser(data_conf['data_path'])
        data_conf_obj['label1'] = os.path.expanduser(data_conf['label1'])
        data_conf_obj['label2'] = os.path.expanduser(data_conf['label2'])
        data_conf_obj['meta_path'] = os.path.expanduser(data_conf['meta_path'])
        self.datasets[name] = data_conf_obj

    def add_both_datasets(self):
        self.both_data = dict(
            adni = None,
            aud = None
        )
        
        def create_data_conf(conf_path):
            data_conf_obj = dict()
            data_conf = ConfigFactory.parse_file(conf_path)
            data_conf_obj["data_path"] = os.path.expanduser(data_conf['data_path'])
            data_conf_obj["split1_path"] = os.path.expanduser(data_conf['split1_path'])
            data_conf_obj["split2_path"] = os.path.expanduser(data_conf['split2_path'])
            data_conf_obj["label1"] = data_conf['label1']
            data_conf_obj["label2"] = data_conf['label2']
            data_conf_obj["meta_path"] = os.path.expanduser(data_conf['meta_path'])
            data_conf_obj["normalize_mean_std"] = data_conf['normalize_mean_std']
            return data_conf_obj
        adni_conf_obj = create_data_conf(adni_conf_path)
        aud_conf_obj = create_data_conf(aud_conf_path)
        self.both_data["adni"] = adni_conf_obj
        self.both_data["aud"] = aud_conf_obj        


class ExperimentConfig(DatasetConfig):
    def __init__(self, conf_path):
        super().__init__(conf_path)
        self.exp_conf = ConfigFactory.parse_file(conf_path)
        self.eval = EvalConfig()
        self.dataset = self.exp_conf['dataset']

        # add dataset confs
        adni_conf_path = "configs/adni.hocon"
        aud_conf_path = "configs/aud.hocon"
        self.add_dataset(adni_conf_path, "adni")
        self.add_dataset(aud_conf_path, "aud")
        self.data_path = self.datasets[self.dataset]["data_path"]
        # add experiment confs
        self.len_z = self.exp_conf["len_z"]
        self.exp_id = self.exp_conf['exp_id']
        self.test_mode = self.exp_conf['test_mode']
        self.mode = self.exp_conf['mode']
        self.retrain = None if len(self.exp_conf['retrain']) == 0 else os.path.expanduser(self.exp_conf['retrain'])
        self.resume = None if len(self.exp_conf['resume']) == 0 else os.path.expanduser(self.exp_conf['resume'])
        self.notes = self.conf['notes']

class LoCAConfig(ExperimentConfig):
    def __init__(self, conf_path):        
        super().__init__(conf_path)
        self.len_k = self.exp_conf["len_k"]

class BetaVAEConfig(ExperimentConfig):
    def __init__(self, conf_path):        
        super().__init__(conf_path)
        self.beta = self.exp_conf["beta"]

class EvalConfig:
    def __init__(self, conf_path="configs/eval.hocon"):
        self.eval_conf = ConfigFactory.parse_file(conf_path)
        self.plot = self.eval_conf['plot']
        self.num_epochs = self.eval_conf['num_epochs']
        self.lr = self.eval_conf['lr']
        self.weight_decay = self.eval_conf['weight_decay']
        self.batch_size = self.eval_conf['batch_size']
        self.show_plots = self.eval_conf['show_plots']
