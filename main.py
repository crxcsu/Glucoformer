import os
from exp.train_val import main
from utils.global_options import global_options
os .environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    config = global_options().parse()
    config.model_name = 'Glucoformer'
    config.dataset = 'OhioT1DM'
    main(config, main_path=os.getcwd())
    
   


   