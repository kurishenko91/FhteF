from FhteF_models import *
from experiment import *
from experiment_fair import *

def main():
    res=[]
    res_f=[]
    for i,md in enumerate([2,3,4,5,10,50]):
        for j,t in enumerate([500,1000]):
            for dataset_flag in [1,2,3]:
                res.append(experiment(dataset_flag,max_depth=md,T=t))
                res_f.append(experiment_fair(dataset_flag,max_depth=md,T=t,alpha_list=[2**i for i in range(-2,2)]))
    return res, res_f
    
if __name__== "__main__":
    res, res_f=main()
  
