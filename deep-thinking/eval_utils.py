import os
import random
import numpy as np
import torch

import pandas as pd 
import wandb
import numpy as np
from tqdm import tqdm

import sys
import traceback

from functools import partial

## eval tools
## PUT YOUR WANDB PROJECT HERE
entity='entity'
project='neuralsolver'

def set_wandb_project(project_arg):
    global project
    project = project_arg

def fix_run(run_id):
    online_run =  wandb.init(id=run_id, entity=entity, project=project, resume="must")
    online_run.finish(exit_code=0)

def fix_run_tag(tag):
    api = wandb.Api()
    runs = list(api.runs(path=f"{entity}/{project}", filters={
        # "state":"finished",
        "tags":tag,
    }))
    for run in tqdm(runs):
        fix_run(run.id)

def do_eval(runs, eval_fn, check_eval_name='', force_eval=False, keep_previous_exit_code=True,use_online_run=True)->list:
    assert isinstance(runs, list) and len(runs) > 0, "runs must be a list of runs"
    assert  isinstance(runs[0], wandb.apis.public.Run), "runs must be a list of runs"
    assert callable(eval_fn), "eval_fn must be a callable function"

    # if check_eval_name:
    #     raise NotImplementedError("check_eval_name not implemented yet")

    error_runs = []

    for run in tqdm(runs):

        if not force_eval and check_eval_name and check_eval_name in run.summary.keys() and run.summary[check_eval_name]:
            print(f"skipping {run.name}")
            continue

        previous_exit_code = int(run.state!='finished') # 0 if finished, 1 if failed/crashed
        dit_it_fail = 0
        # print(f"evaluating {run.name}")
        if use_online_run:
            online_run =  wandb.init(id=run.id, entity=entity, project=project,group=run.group, resume="must")
        else:
            online_run = run
        try:
            eval_fn(online_run)

            # set flag eval_name 
            if check_eval_name:
                online_run.summary[check_eval_name] = True

        except KeyboardInterrupt:
            print(f"KeyboardInterrupt evaluating {run.name}, STOPPING")
            if use_online_run:
                online_run.finish(exit_code=previous_exit_code)

            sys.exit(0)
            

        except Exception as e:
            # print exception traceback
            traceback.print_exc()

            print(f"Error evaluating {run.name}: {e}")

            error_runs.append(run)
            dit_it_fail = 1
            continue


        finally:
            if use_online_run:
                if keep_previous_exit_code:
                    online_run.finish(exit_code=previous_exit_code)
                else:
                    online_run.finish(exit_code=dit_it_fail)
            else:
                run.update()
                

    if len(error_runs) > 0:
        print(f"Error evaluating {len(error_runs)} runs")
        print(f"Error runs: {[r.name for r in error_runs]}")

    return error_runs 



def do_eval_online(online_run,eval_fn, check_eval_name='', force_eval=False, keep_previous_exit_code=True,use_online_run=True)->list:
    # assert isinstance(runs, list) and len(runs) > 0, "runs must be a list of runs"
    # assert  isinstance(runs[0], wandb.apis.public.Run), "runs must be a list of runs"
    assert callable(eval_fn), "eval_fn must be a callable function"

    # if check_eval_name:

    # if use_online_run:
    #     online_run = wandb.run
    if not use_online_run:
        raise NotImplementedError("do_eval_online not implemented for offline runs")


    try:
        eval_fn(online_run)

        # set flag eval_name 
        if check_eval_name:
            online_run.summary[check_eval_name] = True

    except KeyboardInterrupt:
        print(f"KeyboardInterrupt evaluating {online_run.name}, STOPPING")
        sys.exit(0)
        

    except Exception as e:
        # print exception traceback
        traceback.print_exc()

        print(f"Error evaluating {online_run.name}: {e}")


##### wandb save utils

def wandb_save_plot(run, plot_name, x_values, y_values):
    # wandb.init(id=run.id, entity=entity, project=project, resume="must")
    
    data = [[x, y] for (x, y) in zip(x_values, y_values)]

    print(data)

    table = wandb.Table(data=data, columns = ["x", "y"])
    wandb.log({plot_name + '_plot' : wandb.plot.line(table, "x", "y", title=plot_name+'_plot')})
    # wandb.finish()


    ## why am I saving this? as a summary? I think we might have problems with wandb.
    # OKAY it was essential for future plotting...
    # run.summary[plot_name+'_x'] = list(x_values)
    # run.summary[plot_name+'_y'] = list(y_values)




### gpu utils
import subprocess

def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

##RNL.....
def get_freer_gpu():
    out=subprocess.getoutput('nvidia-smi -q -d Memory |grep -A6 GPU|grep Free')
    memory_available = [int(x.split()[2]) for x in out.strip().split("\n")]


    if len(memory_available) != torch.cuda.device_count():
        print(f"Error, number of available GPUs ({len(memory_available)}) different from number of devices ({torch.cuda.device_count()})")
        print(f"Returning 0")
        return 0
    else:
        return int(np.argmax(memory_available))

def get_freer_gpu_device():
    free_gpu_id = get_freer_gpu()
    return torch.device(f'cuda:{free_gpu_id}')

def set_freer_gpu():
    try:
        free_gpu_id = get_freer_gpu()
        torch.cuda.set_device(free_gpu_id)
    except:
        print("Error trying to set GPU device, using default")
        pass