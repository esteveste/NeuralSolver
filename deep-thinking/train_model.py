import json
import logging
import os
import sys
from collections import OrderedDict

import hydra
import numpy as np
import torch
from icecream import ic
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

import deepthinking as dt
import deepthinking.utils.logging_utils as lg

import wandb

from tqdm import tqdm

torch.set_num_threads(1)

print(f"Available cuda devices: {torch.cuda.device_count()}")

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115

global_prob_name=None

@hydra.main(config_path="config", config_name="train_model_config")
def main(cfg: DictConfig):
    global global_prob_name

    # uncomment to use wandb
    # wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity,
    #            group=cfg.wandb_group, name=cfg.wandb_name, config=dict(cfg),
    #            sync_tensorboard=True, )

    # remove this line to use wandb
    wandb.init(mode="disabled")

    device = "cuda" if torch.cuda.is_available() else "cpu"


    assert torch.cuda.is_available(), "CUDA not available, exiting."
    torch.backends.cudnn.benchmark = True
    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info("train_model.py main() running.")
    log.info(OmegaConf.to_yaml(cfg))

    cfg.problem.model.test_iterations = list(range(cfg.problem.model.test_iterations["low"],
                                                   cfg.problem.model.test_iterations["high"] + 1))
    assert 0 <= cfg.problem.hyp.alpha <= 1, "Weighting for loss (alpha) not in [0, 1], exiting."
    writer = SummaryWriter(log_dir=f"tensorboard-{cfg.problem.model.model}-{cfg.problem.hyp.alpha}")

    ####################################################
    #               Dataset and Network and Optimizer
    loaders = dt.utils.get_dataloaders(cfg.problem)

    if cfg.problem.hyp.use_multiple_eval_test:
        loaders_eval_test = dt.utils.get_multiple_test_dataloaders(cfg.problem)

    net, start_epoch, optimizer_state_dict = dt.utils.load_model_from_checkpoint(cfg.problem.name,
                                                                                 cfg.problem.model,
                                                                                 device,
                                                                                 use_data_parallel=cfg.problem.hyp.use_data_parallel,
                                                                                 use_compile=cfg.problem.hyp.use_compiler)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    log.info(f"This {cfg.problem.model.model} has {pytorch_total_params/1e6:0.3f} million parameters.")
    log.info(f"Training will start at epoch {start_epoch}.")
    optimizer, warmup_scheduler, lr_scheduler, earlystopping = dt.utils.get_optimizer(cfg.problem.hyp,
                                                                       cfg.problem.model,
                                                                       net,
                                                                       optimizer_state_dict)
    
    ### data structure for training setup, list
    train_setup = dt.TrainingSetup(optimizer=optimizer,
                                   scheduler=lr_scheduler,
                                   warmup=warmup_scheduler,
                                   clip=cfg.problem.hyp.clip,
                                   alpha=cfg.problem.hyp.alpha,
                                   max_iters=cfg.problem.model.max_iters,
                                   problem=cfg.problem.name,

                                   custom_beta=cfg.problem.hyp.custom_beta,                                   
                                   noise_l1_alpha=cfg.problem.hyp.noise_l1_alpha,
                                   noise_l1_only_final_and_next=cfg.problem.hyp.noise_l1_only_final_and_next,
                                   min_iters=cfg.problem.hyp.min_iters,
                                   )
    ####################################################

    ####################################################
    #        Train
    log.info(f"==> Starting training for {max(cfg.problem.hyp.epochs - start_epoch, 0)} epochs...")
    highest_train_acc_so_far = -1
    highest_val_acc_so_far = -1
    highest_val_acc_so_far_overthinking = -1
    highest_val_acc_so_far_window = -1
    best_so_far = False
    best_so_far_overthinking = False
    best_so_far_or_equal = False
    best_so_far_window = False

    train_acc,val_acc,test_acc=[],[],[]

    train_losses_list, val_accs_list,vall_accs_OT_list =[],[],[]

    last_test_list = []

    for epoch in tqdm(range(start_epoch, cfg.problem.hyp.epochs)):


        loss, acc = dt.train(net, loaders, cfg.problem.hyp.train_mode, train_setup, device)

        # save all accs until OT iters
        val_acc_dict = dt.test(net, [loaders["val"]], cfg.problem.hyp.test_mode, list(range(1,cfg.problem.hyp.ot_val_iterations+1)),
                          cfg.problem.name, device)[0]
        
        val_acc=val_acc_dict[cfg.problem.model.max_iters]
        val_acc_overthinking=val_acc_dict[cfg.problem.hyp.ot_val_iterations]
        val_acc_window = np.mean(list(val_acc_dict.values()))

        train_losses_list.append(acc)
        val_accs_list.append(val_acc)
        vall_accs_OT_list.append(val_acc_overthinking)

        ## do schedulers here...
        if cfg.problem.hyp.scheduler_check_value =='train':
            scheduler_check = acc
        elif cfg.problem.hyp.scheduler_check_value =='val':
            scheduler_check = val_acc
        elif cfg.problem.hyp.scheduler_check_value =='val_ot':
            scheduler_check = val_acc_overthinking
        elif cfg.problem.hyp.scheduler_check_value =='loss':
            scheduler_check = loss

        elif cfg.problem.hyp.scheduler_check_value =='zero':
            scheduler_check = 0

        else:
            raise Exception(f"cfg.problem.hyp.scheduler_check_value {cfg.problem.hyp.scheduler_check_value} not implemented")


        if cfg.problem.hyp.lr_decay=='reduce_on_plateau':
            lr_scheduler.step(scheduler_check)
        else:
            lr_scheduler.step()

        if cfg.problem.hyp.use_early_stopping:
            earlystopping.step(scheduler_check)
            writer.add_scalar("earlystopping_nr_bad_epochs", earlystopping.num_bad_epochs, epoch)

        warmup_scheduler.dampen()
        
        if acc > highest_train_acc_so_far:
            highest_train_acc_so_far=acc

        if val_acc > highest_val_acc_so_far:
            best_so_far = True
            highest_val_acc_so_far = val_acc

        if val_acc >= highest_val_acc_so_far:
            best_so_far_or_equal=True

        if val_acc_overthinking >= highest_val_acc_so_far_overthinking:
            best_so_far_overthinking=True
            highest_val_acc_so_far_overthinking = val_acc_overthinking

        if val_acc_window >= highest_val_acc_so_far_window:
            best_so_far_window=True
            highest_val_acc_so_far_window = val_acc_window

        log.info(f"Training loss at epoch {epoch}: {loss}")
        log.info(f"Training accuracy at epoch {epoch}: {acc}")
        log.info(f"Val accuracy at epoch {epoch}: {val_acc}")
        log.info(f"Val accuracy overthinking at epoch {epoch}: {val_acc_overthinking}")
        log.info(f"Val accuracy window at epoch {epoch}: {val_acc_window}")

        # if the loss is nan, then stop the training
        if np.isnan(float(loss)):
            raise ValueError(f"{ic.format()} Loss is nan, exiting...")

        # TensorBoard loss writing
        writer.add_scalar("Loss/loss", loss, epoch)
        writer.add_scalar("Accuracy/acc", acc, epoch)
        writer.add_scalar("Accuracy/val_acc", val_acc, epoch)
        writer.add_scalar("Accuracy/val_acc_overthinking", val_acc_overthinking, epoch)
        writer.add_scalar("Accuracy/val_acc_window", val_acc_window, epoch)

        for i in range(len(optimizer.param_groups)):
            writer.add_scalar(f"Learning_rate/group{i}",
                              optimizer.param_groups[i]["lr"],
                              epoch)

        # evaluate the model periodically and at the final epoch
        if (epoch + 1) % cfg.problem.hyp.val_period == 0 or (epoch + 1 == cfg.problem.hyp.epochs and not cfg.problem.hyp.single_batch):
            
            if cfg.problem.hyp.use_multiple_eval_test:
                ## multiple test

                ## make ordered dict
                loaders_eval_test = OrderedDict(loaders_eval_test)

                test_accs = dt.test(net,loaders_eval_test.values(),
                                                    cfg.problem.hyp.test_mode,
                                                    cfg.problem.model.test_iterations,
                                                    cfg.problem.name,
                                                    device)
                
                tb_last = cfg.problem.model.test_iterations[-1]

                metric_list=[]
                name_list=[]
                for i, size in enumerate(loaders_eval_test.keys()):
                    metric_list.append(test_accs[i][tb_last])
                    name_list.append(f"multi_eval/acc_{size}")

                    metric_list.append(max(test_accs[i].values()))
                    name_list.append(f"multi_eval/max_acc_{size}")
                

                last_test_list.append(test_accs[i][tb_last])
                lg.write_to_tb(metric_list, name_list, epoch, writer)

                # log info best values
                log.info(f"Testing sizes: {list(loaders_eval_test.keys())}")
                log.info(f"Testing accuracy max: {[max(test_acc.values()) for test_acc in test_accs]}")

                test_acc=test_accs
                train_acc,val_acc=[],[]

            else:
                ## default single test
            
                test_acc, val_acc, train_acc = dt.test(net,
                                                    [loaders["test"],
                                                        loaders["val"],
                                                        loaders["train"]],
                                                    cfg.problem.hyp.test_mode,
                                                    cfg.problem.model.test_iterations,
                                                    cfg.problem.name,
                                                    device)
                log.info(f"Training accuracy: {train_acc}")
                log.info(f"Val accuracy: {val_acc}")
                log.info(f"Test accuracy (hard data): {test_acc}")


                tb_last = cfg.problem.model.test_iterations[-1] ##always saves last
                
                last_test_list.append(test_acc[tb_last])
                
                lg.write_to_tb([train_acc[tb_last], val_acc[tb_last], test_acc[tb_last], max(train_acc.values()), max(val_acc.values()), max(test_acc.values())],
                            ["train_acc", "val_acc", "test_acc", "max_train_acc","max_val_acc","max_test_acc"],
                            epoch,
                            writer)
            


        # check to see if we should save
        save_now = (epoch + 1) % cfg.problem.hyp.save_period == 0 or \
                   (epoch + 1) == cfg.problem.hyp.epochs or best_so_far
        if save_now and cfg.save_model:
            state = {"net": net.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}

            if best_so_far:
                out_str = f"model_best.pth"
            else:
                out_str = f"model_.pth"
            best_so_far = False
            log.info(f"Saving model to: {out_str}")
            torch.save(state, out_str)
            # wandb.save(out_str)

        
        if best_so_far_or_equal and cfg.save_model:
            state = {"net": net.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}

            out_str = f"model_best_val2.pth"

            best_so_far_or_equal = False
            log.info(f"Saving model to: {out_str}")
            torch.save(state, out_str)
            # wandb.save(out_str)


        if best_so_far_overthinking and cfg.save_model:
            state = {"net": net.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}

            out_str = f"model_best_overthinking.pth"

            best_so_far_overthinking = False
            log.info(f"Saving model to: {out_str}")
            torch.save(state, out_str)
            # wandb.save(out_str)


        if best_so_far_window and cfg.save_model:
            state = {"net": net.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}

            out_str = f"model_best_window.pth"

            best_so_far_window = False
            log.info(f"Saving model to: {out_str}")
            torch.save(state, out_str)

        if earlystopping.stop:
            print("End of Training because of early stopping at epoch {}".format(epoch))
            break

    
    writer.flush()
    writer.close()

    # save some accuracy stats (can be used without testing to discern which models trained)
    stats = OrderedDict([("max_iters", cfg.problem.model.max_iters),
                         ("run_id", cfg.run_id),
                         ("test_acc", test_acc),
                         ("test_data", cfg.problem.test_data),
                         ("test_iters", list(cfg.problem.model.test_iterations)),
                         ("test_mode", cfg.problem.hyp.test_mode),
                         ("train_data", cfg.problem.train_data),
                         ("train_acc", train_acc),
                         ("val_acc", val_acc)])
    with open(os.path.join("stats.json"), "w") as fp:
        json.dump(stats, fp)
    log.info(stats)
    ####################################################

    global_prob_name = cfg.problem.name

    run = wandb.run
    run.summary["train_check_acc"] = highest_train_acc_so_far==100
    run.summary["train_check_acc90"] = highest_train_acc_so_far>=90
    run.summary["train_check_acc99"] = highest_train_acc_so_far>=99
    run.summary["train_check_val_acc"] =highest_val_acc_so_far==100
    run.summary["train_check_val_acc90"] =highest_val_acc_so_far>=90
    run.summary["train_check_val_acc99"] =highest_val_acc_so_far>=99
    run.summary["train_check_val_acc_OT"] =highest_val_acc_so_far_overthinking==100
    run.summary["train_check_val_acc90_OT"] =highest_val_acc_so_far_overthinking>=90
    run.summary["train_check_val_acc99_OT"] =highest_val_acc_so_far_overthinking>=99
    run.summary["train_check_val_acc_window"] =highest_val_acc_so_far_window==100
    run.summary["train_check_val_acc90_window"] =highest_val_acc_so_far_window>=90
    run.summary["train_check_val_acc99_window"] =highest_val_acc_so_far_window>=99
    run.summary["evals/train_check_list_evalV2"] = True

    ## area under loss curve
    run.summary["train_loss_mean_area"] = np.mean(train_losses_list)
    run.summary["val_acc_mean_area"] = np.mean(val_accs_list)
    run.summary["val_acc_mean_area_OT"] = np.mean(vall_accs_OT_list) 

    run.summary["test_acc_mean_area"] = np.mean(last_test_list)


    # final save
    log.info(f"Saving model to: model_.pth")
    state = {"net": net.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}

    out_str = f"model_.pth"
    log.info(f"Saving model to: {out_str}")
    torch.save(state, out_str)


    if cfg.problem.hyp.single_batch or cfg.problem.hyp.no_final_eval:
        ## dont do final testing
        wandb.finish()
        sys.exit(0)


def eval(name,run_id):
    import os

    os.chdir("./deep-thinking/")


    # from eval_utils import *
    from eval_definition import do_eval_online,get_wandb_plot_values,partial,eval_supervised_and_fixed_point, eval_supervised, eval_env

    run = wandb.run

    # print(run.summary.keys())
    # print("history: ",hasattr(run,"history"))
    # print(name)

    ### evaluations
    if 'maze' in name:
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="13",max_iters=500), check_eval_name='evals/40_std_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="13",max_iters=500,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/13_OT_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="13",max_iters=500,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/13_last_f', force_eval=False)


        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="22",max_iters=500), check_eval_name='evals/22_std_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="22",max_iters=500,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/22_OT_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="22",max_iters=500,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/22_last_f', force_eval=False)


        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="33",max_iters=500), check_eval_name='evals/33_std_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="33",max_iters=500,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/33_OT_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="33",max_iters=500,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/33_last_f', force_eval=False)


        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="59",max_iters=1000), check_eval_name='evals/59_std_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="59",max_iters=1000,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/59_OT_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="59",max_iters=1000,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/59_last_f', force_eval=False)



        ## simplify
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="13",max_iters=500), check_eval_name='evals/40_std_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="13",max_iters=500,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/13_last_f', force_eval=False)


        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="22",max_iters=500), check_eval_name='evals/22_std_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="22",max_iters=500,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/22_OT_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="22",max_iters=500,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/22_last_f', force_eval=False)


        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="33",max_iters=500), check_eval_name='evals/33_std_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="33",max_iters=500,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/33_last_f', force_eval=False)

        # WAS ON
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="13",max_iters=500,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/13_OT_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="33",max_iters=500,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/33_OT_f', force_eval=False)

        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="59",max_iters=1000), check_eval_name='evals/59_std_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="59",max_iters=1000,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/59_last_f', force_eval=False)
        
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="59",max_iters=1000,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/59_OT_f', force_eval=False)

        if '1s' in name:
            # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="65",max_iters=1000,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/65_OT_f', force_eval=False)

            # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="95",max_iters=2000,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/95_OT_f', force_eval=False)



            do_eval_online(run,partial(eval_supervised,run_id=run_id,size="121",max_iters=2000,), check_eval_name='evals/121_std_f', force_eval=False)
            # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="121",max_iters=2000,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/121_OT_f', force_eval=False)
            # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="121",max_iters=2000,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/121_last_f', force_eval=False)
            do_eval_online(run,partial(eval_supervised,run_id=run_id,size="121",max_iters=2000,plot_name='evals/{}_size_test_best_eq',file_name='model_best_val2'), check_eval_name='evals/121_best_eq_f', force_eval=False)
            # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="121",max_iters=2000,plot_name='evals/{}_size_test_window',file_name='model_best_window'), check_eval_name='evals/121_window_f', force_eval=False)


        else:
            do_eval_online(run,partial(eval_supervised,run_id=run_id,size="59",max_iters=1000), check_eval_name='evals/59_std_f', force_eval=False)
            # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="59",max_iters=1000,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/59_last_f', force_eval=False)
            do_eval_online(run,partial(eval_supervised,run_id=run_id,size="59",max_iters=1000,plot_name='evals/{}_size_test_best_eq',file_name='model_best_val2'), check_eval_name='evals/59_best_eq_f', force_eval=False)
            # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="59",max_iters=1000,plot_name='evals/{}_size_test_window',file_name='model_best_window'), check_eval_name='evals/59_window_f', force_eval=False)



        # if 'half' in name: 
        #     # we probably want to evaluate on 5000
        #     do_eval_online(run, partial(eval_supervised,run_id=run_id,size="59",max_iters=5000,plot_name='evals/{}_size_test_OT_5x',file_name='model_best_overthinking'), check_eval_name='evals/59_OT_f5x', force_eval=False)
        #     do_eval_online(run, partial(eval_supervised,run_id=run_id,size="59",max_iters=5000,plot_name='evals/{}_size_test_5x'), check_eval_name='evals/59_std_f5x', force_eval=False)

    elif 'chess' in name:

        do_eval_online(run,partial(eval_supervised,run_id=run_id,size=700_000,max_iters=200,plot_name='evals/{}_size_test',folder_path='../../../outputs/chess_ablation'), check_eval_name='evals/chess_700k', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size=700_000,max_iters=200,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking',folder_path='../../../outputs/chess_ablation'), check_eval_name='evals/chess_700k_OT', force_eval=False)
        do_eval_online(run,partial(eval_supervised,run_id=run_id,size=700_000,max_iters=200,plot_name='evals/{}_size_test_best_eq',file_name='model_best_val2',folder_path='../../../outputs/chess_ablation'), check_eval_name='evals/chess_700k_best_eq', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size=700_000,max_iters=200,plot_name='evals/{}_size_test_window',file_name='model_best_window',folder_path='../../../outputs/chess_ablation'), check_eval_name='evals/chess_700k_window', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size=700_000,max_iters=200,plot_name='evals/{}_size_test_last',file_name='model_',folder_path='../../../outputs/chess_ablation'), check_eval_name='evals/chess_700k_last', force_eval=False)

    elif 'prefix_sums' in name:
        folder_path='../../../outputs/new_arches'
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,folder_path=folder_path,size="48",max_iters=200), check_eval_name='evals/40_std_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,folder_path=folder_path,size="48",max_iters=200,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/13_OT_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,folder_path=folder_path,size="48",max_iters=200,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/13_last_f', force_eval=False)

        # do_eval_online(run,partial(eval_supervised,run_id=run_id,folder_path=folder_path,size="128",max_iters=500), check_eval_name='evals/128_std_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,folder_path=folder_path,size="128",max_iters=500,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/128_OT_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,folder_path=folder_path,size="128",max_iters=500,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/128_last_f', force_eval=False)


        # do_eval_online(run,partial(eval_supervised,run_id=run_id,folder_path=folder_path,size="256",max_iters=500), check_eval_name='evals/256_std_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,folder_path=folder_path,size="256",max_iters=500,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/256_OT_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,folder_path=folder_path,size="256",max_iters=500,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/256_last_f', force_eval=False)

        do_eval_online(run,partial(eval_supervised,run_id=run_id,folder_path=folder_path,size="512",max_iters=500), check_eval_name='evals/59_std_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,folder_path=folder_path,size="512",max_iters=500,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/59_OT_f', force_eval=False)
        do_eval_online(run,partial(eval_supervised,run_id=run_id,folder_path=folder_path,size="512",max_iters=500,plot_name='evals/{}_size_test_best_eq',file_name='model_best_val2'), check_eval_name='evals/59_best_eq_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,folder_path=folder_path,size="512",max_iters=500,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/59_last_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,folder_path=folder_path,size="512",max_iters=500,plot_name='evals/{}_size_test_window',file_name='model_best_window'), check_eval_name='evals/59_window_f', force_eval=False)



    # elif 'line' in name:
    #     do_eval_online(run,partial(eval_supervised,run_id=run_id,size="256",max_iters=500), check_eval_name='evals/59_std_f', force_eval=False)
    #     do_eval_online(run,partial(eval_supervised,run_id=run_id,size="256",max_iters=500,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/59_OT_f', force_eval=False)

    #     do_eval_online(run, partial(eval_supervised,run_id=run_id,size="32",max_iters=200), check_eval_name='evals/40_std_f', force_eval=False)
    #     do_eval_online(run, partial(eval_supervised,run_id=run_id,size="32",max_iters=200,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/13_OT_f', force_eval=False)

    #     do_eval_online(run,partial(eval_supervised,run_id=run_id,size="256",max_iters=5000,plot_name='evals/{}_size_test_5000'), check_eval_name='evals/59_std_f5000', force_eval=False)
    #     do_eval_online(run,partial(eval_supervised,run_id=run_id,size="256",max_iters=5000,plot_name='evals/{}_size_test_OT_5000',file_name='model_best_overthinking'), check_eval_name='evals/59_OT_f5000', force_eval=False)
    #     do_eval_online(run,partial(eval_supervised,run_id=run_id,size="512",max_iters=5000,plot_name='evals/{}_size_test_OT_5000',file_name='model_best_overthinking'), check_eval_name='evals/512_OT_f5000', force_eval=False)


    elif 'pong' in name:
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="13",max_iters=100), check_eval_name='evals/40_std_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="13",max_iters=100,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/13_OT_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="13",max_iters=100,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/13_last_f', force_eval=False)


        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="23",max_iters=500), check_eval_name='evals/22_std_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="23",max_iters=500,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/22_OT_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="23",max_iters=500,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/22_last_f', force_eval=False)


        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="32",max_iters=300), check_eval_name='evals/33_std_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="32",max_iters=300,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/33_OT_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="32",max_iters=300,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/33_last_f', force_eval=False)


        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="64",max_iters=500), check_eval_name='evals/59_std_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="64",max_iters=500,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/59_OT_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="64",max_iters=500,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/59_last_f', force_eval=False)

        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="128",max_iters=500,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/128_OT_f', force_eval=False)
        do_eval_online(run,partial(eval_supervised,run_id=run_id,size="128",max_iters=500,plot_name='evals/{}_size_test_best_eq',file_name='model_best_val2'), check_eval_name='evals/128_best_eq_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="128",max_iters=500,plot_name='evals/{}_size_test_window',file_name='model_best_window'), check_eval_name='evals/128_window_f', force_eval=False)

        do_eval_online(run,partial(eval_supervised,run_id=run_id,size="128",max_iters=500), check_eval_name='evals/128_std_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="128",max_iters=500,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/128_last_f', force_eval=False)

    elif 'goto' in name:
        ### 1 step minigrid
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="24",max_iters=300), check_eval_name='evals/24_std_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="24",max_iters=300,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/24_OT_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="24",max_iters=300,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/24_last_f', force_eval=False)


        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="32",max_iters=300,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/33_OT_f', force_eval=False)

        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="64",max_iters=500), check_eval_name='evals/59_std_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="64",max_iters=500,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/59_OT_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="64",max_iters=500,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/59_last_f', force_eval=False)

        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="100",max_iters=1000,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/59_OT_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="100",max_iters=1000,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/100_OT_f', force_eval=False)

        do_eval_online(run,partial(eval_supervised,run_id=run_id,size="128",max_iters=1000,plot_name='evals/{}_size_test_best_eq',file_name='model_best_val2'), check_eval_name='evals/128_best_f', force_eval=False)
        do_eval_online(run,partial(eval_supervised,run_id=run_id,size="128",max_iters=1000), check_eval_name='evals/128_std_f', force_eval=False)

        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="128",max_iters=1000,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/128_OT_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="128",max_iters=1000,plot_name='evals/{}_size_test_window',file_name='model_best_window'), check_eval_name='evals/128_window_f', force_eval=False)
        
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="128",max_iters=1000,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/128_last_f', force_eval=False)

    # elif 'minigrid_empty' in name or 'minigrid' in name or 'simple' in name:
    #     do_eval_online(run,partial(eval_supervised,run_id=run_id,size="(124, 124)",max_iters=500), check_eval_name='evals/59_std_f', force_eval=False)
    #     do_eval_online(run,partial(eval_supervised,run_id=run_id,size="(124, 124)",max_iters=500,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/59_OT_f', force_eval=False)

    #     do_eval_online(run,partial(eval_supervised,run_id=run_id,size="(32, 32)",max_iters=200), check_eval_name='evals/40_std_f', force_eval=False)
    #     do_eval_online(run,partial(eval_supervised,run_id=run_id,size="(32, 32)",max_iters=200,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/13_OT_f', force_eval=False)

    elif 'doorkey' in name:

        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="64",max_iters=500,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/59_OT_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="64",max_iters=500,plot_name='evals/{}_size_test_best_eq',file_name='model_best_val2'), check_eval_name='evals/64_best_f', force_eval=False)

        do_eval_online(run,partial(eval_supervised,run_id=run_id,size="128",max_iters=1000), check_eval_name='evals/128_std_f', force_eval=False)
        do_eval_online(run,partial(eval_supervised,run_id=run_id,size="128",max_iters=1000,plot_name='evals/{}_size_test_best_eq',file_name='model_best_val2'), check_eval_name='evals/128_best_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="128",max_iters=1000,plot_name='evals/{}_size_test_OT',file_name='model_best_overthinking'), check_eval_name='evals/128_OT_f', force_eval=False)
        # do_eval_online(run,partial(eval_supervised,run_id=run_id,size="128",max_iters=1000,plot_name='evals/{}_size_test_last',file_name='model_'), check_eval_name='evals/128_last_f', force_eval=False)

        # sequential evals
        # do_eval_online(run,partial(eval_env,size=5,max_iters=30,summary_name='evals/env_{}_test_v2'), check_eval_name='evals/5_f3', force_eval=False)
        # do_eval_online(run,partial(eval_env,size=8,max_iters=30,summary_name='evals/env_{}_test_v2'), check_eval_name='evals/8_f3', force_eval=False)
        # do_eval_online(run,partial(eval_env,size=20,max_iters=30,summary_name='evals/env_{}_test_v2'), check_eval_name='evals/20_f3', force_eval=False)
        # do_eval_online(run,partial(eval_env,size=32,max_iters=100,summary_name='evals/env_{}_test_v2'), check_eval_name='evals/32_f3', force_eval=False)
        # # do_eval_online(runs,partial(eval_env,size=64,max_iters=200,summary_name='evals/env_{}_test_v2'), check_eval_name='evals/64_200_f3', force_eval=False)
        # do_eval_online(run,partial(eval_env,size=64,max_iters=200,summary_name='evals/env_{}_test_v2'), check_eval_name='evals/64_200_f3', force_eval=False)


        do_eval_online(run,partial(eval_env,run_id=run_id,size=128,max_iters=400,summary_name='evals/env_{}_test_v2'), check_eval_name='evals/128_f3', force_eval=False)
        do_eval_online(run,partial(eval_env,run_id=run_id,size=128,max_iters=400,summary_name='evals/env_{}_test_best_eq_v2',file_name='model_best_val2'), check_eval_name='evals/128_best_eq_f3', force_eval=False)


    else:
        raise NotImplementedError

if __name__ == "__main__":
    run_id = dt.utils.generate_run_id()
    sys.argv.append(f"+run_id={run_id}")
    main()
    try:
        # this needs to be done after main() because of hydra
        eval(global_prob_name,run_id)
    except Exception as e:
        print("eval failed")
        print(e)
        raise e
    wandb.finish()
