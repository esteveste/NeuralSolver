import re
from matplotlib import pyplot as plt
from eval_utils import *

import json



def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


import collections
## lets do single level columns for now

def get_wandb_plot_values(run, plot_name):
    if plot_name+'_x' in run.summary.keys():
        x,y= run.summary[plot_name+'_x'], run.summary[plot_name+'_y']
    elif plot_name in run.summary_metrics.keys():
        hist = run.history() # very slow call
        x,y = None, list(hist[plot_name].values)
        
        assert len(y)>0, f"Plot {plot_name} not found in run {run.name}"

        ## remove final nans from y
        while np.isnan(y[-1]):
            y.pop()

        x = list(range(len(y)))

    else:
        raise ValueError(f"Plot {plot_name} not found in run {run.name}")

    return x,y


def _group_mean_agg(values):
    return np.nanmean(values)

def _group_ci_agg_str(values):
    return f"{np.nanmean(values):.2f} +- {np.nanstd(values):.2f}"

def _group_mean_sem_agg_str(values):
    return f"{np.nanmean(values):.2f} +- {np.nanstd(values)/np.sqrt(len(values)):.2f}"


## FIXME enumerate for group ci

def get_wandb_summary_df(runs, keys, 
                         group_by=lambda run: run.group,
                         column_group_by=lambda run,key: key,
                         group=True, group_agg='ci', value_agg=None,
                        #  filter_rows=lambda run: True,  #not sure if should implement
                         group_run_filter=lambda run: True,
                         number_sort=True,
                         allow_missing=False,
                         show_group_seeds_legend=False,
                        remove_none_from_group= True,
                         ):
    '''
    runs: list of wandb runs
    keys: list of keys to get from summary, or list[name,callable,default] with run (useful for configs)
    group: if true, group by group name
    group_agg: function to aggregate group values
    value_agg: function to aggregate individual value values (in the case of value being a list)

    group_run_filter: lambda function to filter runs before grouping, ex. lambda run: run.summary['acc']>0.95
    
    '''

    if group_agg == 'mean':
        group_agg = _group_mean_agg
    elif group_agg == 'ci':
        group_agg = _group_ci_agg_str
    elif group_agg == 'mean_sem':
        group_agg = _group_mean_sem_agg_str
    elif group_agg == 'max':
        group_agg = np.nanmax
    elif group_agg == 'min':
        group_agg = np.nanmin
    elif isinstance(group_agg, str):
        raise ValueError(f"Unknown group_agg {group_agg}")
    else:
        pass



    if group:
        ## if this happens we need to aggregate results
        # groups = set([run.group for run in runs])
        groups = set([group_by(run) for run in runs])
        
        # if show_group_seeds_legend:
        #     group_list= [group_by(run) for run in runs]
        #     ## count each time an item appears in the list (group_list) by looking at the set of unique items (groups)
        #     groups = [f"{group} ({group_list.count(group)})" for group in groups]


        table = {group:{} for group in groups}
        seed_table = {group:{} for group in groups}
        for group in groups:
            group_runs = [run for run in runs if group_by(run)==group]
            
            # FIXME
            # if column_group_by is not None:
            #     key = column_group_by(run,key)


            ## do group filtering
            group_runs = [run for run in group_runs if group_run_filter(run)]
            

            for key in keys:
                ## we should agregate here, not all values will be present!
                if isinstance(key, str):
                    group_values = [run.summary[key] for run in group_runs if key in run.summary or not allow_missing]
                
                elif isinstance(key, list) or isinstance(key, tuple):
                    def _get_value(run):
                        if key[0] in run.config:
                            return run.config[key[0]]
                        else:
                            try:
                                q = key[1](run)
                            except:
                                if len(key)>2:
                                    q = key[2]
                                else:
                                    q = None
                            return q

                    group_values = [_get_value(run) for run in group_runs]
                    key = key[0]

                else:
                    raise NotImplementedError(f"Not implemented for key {key}")
                group_seeds = len(group_values)

                if remove_none_from_group:
                    # remove None values
                    group_values = [v for v in group_values if v is not None]

                if group_seeds>0 and isinstance(group_values[0], str):
                    ## then is a string they should all be the same
                    
                    set_values = set(group_values)

                    assert len(set_values)==1, f"Values for key {key} are strings, and are not the same for group {group}: {set_values}"

                    group_values = group_values[0]

                else:

                    if value_agg is not None:
                        group_values = value_agg(group_values)

                    if group_agg is not None:
                        group_values = group_agg(group_values)



                table[group][key] = group_values
                seed_table[group][key] = group_seeds

    else:
        table = {run.name:{} for run in runs}
        seed_table = None
        for run in runs:
            for key in keys:

                # if isinstance(key, str):
                ### FIXME do if key is a callable here!

                if key not in run.summary and allow_missing:
                    continue

                run_value = run.summary[key]
                if value_agg is not None:
                    run_value = value_agg(run_value)

                if column_group_by is not None:
                    key = column_group_by(run,key)

                table[run.name][key] = run_value
    
    if number_sort:
        table = collections.OrderedDict(sorted(table.items(), key=lambda t: natural_keys(t[0])))
        if seed_table is not None:
            seed_table = collections.OrderedDict(sorted(seed_table.items(), key=lambda t: natural_keys(t[0])))
    else:
        table = collections.OrderedDict(sorted(table.items(), key=lambda t: t[0]))
        if seed_table is not None:
            seed_table = collections.OrderedDict(sorted(seed_table.items(), key=lambda t: t[0]))

    

    group_table = pd.DataFrame(table).T
    if seed_table is not None:

        seed_table_df =pd.DataFrame(seed_table).T

        if show_group_seeds_legend:

            # new version
            for row, row_seed in zip(group_table.iloc, seed_table_df.iloc):

                for column in group_table.columns:
                    # if not isinstance(row[column], str):
                    #     continue
                    row[column] = f"{row[column]} ({row_seed[column]})"




            # for row in seed_table_df.iloc:
            #     key = row.name
            #     seeds = row[0]

            #     group_table = group_table.rename(index={key: f"{key} ({seeds} seeds)" })
        
        group_table.seeds = seed_table_df

    return group_table

    
# solucao nao e' perfeita,
## visto q forca verificar se valores finais nao sao nan.... 

def get_wandb_plot(runs, plot_name, group=False,
                   
                   group_by=lambda run: run.group,

                   show_group_seeds_legend=True, sort_groups=True,
                   show_group_name=True, log_scale=False, show_ci=True,
                   number_sort=True, group_op='mean',
                   add_data_table=None,
                   set_title=True,set_legend=True,legend_ncol=1,
                   use_previous_plot=None,
                   api=None,use_artifact=False,
                   group_names_map = None,
                   max_ci=np.inf,
                   min_ci=-np.inf,):
    ## they need to have same sequence

    global debug
    if use_previous_plot is not None:
        fig, ax = use_previous_plot
    else:
        fig, ax = plt.subplots()

    if use_artifact:
        assert api is not None, "api must be provided for use_artifact"


    if group:
        groups = set([group_by(run) for run in runs])
        table = {}
        for group in groups:
            group_runs = [run for run in runs if group_by(run)==group]
            
            ## do mean std over values
            run_xs = []
            run_ys = []

            for run in group_runs:
                try:
                    if use_artifact:
                        x,y = get_x_y_table_wandb(api,run, plot_name)
                    else:
                        x,y = get_wandb_plot_values(run, plot_name)

                except Exception as e:
                    # print(f"Error getting plot {plot_name} from run {run.name}: {e}")
                    continue
                run_xs.append(np.array(x))
                run_ys.append(np.array(y))

            if len(run_xs) == 0:
                continue # skip

            assert np.all([np.all(run_xs[0]==run_x) for run_x in run_xs])
            x = run_xs[0]

            ## convert to np infinite or nan values
            ## we use nanmean to ignore nan values
            run_ys = np.array(run_ys).astype(np.float32)
            
            y_mean = np.nanmean(run_ys, axis=0)
            y_std = np.nanstd(run_ys, axis=0)
            y_max = np.nanmax(run_ys, axis=0)
            y_min = np.nanmin(run_ys, axis=0)


            if group_names_map is not None and group in group_names_map.keys():
                group = group_names_map[group]

            if group not in table.keys():
                table[group] = {}

            table[group]['x'] = x
            table[group]['y_mean'] = y_mean
            table[group]['y_std'] = y_std
            table[group]['y_max'] = y_max
            table[group]['y_min'] = y_min
            table[group]['seeds'] = len(run_xs)

        if add_data_table is not None:
            for k,v in add_data_table.items():
                table[k] = v


        if sort_groups:
            ## FIXME please check this key sorting, I think it only looks at numbers
            if number_sort:
                table = collections.OrderedDict(sorted(table.items(), key=lambda t: natural_keys(t[0])))
            else:
                table = collections.OrderedDict(sorted(table.items(), key=lambda t: t[0]))

        for group, values in table.items():
            if len(values) == 0:
                continue # skip
            
            if add_data_table is not None and group in add_data_table.keys():
                x,y_mean = values['x'], values['y']
                if 'y_std' in table[group].keys():
                    y_std = values['y_std']
                else:
                    y_std = np.zeros_like(y_mean)

                ax.plot(x,y_mean, label=group)
            else:
                ## normal group plot 
                
                x,y_mean,y_std,y_max,y_min = values['x'], values['y_mean'], values['y_std'], values['y_max'], values['y_min']
                seeds = values['seeds']
                if show_group_seeds_legend:
                    label = f"{group} ({seeds} seeds)"
                else:
                    label = group
                if group_op == 'max':
                    ax.plot(x,y_max, label=label)
                elif group_op == 'min':
                    ax.plot(x,y_min, label=label)
                elif group_op == 'mean':
                    ax.plot(x,y_mean, label=label)
                else:
                    raise NotImplementedError
            
            
            if show_ci:
                ax.fill_between(x, np.maximum(y_mean-y_std,min_ci), np.minimum(y_mean+y_std,max_ci), alpha=0.3)
            # ax.set_title(plot_name)
            # ax.legend()

    else:
        table = {}
        for run in runs:
            try:
                x,y = get_wandb_plot_values(run, plot_name)
            except:
                continue

            if show_group_name:
                name = f"{run.group} - {run.name}"
            else:
                name = run.name
            table[name] = {'x':x, 'y':y}

        if add_data_table is not None:
            for k,v in add_data_table.items():
                table[k] = v

        if sort_groups:
            table = collections.OrderedDict(sorted(table.items()))

        
        for name, values in table.items():
            x,y = values['x'], values['y']
            ax.plot(x,y, label=name)
            # ax.set_title(plot_name)
            # ax.legend()
    if set_title:
        ax.set_title(plot_name)
    if set_legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        fancybox=True, shadow=True, ncol=legend_ncol)

    if log_scale:
        ax.set_yscale('log')

    return fig,ax




def get_x_y_table_wandb(api,run, table_name):
    artifact_path = f"{entity}/{project}/run-{run.id}-{table_name.replace('/','')}:latest"

    # get the artifact
    # api = wandb.Api()
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download()
    artifact_dir


    table_path = artifact_dir+f"/{table_name}.table.json"


    # open json
    with open(table_path,"r") as json_file:
        data = json.load(json_file)


    XY = data['data']
    X,Y = list(zip(*XY))

    return X,Y


def get_groups(runs):
    return set([run.group for run in runs])



###############3 highlight
def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    #remove % and cast to float
#     data = data.replace(' +- ','', ).astype(float)
#     print(data)
    data=data.str.split().str[0].astype(float)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)
    

### FIXME amarelo quando toca na std superior
def highlight_max_std(data, color='limegreen', std_color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    std_attr = 'background-color: {}'.format(std_color)
    #remove % and cast to float
#     data = data.replace(' +- ','', ).astype(float)
#     print(data)


    # data_std=data.str.split().str[-1].astype(float)
    # data=data.str.split().str[0].astype(float)

    # new version, always try to remove seeds first
    no_seed_data = data.str.split('(').str[0].str.split()
    data_std=no_seed_data.str[-1].astype(float)
    data=no_seed_data.str[0].astype(float)


    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()

        max_var = data_std[is_max].max() # get max variation
        is_in_std = (data + data_std)>=(data.max()-max_var)
        
#         return [attr if v else '' for v in is_max]

        attrs_list = []
        for is_v_max, is_v_std in zip(is_max,is_in_std):
            if is_v_max:
                attrs_list.append(attr)
            elif is_v_std:
                attrs_list.append(std_attr)
            else:
                attrs_list.append('')
        return attrs_list

    else:  # from .apply(axis=None)
        raise Exception('I dindnt do this')
#             is_max = data == data.max().max()
#             return pd.DataFrame(np.where(is_max, attr, ''),
#                                 index=data.index, columns=data.columns)


def highlight_min_std(data, color='limegreen', std_color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    std_attr = 'background-color: {}'.format(std_color)
    #remove % and cast to float
#     data = data.replace(' +- ','', ).astype(float)
#     print(data)
    # data_std=data.str.split().str[-1].astype(float)
    # data=data.str.split().str[0].astype(float)


    # new version, always try to remove seeds first
    no_seed_data = data.str.split('(').str[0].str.split()
    data_std=no_seed_data.str[-1].astype(float)
    data=no_seed_data.str[0].astype(float)

    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_min = data == data.min()

        max_var = data_std[is_min].max() # get max variation
        is_in_std = (data - data_std)<=(data.min()+max_var)
        
#         return [attr if v else '' for v in is_max]

        attrs_list = []
        for is_v_max, is_v_std in zip(is_min,is_in_std):
            if is_v_max:
                attrs_list.append(attr)
            elif is_v_std:
                attrs_list.append(std_attr)
            else:
                attrs_list.append('')
        return attrs_list

    else:  # from .apply(axis=None)
        raise Exception('I dindnt do this')
#             is_max = data == data.max().max()
#             return pd.DataFrame(np.where(is_max, attr, ''),
#                                 index=data.index, columns=data.columns)


### FIXME amarelo quando toca na std superior
def highlight_more_than(data, color='limegreen', std_color='yellow',value=99):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    std_attr = 'background-color: {}'.format(std_color)
    #remove % and cast to float
#     data = data.replace(' +- ','', ).astype(float)
#     print(data)
    data_std=data.str.split().str[-1].astype(float)
    data=data.str.split().str[0].astype(float)

    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data >= value

        is_in_std = (data + data_std)>=value
        
#         return [attr if v else '' for v in is_max]

        attrs_list = []
        for is_v_max, is_v_std in zip(is_max,is_in_std):
            if is_v_max:
                attrs_list.append(attr)
            elif is_v_std:
                attrs_list.append(std_attr)
            else:
                attrs_list.append('')
        return attrs_list

    else:  # from .apply(axis=None)
        raise Exception('I dindnt do this')

def count_wins(d):
    d["win_count"] = [0]*len(d)
    for k,v in d.items():
#         print(k,v.str)
        if hasattr(v,"str"):
            data=v.str.split().str[0].astype(float)
            place = data.argmax()
            d["win_count"][place]+=1
    
    return d

from functools import partial
highlight_max_std_dark = partial(highlight_max_std, color='red', std_color='darkblue')
highlight_more_than_dark = partial(highlight_more_than, color='red', std_color='darkblue')


import ast
def safe_eval(node_or_string):
    return ast.literal_eval(node_or_string)


### train checklist

from ast import literal_eval

def get_config_problem(run):
    # return run.config['_content']

    problem = run.config['_content']['problem']
    return literal_eval(problem)