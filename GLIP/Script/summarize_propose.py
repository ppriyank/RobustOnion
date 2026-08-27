import sys 
import os 
from collections import defaultdict
import pandas as pd 
import ast

folder = sys.argv[1]
# folder="ucf_output/proposed/lrtko"
print("\n\n\n")
MAX_COUNT = 14 
separator = "OrderedDict([('AP', "
acc_dict = defaultdict(list)
acc_bbdfk = defaultdict(list)

ENABLE_PROPOSED = False 
ENABLE_PREV=  False 
ENABLE_TOY=  False 
ENABLE_PROPOSED2=  False 
ENABLE_Cross_Exchange=  True  


print(" ** Summarize ** ")

new_order = ['model_name', 'OG', 'pixel_dropout', 'iso_blur', 'salt_pepper', 'low-res', 'low-res2', 'focus_blur',
    'jpg_compression', 'chromatic', 'motion_blur2', 'fog', 'rain_model2', 'snow_model2',
    'atmospheric', 'BBDK', 'DAWN', 'WEDGE']
toy_noise = ['model_name', 'OG', 'pixel_dropout', 'iso_blur', 'salt_pepper', 'low-res', 'low-res2', 'focus_blur',
    'jpg_compression', 'chromatic', 'motion_blur2', 'fog', 'rain_model2', 'snow_model2',
    'atmospheric']


def process_file(file, acc_dict):
    name_found = 0 
    counter = 0 
    encountered_in_this_set = set()
    with open(file, "r") as f:
        Lines = f.readlines()
        for line in Lines:
            if "####" in line:
                # print("----", line)
                assert name_found == 0 , f"Results missing {line} :: {name}"
                name = line.split("#")[-1].strip()
                if name not in toy_noise:continue
                name_found = 1
            elif "=====" in line and name_found == 1:
                assert name not in encountered_in_this_set, f"{name} appeareing twice :O"
                encountered_in_this_set.add(name)
                counter += 1
                name_found = 0 
                splits = line.split(separator)
                acc = splits[1].split("),")[0]
                acc = float(acc) *  100
                # print(f"{name:<30}\t{acc:.3f}")
                acc_dict[name].append(acc)
    
    # assert counter == MAX_COUNT or counter == MAX_COUNT - 1, f"Not all results present in {file} {encountered_in_this_set}"
    for col in toy_noise:
        if col not in encountered_in_this_set and col != 'model_name':
            print(col)
            acc_dict[col].append(-1)



def process_external(file, acc_bbdfk):
    found = 0 
    
    curr_mode = 'wedge'
    with open(file, "r") as f:
        Lines = f.readlines()
        for i,line in enumerate(Lines):
            if '#### BBDK' in line:
                curr_mode = 'bbdk'
                prefix = 'bbdk'
                continue 
            elif '#### DAWN' in line :
                curr_mode = 'dawn'
                prefix = 'dawn'
                continue 
            elif '#### WEDGE' in line :
                curr_mode = 'wedge'
                prefix = 'wedge'
                continue 
            elif '#### WEDGE' in line :
                curr_mode = 'wedge'
                prefix = 'wedge'
                continue 
            elif '#### FoggyCitiscape' in line :
                curr_mode = 'foggycity'
                prefix = 'FoggyCity'
                adjective = line.split("(")[-1].split(')')[0]
                fog_level = line.split(":")[-1].strip()
                prefix = f"{prefix}-{adjective}:{fog_level}"
                
            elif '#### Virtual Kitti' in line :
                curr_mode = 'virtualkitti'
                prefix = 'virtualkitti'
                
            
            ########## BBDK 
            if curr_mode == 'bbdk' and ":::::: Results " in line :
                found = 1
                i +=1 
                line = Lines[i]    
                while '##'  not in line and i < len(Lines)-1 :
                    splits = line.replace("\t", ":").split(":")
                    splits = [e.strip() for e in splits if e!= '']
                    if len(splits) != 1:
                        assert splits[3] == 'AP50', "AP50 only reported"
                        acc = float(splits[4]) *  100    
                        
                        if "WEATHER" in line:
                            condition = line.split('WEATHER-')[1].split(":")[0].strip()
                            acc_bbdfk[prefix + "-weather_" + condition].append(acc)
                        if "TIME" in line:
                            condition = line.split('TIME-')[1].split(":")[0].strip()
                            acc_bbdfk[prefix + "-time_" + condition].append(acc)
                        if 'Overall-' in line :
                            acc_bbdfk[prefix + "-overall"].append(acc)
                        # print(line, splits)
                        # print(condition, acc)
                    i +=1 
                    line = Lines[i]
                    continue 
                    
                curr_mode = None 
                
            # if ":::::: WEATHER" in line and curr_mode == 'bbdk':
            #     found = 1
            #     line = line.replace(":::::: WEATHER :", "")
            #     line = line.strip()
            #     line = line.replace("OrderedDict(", "")
            #     line = line.replace(")])", ")]")
                
            #     data_dict = ast.literal_eval(line)
            #     for weather in data_dict.keys():
            #         acc = data_dict[weather][1]
            #         assert acc[0] == 'AP50', "AP50 only reported"
            #         acc = float(acc[1]) *  100
            #         acc_bbdfk[prefix + "-weather_" + weather].append(acc)
            # if ":::::: TIME" in line and curr_mode == 'bbdk':
            #     found = 1
            #     line = line.replace(":::::: TIME    :  ", "")
            #     line = line.strip()
            #     line = line.replace("OrderedDict(", "")
            #     line = line.replace(")])", ")]")
                
            #     data_dict = ast.literal_eval(line)
            #     for time_label in data_dict.keys():
                    # acc = data_dict[time_label][1]
                    # assert acc[0] == 'AP50', "AP50 only reported"
                    # acc = float(acc[1]) *  100
                    # acc_bbdfk[prefix + "-time_" + time_label].append(acc)
            if ":::::: Results " in line and (curr_mode == 'dawn' or curr_mode == 'wedge' or curr_mode == 'virtualkitti'):
                found = 1
                i +=1 
                line = Lines[i]    
                while line != '\n' :
                    splits = line.split(" ")
                    splits = [e for e in splits if e!= '']
                    assert splits[2] == 'AP'
                    acc = float(splits[4].split("\t")[0]) * 100 
                    weather_name = splits[0]
                    acc_bbdfk[f"{prefix}-{weather_name}"].append(acc)
                    i +=1 
                    line = Lines[i]
                curr_mode = None 
            elif "====" in line and (curr_mode == 'foggycity'):
                splits = line.split(separator)
                acc = splits[1].split("),")[0]
                acc = float(acc) *  100
                acc_bbdfk[f"{prefix}"].append(acc)
                curr_mode = None 

    if found == 0 :
        for col in acc_bbdfk.keys():
            if col != 'model_name':
                acc_bbdfk[col].append(-1)


acc_bbdfk["model_name"].append("vanilla")
process_external('ucf_output/Benchmark/GLIP-T_5-EVAL.txt', acc_bbdfk)

if ENABLE_TOY:
    diff_allowed = 0 
    baseline_root = 'ucf_output/proposed/Baseline'
    for baseline_folder in sorted(os.listdir(baseline_root)):
        baseline_folder = os.path.join(baseline_root, baseline_folder)
        for file in sorted(os.listdir(baseline_folder)):
            if "EVAL" not in file:continue 
            # print(baseline_folder, file)
            potential_name = file
            file = os.path.join(baseline_folder, file)
            strings = f"{potential_name}"
            acc_bbdfk["model_name"].append(potential_name)
            process_external(file, acc_bbdfk)

            if len(set([len(acc_bbdfk[e]) for e in acc_bbdfk])) != diff_allowed + 1:
                diff_allowed += 1
                print([(e,len(acc_bbdfk[e])) for e in acc_bbdfk])
                print(f"Something is wrong with {file}" )
                continue 
                assert False, f"Something is wrong with {file}" 
                       
if ENABLE_PROPOSED:
    atleast_one_det = 0 
    for folder_plot in sorted(os.listdir(folder)):
        if folder_plot == "training_logs": continue 
        folder_plot = os.path.join(folder, folder_plot)
        for file in sorted(os.listdir(folder_plot)):
            if "EVAL" not in file:continue 
            potential_name = file
            print(f"\n {file} \n")    
            file = os.path.join(folder_plot, file)
            strings = f"{potential_name}"
            acc_dict["model_name"].append(potential_name)
            acc_bbdfk["model_name"].append(potential_name)
            process_file(file, acc_dict)
            process_external(file, acc_bbdfk)
            print([(e,len(acc_dict[e])) for e in acc_dict])
            print([(e,len(acc_bbdfk[e])) for e in acc_bbdfk])
            atleast_one_det = 1
        if atleast_one_det == 1:
            for key in acc_dict:
                acc_dict[key].append(-1)

if ENABLE_PREV:
    count = 1
    for file_name in sorted(os.listdir(folder)):
        file = os.path.join(folder, file_name)
        if "EVAL" not in file_name:continue 
        count += 1
        
        potential_name = file_name
        print(f"\n {file_name} \n")    
        
        strings = f"{potential_name}"
        acc_dict["model_name"].append(potential_name)
        acc_bbdfk["model_name"].append(potential_name)

        # process_file(file, acc_dict)
        # print([(e,len(acc_dict[e])) for e in acc_dict])
        process_external(file, acc_bbdfk)
        
        keys = [key for key in acc_bbdfk if "wedge" in key]
        for key in keys:
            del acc_bbdfk[key]

    

        problemetic_flag = 0 
        for key in acc_bbdfk:
            if len(acc_bbdfk[key]) == count:
                continue 
            elif len(acc_bbdfk[key]) == count -1:
                acc_bbdfk[key].append(-1)
            else:
                problemetic_flag = 1
                break 
        if problemetic_flag:
            print(f"Problemetic : {file_name}, with expected count {count}")
            print([(e,len(acc_bbdfk[e])) for e in acc_bbdfk])
            quit()

if ENABLE_PROPOSED2:
    print( [(key, len(acc_bbdfk[key])) for key in acc_bbdfk] )

    acc_bbdfk["model_name"].append("LR_TKO-low-res")
    process_external('ucf_output/proposed/Baseline/lrtko/GLIP-T_5-LR_TKO-low-res-EVAL.txt', acc_bbdfk)
    print( [(key, len(acc_bbdfk[key])) for key in acc_bbdfk] )

    acc_bbdfk["model_name"].append("LR_TKO-atmospheric")
    process_external('ucf_output/proposed/Baseline/lrtko/GLIP-T_5-LR_TKO-atmospheric-EVAL.txt', acc_bbdfk)
    print( [(key, len(acc_bbdfk[key])) for key in acc_bbdfk] )

    # pd.DataFrame.from_dict(acc_bbdfk)
    count = 3
    for file_name in sorted(os.listdir(folder)):
        file = os.path.join(folder, file_name)
        if "EVAL" not in file_name:continue 
        potential_name = file_name
        print(f"\n {file_name} \n")    
        
        strings = f"{potential_name}"
        acc_bbdfk["model_name"].append(potential_name)

        process_external(file, acc_bbdfk)
        # print( [(key, len(acc_bbdfk[key])) for key in acc_bbdfk] )
        count += 1 

        problemetic_flag = 0 
        for key in acc_bbdfk:
            if len(acc_bbdfk[key]) == count:
                continue 
            elif len(acc_bbdfk[key]) == count -1:
                acc_bbdfk[key].append(-1)
            else:
                problemetic_flag = 1
                break 
        if problemetic_flag:
            print(f"Problemetic : {file_name}, with expected count {count}")
            print([(e,len(acc_bbdfk[e])) for e in acc_bbdfk])
            quit()
            
if ENABLE_Cross_Exchange:
    process_file('ucf_output/Benchmark/GLIP-T_5-EVAL.txt', acc_bbdfk)

    count = 1
    problemetic_flag = 0 
    parent_folder= folder
    for root_folder in os.listdir(folder):
        root_folder = os.path.join(folder, root_folder)
        for file_name in sorted(os.listdir(root_folder)):
            file = os.path.join(root_folder, file_name)
            if "EVAL" not in file_name:continue 
            count += 1 
            potential_name = file_name.replace("-EVAL.txt", "").replace("GLIP-T_5-", "")
            potential_name = potential_name.replace("+low-res", "").replace("-low-res", "")
            print(f"\n {file_name} \n")    
            strings = f"{potential_name}"
            acc_bbdfk["model_name"].append(potential_name)

            process_external(file, acc_bbdfk)
            process_file(file, acc_bbdfk)
            # print( [(key, len(acc_bbdfk[key])) for key in acc_bbdfk] )

            for key in acc_bbdfk:
                if len(acc_bbdfk[key]) == count:
                    continue 
                elif len(acc_bbdfk[key]) == count -1:
                    acc_bbdfk[key].append(-1)
                else:
                    problemetic_flag = 1
                    break 

    print( [(key, len(acc_bbdfk[key])) for key in acc_bbdfk] )
    if problemetic_flag:
        print(f"Problemetic : {file_name}, with expected count {count}")
        print([(e,len(acc_bbdfk[e])) for e in acc_bbdfk])
        quit()    
    # df_bbdfk = pd.DataFrame.from_dict(acc_bbdfk)
    

    

# for key in acc_bbdfk:key, acc_bbdfk[key]


df = pd.DataFrame.from_dict(acc_dict)

# for key in acc_bbdfk:key, len(acc_bbdfk[key])
df_bbdfk = pd.DataFrame.from_dict(acc_bbdfk)
df_bbdfk = df_bbdfk.set_index('model_name')
if ENABLE_PREV:
    df_bbdfk.to_csv('prev-summary-external.csv')
elif ENABLE_PROPOSED2:
    df_bbdfk.to_csv('all_proposed_technique.csv')
elif ENABLE_Cross_Exchange:
    df_bbdfk.to_csv('non_local.csv')
else:
    df_bbdfk.to_csv('proposed-summary-external.csv')
# print(df_bbdfk[['dawn-Overall', 'wedge-Overall', 'FoggyCitiscape-Amodal', 'FoggyCitiscape-modal']])
# print(df_bbdfk[['FoggyCitiscape-Amodal', 'FoggyCitiscape-modal']])


print(df_bbdfk[['bbdk-overall', 'dawn-Overall', 'FoggyCity-Amodal:0.005', 'FoggyCity-modal:0.005', 'virtualkitti-Overall']])



if ENABLE_PROPOSED:
    df = df[new_order]     
    print(df.drop(['model_name'], axis=1))
    df = df.set_index('model_name')

    print(df)
    df.to_csv('proposed-summary.csv')




# cd ~/robustness_object_detection/GLIP/
# python Script/summarize_propose.py ucf_output/prev_work/NN/
# python Script/summarize_propose.py ucf_output/proposed/LangTri/
# python Script/summarize_propose.py ucf_output/prev_work/Done
# python Script/summarize_propose.py ucf_output/proposed/new_exp/Done/


