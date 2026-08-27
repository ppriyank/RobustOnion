import sys 
import os 
from collections import defaultdict
import pandas as pd 

summarize_all = False 
folder = None 
if len(sys.argv) == 1:
    summarize_all = True 
else:
    folder = sys.argv[1]

# folder="ucf_output/proposed/lrtko"
print("\n\n\n")
MAX_COUNT = 14
separator = "OrderedDict([('AP', "
acc_dict = defaultdict(list)
print(" ** Summarize ** ")

og_sets = ['OG', 'pixel_dropout', 'iso_blur', 'salt_pepper', 'low-res', 'focus_blur',
    'jpg_compression', 'chromatic', 'motion_blur2', 'fog', 'rain_model2', 'snow_model2',
    'atmospheric']    

bddk_sets_1 = ['BBDK-WEATHER-partly cloudy', 'BBDK-WEATHER-snowy', 'BBDK-WEATHER-rainy', 'BBDK-WEATHER-foggy', 'BBDK-WEATHER-overcast', 'BBDK-WEATHER-undefined', 'BBDK-WEATHER-clear']
bddk_sets_2 = ['BBDK-TIME-undefined', 'BBDK-TIME-night', 'BBDK-TIME-dawn/dusk', 'BBDK-TIME-daytime', 'BBDK-Overall-bbox']


wedge_sets =  ['WEDGE-cloudy', 'WEDGE-day', 'WEDGE-dust', 'WEDGE-fall', 'WEDGE-fog', 'WEDGE-hurricane', 'WEDGE-lightning', 'WEDGE-night', 'WEDGE-rain', 'WEDGE-snow', 'WEDGE-spring',
    'WEDGE-summer', 'WEDGE-sun', 'WEDGE-tornado', 'WEDGE-windy', 'WEDGE-winter', 'WEDGE-Overall', 'WEDGE', ]

dawn_sets = ['DAWN-Fog', 'DAWN-Rain', 'DAWN-Snow', 'DAWN-Sand', 'DAWN-Overall']

foggy_sets = [
    'FoggyCitiscape (modal) FOG - Level : 0.02', 'FoggyCitiscape (Amodal) FOG - Level : 0.02', 
    'FoggyCitiscape (modal) FOG - Level : 0.01', 'FoggyCitiscape (Amodal) FOG - Level : 0.01', 
    'FoggyCitiscape (modal) FOG - Level : 0.005', 'FoggyCitiscape (Amodal) FOG - Level : 0.005', 
]

foggy_sets_rename = {
    'FoggyCitiscape (Amodal) FOG - Level : 0.005':'FoggyCity-Amodal:0.005', 
    'FoggyCitiscape (modal) FOG - Level : 0.005':'FoggyCity-Modal:0.005', 
    
    'FoggyCitiscape (Amodal) FOG - Level : 0.01':'FoggyCity-Amodal:0.01', 
    'FoggyCitiscape (modal) FOG - Level : 0.01':'FoggyCity-Modal:0.01', 

    'FoggyCitiscape (Amodal) FOG - Level : 0.02':'FoggyCity-Amodal:0.02', 
    'FoggyCitiscape (modal) FOG - Level : 0.02':'FoggyCity-Modal:0.02', 
    
}
new_foggy_sets = [foggy_sets_rename[key] for key in foggy_sets]

new_order = og_sets + bddk_sets_1 + bddk_sets_2 + wedge_sets + dawn_sets + foggy_sets
    

y_axis = ['OG', 'pixel_dropout', 'iso_blur', 'salt_pepper', 
          'low-res', 'focus_blur', 'jpg_compression', 'chromatic', 'motion_blur2', 
          'fog', 'rain_model2', 'snow_model2', 'atmospheric'
]



def process_file(file, acc_dict):
    name_found = 0 
    counter = 0 
    encountered_in_this_set = set()
    with open(file, "r") as f:
        Lines = f.readlines()
        index = 0
        while index < len(Lines):
            line  = Lines[index]
            # print(line)
            if "####" in line:
                assert name_found == 0 , f"Results missing {line} :: {name}"
                name = line.split("#")[-1].strip()
                name_found = 1
                # print(name)
            elif "=====" in line:
                assert name not in encountered_in_this_set, f"{name} appeareing twice :O, {encountered_in_this_set}"
                encountered_in_this_set.add(name)
                counter += 1
                name_found = 0 
                if "FoggyCitiscape" in name :
                    splits = line.split("'AP50', ")
                    acc = splits[1].split("),")[0]
                else:
                    splits = line.split(separator)
                    acc = splits[1].split("),")[0]
                acc = float(acc) *  100
                # print(f"{name:<30}\t{acc:.3f}")
                acc_dict[name].append(acc)
            elif ":::::: Results :" in line:
                counter += 1
                name_found = 0 
                curr = index
                if name == 'DAWN':
                    splitter = "AP  :"
                else:
                    splitter = "AP50:"
                while curr < len(Lines):
                    curr += 1
                    line = Lines[curr]
                    if "####" in line or "=====" in line:
                        break 
                    if "::" in line :
                        # if name == 'DAWN':
                        #     import pdb
                        #     pdb.set_trace()
                        
                        acc = float(line.split(splitter)[1].split("\t")[0]) * 100 
                        local_name = line.split("::")[0].strip()
                        local_name = name + "-" +local_name
                        acc_dict[local_name].append(acc)
                        assert local_name not in encountered_in_this_set, f"{local_name} appeareing twice :O"
                        encountered_in_this_set.add(local_name)
                index = curr -1 
            index += 1
    for col in new_order:
        if col not in encountered_in_this_set:
            acc_dict[col].append(-1)
    
    

src_file = "ucf_output/Benchmark/GLIP-T_5-EVAL.txt"    
if folder:
    if "lrtko" in folder:
        index_name = "GLIP-T_5-LR_TKO"
        acc_dict[index_name].append("OG")
        process_file(src_file, acc_dict)
    elif "Adapter" in folder:
        index_name = "GLIP-T [5] Adapter"
        acc_dict[index_name].append("OG")
        process_file(src_file, acc_dict)
    elif "LoRA" in folder:
        index_name = "GLIP-T [5] LoRA"
        acc_dict[index_name].append("OG")
        process_file(src_file, acc_dict)
    elif "VPT" in folder:
        index_name = "GLIP-T [5] VPT"
        acc_dict[index_name].append("OG")
        process_file(src_file, acc_dict)

    print( len(acc_dict) )
    print([(e,len(acc_dict[e])) for e in acc_dict])

    for file in os.listdir(folder):
        if "EVAL" not in file:continue 
        potential_name = file.split("-EVAL")[0].split("-")[-1]
        if 'res' in potential_name and "jpg_compression" not in potential_name:
            potential_name = "low-res"
        print(f"\n {file} \n")    
        file = os.path.join(folder, file)
        strings = f"{potential_name}"
        acc_dict[index_name].append(potential_name)
        process_file(file, acc_dict)
        print([(e,len(acc_dict[e])) for e in acc_dict])


    df = pd.DataFrame.from_dict(acc_dict)
    df = df.set_index(index_name)
    df = df[new_order]               							
    df = df.reindex(y_axis)    
    df = df.rename(mapper=foggy_sets_rename, axis=1)

    for column in [og_sets, bddk_sets_1, bddk_sets_2, dawn_sets, new_foggy_sets]:
        print(df[column])



    df.to_csv(f'{index_name}-summary.csv')

else:
    vanilla_results = None 
    overall_df = None 
    for file in [
        'GLIP-T [5] Adapter-summary.csv', 'GLIP-T [5] LoRA-summary.csv', 
        'GLIP-T [5] VPT-summary.csv', 'GLIP-T_5-LR_TKO-summary.csv'
        ]:
        prefix =  None 
        if 'Adapter' in file:
            prefix = 'GLIP-T_5-ADAPTER'
        elif  'LoRA' in file:
            prefix = 'GLIP-T_5-LORA'
        elif  'VPT' in file:
            prefix = 'GLIP-T_5-VPT_DEEP'
        elif  'LR_TKO' in file:
            prefix = 'GLIP-T_5-LR_TKO'

        df = pd.read_csv(file)
        df['model_name'] = prefix + "-" +df[df.columns[0]]
        df = df.drop(columns=df.columns[0])
        df.loc[0, "model_name"] = 'vanilla'
        df = df.set_index('model_name')

        if vanilla_results is None :
            vanilla_results = df.iloc[0]
            # vanilla_results[]
        else:
            assert (vanilla_results.tolist() == df.iloc[0].tolist())
        
        df = df.drop(index='vanilla') 
        if overall_df is not None : 
            overall_df = pd.concat([overall_df, df])
        else:
            overall_df = df
        print(df)
    
    df = pd.concat([vanilla_results.to_frame().T, overall_df])
    df.to_csv('proposed-summary-external.csv')
        

# cd ~/robustness_object_detection/GLIP/
# python Script/summarize.py ucf_output/proposed/Baseline/lrtko
# python Script/summarize.py ucf_output/proposed/Baseline/Adapter
# python Script/summarize.py ucf_output/proposed/Baseline/VPT
# python Script/summarize.py ucf_output/proposed/Baseline/LoRA

# python Script/summarize.py 


# python summarize.py ucf_output/proposed/Baseline/Adapter
# python summarize.py ucf_output/proposed/Baseline/lrtko
# python summarize.py ucf_output/proposed/Baseline/LoRA
# python summarize.py ucf_output/proposed/Baseline/VPT


