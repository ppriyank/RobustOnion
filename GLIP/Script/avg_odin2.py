import sys 

file = sys.argv[1]
count = int(sys.argv[2])

print("\n\n\n\n\n\n\n")


separator = "OrderedDict([('AP', "
acc_dict = {}
name_found = 0
print(" ** Results ** ")
with open(file, "r") as f:
    Lines = f.readlines()
    for line in Lines:
        if "** Results **" in line:
            # print("***********")
            break  
        if (".yaml" in line and "#### configs" in line) and (not name_found):
            name_found = 1
            if "task_config" in line :
                name = line.split("task_config=")[1].split(".yaml")[0]
            else:
                name = line.replace("\n", "")
            print(name)
        elif "=====" in line :
            name_found = 0 
            splits = line.split(separator)
            acc = splits[1].split("),")[0]
            acc = float(acc) *  100
            print( f"{acc:.2f}")
            acc_dict[name] =acc 


keys = [e for e in acc_dict if "plantdoc_100x100" not in e]
assert (len(acc_dict) == count) or (len(keys) == count)
print(" ========", sum([acc_dict[e] for e in acc_dict if "plantdoc_100x100" not in e]) / count )



# cd ~/robustness_object_detection/GLIP/
# python Script/avg_odin2.py ucf_output/GLIP-A.txt-ODWIN-13