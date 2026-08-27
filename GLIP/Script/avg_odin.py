import sys 

file = sys.argv[1]
count = int(sys.argv[2])

print("\n\n\n\n\n\n\n")

separator="task_config='"

separator = "maskrcnn_benchmark.inference INFO: OrderedDict([('bbox', OrderedDict([('AP', "
acc_dict = {}
name_found = 0
with open(file, "r") as f:
    Lines = f.readlines()
    for line in Lines:
        if (".yaml" in line and "maskrcnn_benchmark INFO: " not in line or "task_config" in line) and (not name_found):
            name_found = 1
            if "task_config" in line :
                name = line.split("task_config=")[1].split(".yaml")[0]
            else:
                name = line.replace("\n", "")
            print(name)
        elif "maskrcnn_benchmark.inference INFO: OrderedDict" in line :
            name_found = 0 
            splits = line.split(separator)
            acc = splits[1].split("),")[0]
            acc = float(acc) *  100
            print( f"{acc:.2f}")
            acc_dict[name] =acc 


assert len(acc_dict) == count
print(" ========", sum([acc_dict[e] for e in acc_dict]) / count )

