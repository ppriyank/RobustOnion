import sys 

file = sys.argv[1]
count = int(sys.argv[2])

print("\n\n\n\n\n\n\n")



acc_dict = {}
name_found = 0
print("** Results **")
with open(file, "r") as f:
    Lines = f.readlines()
    line_found = None
    for line in Lines[::-1]:
        if "bbox_mAP" in line : 
            line_found = line 
            break 
    for line in line_found.split("  "):
        if "bbox_mAP:" in line:
            name, score = line.split(":")
            name = name.split("/")[0]
            acc = float(score) *  100
            acc_dict[name] =acc 
            print(f"{name:<30}\t{acc:.3f}")
            


assert len(acc_dict) == count
print(" \n\n========", sum([acc_dict[e] for e in acc_dict]) / count )


# cd ~/robustness_object_detection/MMGDINO/
# python Script/avg_odin.py ucf_output/GDINO-T.txt-ODWIN13 13
