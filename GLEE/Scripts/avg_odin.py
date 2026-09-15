import sys 

file = sys.argv[1]
count = int(sys.argv[2])

print("\n\n\n\n\n\n\n")


def check_float(s):
    try:
        float(s)  # Attempt to convert the string to a float
        return True
    except ValueError:  # Catch the ValueError if conversion fails
        return False

acc_dict = {}
name_found = 0
acc = 0
name = ""
print(" ** Results ** ")
with open(file, "r") as f:
    Lines = f.readlines()
    for line in Lines[::-1]:
        if "Command Line Args" in line:
            break  
        if "### odinw" in line:
            assert name_found == 0 
            name_found = 1
            name = line.replace(" ### ", "").replace("\n", "")
            print(name)
            print( f"{acc:.3f}")
            acc_dict[name] =acc 
            acc  = 0 
            name = ""
            name_found = 0 
        elif "==== : " in line :
            line = line.replace(" ==== : ", "")
            line = line.split(",")[0]
            if check_float(line):
                assert name_found == 0 and acc == 0 and name == "", f"name_found = {name_found}, acc = {acc}, name = {name}"
                acc = float(line) 
            

                
keys = [e for e in acc_dict]
assert (len(acc_dict) == count) or (len(keys) == count)
print(" ========", sum([acc_dict[e] for e in acc_dict]) / count )



# cd ~/robustness_object_detection/GLEE/
# python Scripts/avg_odin.py ucf_output/GLLE_LITE_STAGE2.txt-NOISE-ODINW13 13