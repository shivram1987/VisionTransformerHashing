import matplotlib.pyplot as plt
from utils.tools import draw_range
plt.rcParams.update({'font.size': 28})
# Precision Recall Curve data
Dataset = "imagenet"

#Method = "DSH"
#Method = "HashNet"
#Method = "GreedyHash"
#Method = "IDHN"
#Method = "CSQ"
Method = "DPN"

Bit = "64"
Model = ["AlexNet", "ResNet", "ViT-B_32", "ViT-B_16"]
Legends = ["AlexNet", "ResNet", "VTS32", "VTS16"]
pathfile = "Checkpoints_Results1/" + Dataset + "/" + Method+"/" 
print(pathfile)

markers = "DdsPvo*xH1234h"
model2marker = {}
i = 0
for model in Model:
    model2marker[model] = markers[i]
    i += 1

plt.figure(figsize=(11, 9))
for model in Model:
    pathfile_model = pathfile + Dataset + "_" + Method + "_" + model + "_Bit" + Bit + ".txt"
    print(pathfile_model)
    file_model = open(pathfile_model, 'r')
    Lines = file_model.readlines()
    count = 0
    for line in Lines:
        count += 1
        if line.find("PR") != -1:
           #print(line.rfind("|"))
           data = line[line.rfind("|")+2:-2].split(' ')
           #print("Line{}: {}".format(count, line.strip()))
    P = [float(data[j]) for j in range(len(data)) if j % 2 != 1]
    R = [float(data[j]) for j in range(len(data)) if j % 2 != 0]
    plt.plot(R, P, linestyle="-", marker=model2marker[model], label=model, linewidth=4, markersize=12)
    print(P)
    print(R)
plt.grid(True)
#plt.xlim(0, 1)
#plt.ylim(0, 1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(Legends)
plt.savefig(Dataset + "_" + Method + "_Bit" + Bit + "_pr.pdf")
plt.show()
pause




"""markers = "DdsPvo*xH1234h"
method2marker = {}
i = 0
for method in pr_data:
    method2marker[method] = markers[i]
    i += 1

plt.figure(figsize=(7, 5))
#plt.subplot(131)
for method in pr_data:
    P, R = pr_data[method]
    plt.plot(R, P, linestyle="-", marker=method2marker[method], label=method)
plt.grid(True)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig(Dataset + "_" + Method + "_Bit" + Bit + "_pr.pdf")

plt.figure(figsize=(7, 5))
#plt.subplot(132)
for method in pr_data:
    P, R = pr_data[method]
    plt.plot(draw_range, R, linestyle="-", marker=method2marker[method], label=method)
plt.xlim(0, max(draw_range))
plt.grid(True)
plt.xlabel('The number of retrieved samples')
plt.ylabel('Recall')
plt.legend()
plt.savefig(Dataset + "_" + Method + "_recall.pdf")

plt.figure(figsize=(7, 5))
#plt.subplot(133)
for method in pr_data:
    P, R = pr_data[method]
    plt.plot(draw_range, P, linestyle="-", marker=method2marker[method], label=method)
plt.xlim(0, max(draw_range))
plt.grid(True)
plt.xlabel('The number of retrieved samples')
plt.ylabel('Precision')
plt.legend()
plt.savefig(Dataset + "_" + Method + "_precision.pdf")
plt.show()"""
