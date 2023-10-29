import numpy as np
cls_label = np.zeros(3)
cls_label_set = [0,1,2]
cls_label = np.zeros(3)
for i in cls_label_set:
    cls_label[i] += 1
print((cls_label))