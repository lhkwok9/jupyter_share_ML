import math
import numpy as np

softmax_output = [0.7, 0.1, 0.2]
target_output = [1,0,0]

loss = -(math.log(softmax_output[0])*target_output[0] + math.log(softmax_output[1])*target_output[1] + math.log(softmax_output[2])*target_output[2])

softmax_outputs = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])
class_target = [0,1,1]

neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_target])
average_loss = np.mean(neg_log)
print(average_loss) #0.38506088005216804
