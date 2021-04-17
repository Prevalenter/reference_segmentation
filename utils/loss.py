import torch
import torch.nn as nn

def multi_task_loss(output, target, weight):
    # print(output.dim())
    assert output.dim() == 4
    num_channels = output.shape[1]
    loss_list = [torch.mean((output[:, i] - target[:, i])**2)*weight[i]
                                for i in range(num_channels)]
    return sum(loss_list), loss_list

if __name__ == '__main__':
	m = nn.Sigmoid()
	loss = nn.BCELoss()
	input = torch.randn((2, 2), requires_grad=True)
	# target = torch.empty((2, 2)).random_(2)
	target = torch.zeros((2,2))
	pred = m(input)
	print(pred)
	output = loss(pred, target)
	print(pred, output)
	print(target)
	# output.backward()

	print(-torch.log(1-pred).mean())


