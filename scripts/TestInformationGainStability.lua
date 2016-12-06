require 'cunn'
require 'cudnn'
require 'DeepTreeNode'
require 'InformationGainCriterion'
require 'GreedyInformationGainCriterion'
m = DeepTreeNode(2, 3, 64, 32, 32):cuda()
m:zeroGradParameters()
c = GreedyInformationGainCriterion():cuda()
--c = InformationGainCriterion():cuda()
bce = nn.BCECriterion():cuda()
data = torch.load('/mnt/sdb1/jshen/cifar10/data.t7')

-- 2 = car, 4 = cat, 6 = dog, 10 = truck
valid = torch.cmax(torch.cmax(data.labels:eq(2), data.labels:eq(4)), torch.cmax(data.labels:eq(6), data.labels:eq(10)))
data.data = data.data:index(1, torch.nonzero(valid):squeeze())
data.labels = data.labels[valid]
data.labels[data.labels:eq(2)] = 1
data.labels[data.labels:eq(4)] = 2
data.labels[data.labels:eq(6)] = 3
data.labels[data.labels:eq(10)] = 4

local function printDist(m)
  dist = torch.zeros(2, 4)
  for i=1,data.labels:size(1),100 do
    inputs = data.data[{{i,i+99},{},{},{}}]:cuda()
    labels = data.labels[{{i,i+99}}]:cuda()
    out = m:forward(inputs)
    for j=1,labels:size(1) do
      if out[j][1] <= 0.5 then
        dist[1][labels[j]] = dist[1][labels[j]] + 1
      else
        dist[2][labels[j]] = dist[2][labels[j]] + 1
      end
    end
  end
  print(dist)
end

print('Optimizing information gain')
for i=1,10000 do
  batch = torch.randperm(data.labels:size(1))[{{1,100}}]:long()
  inputs = data.data:index(1, batch):cuda()
  labels = data.labels:index(1, batch):clone():cuda()

  out = m:forward(inputs)
  l = c:forward(out, labels)
  if i%1000 == 0 then
    print(i, l)
  end
  grad = c:backward(out, labels)
  m:backward(inputs, grad)
  m:updateParameters(0.01)
  m:zeroGradParameters()
end

print('Dist after optimizing information gain')
printDist(m)

print('Pretraining to split car/cat from dog/truck')
for i=1,10000 do
  batch = torch.randperm(data.labels:size(1))[{{1,100}}]:long()
  inputs = data.data:index(1, batch):cuda()
  labels = data.labels:index(1, batch):clone():cuda()
  labels[labels:eq(1)] = 0
  labels[labels:eq(2)] = 0
  labels[labels:eq(3)] = 1
  labels[labels:eq(4)] = 1

  out = m:forward(inputs)
  l = bce:forward(out, labels)
  if i%1000 == 0 then
    print(i, l)
  end
  grad = bce:backward(out, labels)
  m:backward(inputs, grad)
  m:updateParameters(0.01)
  m:zeroGradParameters()
end

print('Dist after pretraining')
printDist(m)

print('Optimizing information gain')
for i=1,10000 do
  batch = torch.randperm(data.labels:size(1))[{{1,100}}]:long()
  inputs = data.data:index(1, batch):cuda()
  labels = data.labels:index(1, batch):clone():cuda()

  out = m:forward(inputs)
  l = c:forward(out, labels)
  if i%1000 == 0 then
    print(i, l)
  end
  grad = c:backward(out, labels)
  m:backward(inputs, grad)
  m:updateParameters(0.01)
  m:zeroGradParameters()
end

print('Dist after optimizing information gain')
printDist(m)





