local M, parent = torch.class('InformationGainCriterion', 'nn.Criterion')

-- Input: {probabilities, mask}. Calculates information gain of best split for the minibatch
function M:__init()
  self.criterion = nn.BCECriterion()
  self.leftCount = torch.Tensor(1)
  self.rightCount = torch.Tensor(1)
end

function M:updateOutput(input, target)
  input = input:dim() == 1 and input or input:squeeze(2)
  target = target:dim() == 1 and target or target:squeeze(2)

  if self.leftCount:size(1) < target:max() then
    self.leftCount = target.new(target:max())
    self.rightCount = target.new(target:max())
  end
  self.leftCount:zero()
  self.rightCount:zero()
  local leftEntropy = 0
  local rightEntropy = 0
  local N = target:size(1)
  for l=target:min(),target:max() do
    self.rightCount[l] = target:eq(l):sum()
    if self.rightCount[l] > 0 then
      rightEntropy = rightEntropy + self.rightCount[l] * math.log(self.rightCount[l])
    end
  end

  local bestsplit = 0
  self.output = math.log(N) - rightEntropy/N
  local values,I = torch.sort(input)
  for i=1,N do
    local class = target[I[i]]
    if self.leftCount[class] > 0 then
      leftEntropy = leftEntropy - self.leftCount[class] * math.log(self.leftCount[class])
    end
    self.leftCount[class] = self.leftCount[class] + 1
    leftEntropy = leftEntropy + self.leftCount[class] * math.log(self.leftCount[class])

    rightEntropy = rightEntropy - self.rightCount[class] * math.log(self.rightCount[class])
    self.rightCount[class] = self.rightCount[class] - 1
    if self.rightCount[class] > 0 then
      rightEntropy = rightEntropy + self.rightCount[class] * math.log(self.rightCount[class])
    end

    local sumEntropy = (i*math.log(i) - leftEntropy) / N
    if i < N then
      sumEntropy = sumEntropy + ((N-i)*math.log(N-i) - rightEntropy) / N
    end

    if sumEntropy < self.output then
      self.output = sumEntropy
      bestsplit = values[i]
    end
  end
  self.labels = input:gt(bestsplit):type(input:type())

  -- gradients can explode if predicted probability is too close to 0 or 1
  self.pred = torch.cmax(input, input.new(1):fill(1e-3):expandAs(input))
  self.pred = torch.cmin(self.pred, input.new(1):fill(1-1e-3):expandAs(input))
  self.criterion:forward(self.pred, self.labels)
  return self.output
end

function M:updateGradInput(input, target)
  self.gradInput = self.criterion:backward(self.pred, self.labels)
  return self.gradInput
end
