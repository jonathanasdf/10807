local M, parent = torch.class('GreedyInformationGainCriterion', 'nn.Criterion')

-- Input: {probabilities, mask}. Calculates information gain of split greedily for each example.
-- Find the best branch for the current example assuming that the branches for other examples are fixed.
function M:__init()
  self.criterion = nn.BCECriterion()
  self.leftCount = torch.Tensor(1)
  self.rightCount = torch.Tensor(1)
end

local function entropy(labels)
  local res = 0
  if labels:dim() > 0 then
    local N = labels:size(1)
    for l=labels:min(),labels:max() do
      local cnt = labels:eq(l):sum()
      if cnt > 0 then
        res = res + cnt/N * math.log(cnt/N)
      end
    end
  end
  return -res
end

local function sumEntropy(left, right)
  local l = left:dim() == 0 and 0 or left:size(1)
  local r = right:dim() == 0 and 0 or right:size(1)
  return (l * entropy(left) + r * entropy(right)) / (l + r)
end

function M:updateOutput(input, target)
  input = input:dim() == 1 and input or input:squeeze(2)
  target = target:dim() == 1 and target or target:squeeze(2)

  local left = input:le(0.5)
  local right = input:gt(0.5)
  local leftTotal = left:sum()
  local rightTotal = right:sum()
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
    local eq = target:eq(l)
    self.leftCount[l] = torch.cmin(left, eq):sum()
    if self.leftCount[l] > 0 then
      leftEntropy = leftEntropy + self.leftCount[l] * math.log(self.leftCount[l])
    end
    self.rightCount[l] = eq:sum() - self.leftCount[l]
    if self.rightCount[l] > 0 then
      rightEntropy = rightEntropy + self.rightCount[l] * math.log(self.rightCount[l])
    end
  end

  self.labels = input:gt(0.5):type(input:type())
  local base = 0
  if leftTotal > 0 then
    base = base + (leftTotal*math.log(leftTotal) - leftEntropy) / N
  end
  if rightTotal > 0 then
    base = base + (rightTotal*math.log(rightTotal) - rightEntropy) / N
  end
  for i=1,N do
    local class = target[i]
    local diff = 0
    if input[i] <= 0.5 then
      -- switch left to right
      local leftd = self.leftCount[class] <= 1 and 0 or (self.leftCount[class]-1) * math.log(self.leftCount[class]-1) - self.leftCount[class] * math.log(self.leftCount[class])
      local rightd = self.rightCount[class] == 0 and 0 or (self.rightCount[class]+1) * math.log(self.rightCount[class]+1) - self.rightCount[class] * math.log(self.rightCount[class])
      if leftTotal > 1 then
        diff = diff + ((leftTotal-1)*math.log(leftTotal-1) - leftEntropy - leftd) / N
      end
      diff = diff + ((rightTotal+1)*math.log(rightTotal+1) - rightEntropy - rightd) / N
    else
      -- switch right to left
      local leftd = self.leftCount[class] == 0 and 0 or (self.leftCount[class]+1) * math.log(self.leftCount[class]+1) - self.leftCount[class] * math.log(self.leftCount[class])
      local rightd = self.rightCount[class] <= 1 and 0 or (self.rightCount[class]-1) * math.log(self.rightCount[class]-1) - self.rightCount[class] * math.log(self.rightCount[class])
      diff = diff + ((leftTotal+1)*math.log(leftTotal+1) - leftEntropy - leftd) / N
      if rightTotal > 1 then
        diff = diff + ((rightTotal-1)*math.log(rightTotal-1) - rightEntropy - rightd) / N
      end
    end
    if diff < base then
      self.labels[i] = 1-self.labels[i]
    end
  end
  self.output = sumEntropy(target[self.labels:lt(1)], target[self.labels])

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
