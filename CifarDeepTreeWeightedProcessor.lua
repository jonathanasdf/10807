local CifarProcessor = require 'CifarProcessor'
local M = torch.class('CifarDeepTreeWeightedProcessor', 'CifarProcessor')

-- This processor is for the version where examples are sent down to **both** children with weights lambda and 1-lambda.
function M:__init(model, processorOpts)
  CifarProcessor.__init(self, model, processorOpts)

  self.criterion = nn.CrossEntropyCriterion():cuda()
  self.pairwiseDistance = nn.PairwiseDistance(2):cuda()
  self.hingeCriterion = nn.HingeEmbeddingCriterion():cuda()
end

function M:loadAndPreprocessInputs(pathNames, augmentations)
  local inputs = CifarProcessor.loadAndPreprocessInputs(self, pathNames, augmentations)
  return {inputs, torch.ones(inputs:size(1), 1):cuda()}
end

-- Output: {{batchSize x label_dist, batchSize x leaf_prob}, ...}
function M:getLoss(outputs, labels)
  local loss = 0
  local gradOutputs = {}
  for i=1,#outputs do
    gradOutputs[i] = {torch.zeros(outputs[i][1]:size()):cuda(), torch.zeros(outputs[i][2]:size()):cuda()}
  end
  for j=1,labels:size(1) do
    -- Only propagate gradients back from most likely leaf
    I = 1
    for i=2,#outputs do
      if outputs[i][2][j][1] > outputs[I][2][j][1] then
        I = i
      end
    end
    for i=1,#outputs do
      loss = loss + self.criterion:forward(outputs[i][1][j], labels[j]) * outputs[i][2][j][1]
--      if i == I then
      if i == labels[j] then
        gradOutputs[i][1][j] = self.criterion:backward(outputs[i][1][j], labels[j]) * outputs[i][2][j][1]
      end
    end
  end

  -- Add penalty for similar leaves?
  --for i=1,#outputs do
  --  for j=i+1,#outputs do
  --    for k=1,labels:size(1) do
  --      local I = {outputs[i][1][k], outputs[j][1][k]}
  --      local d = self.pairwiseDistance:forward(I)
  --      loss = loss + self.hingeCriterion:forward(d, -1)
  --      local g = self.hingeCriterion:backward(d, -1)
  --      local gradInputs = self.pairwiseDistance:backward(I, g)
  --      gradOutputs[i][1][k] = gradOutputs[i][1][k] + gradInputs[1]
  --      gradOutputs[j][1][k] = gradOutputs[j][1][k] + gradInputs[2]
  --    end
  --  end
  --end

  return loss, gradOutputs
end

function M:resetStats()
  CifarProcessor.resetStats(self)
  self.classCounts = torch.zeros(10)
  self.leafSum = {}
end

-- Output: {{batchSize x label_dist, batchSize x leaf_prob}, ...}
function M:updateStats(pathNames, outputs, labels)
  local avg = torch.zeros(#pathNames, 10)
  for i=1,#pathNames do
    for j=1,#outputs do
      avg[i]:add(outputs[j][1][i]:mul(outputs[j][2][i][1]):float())
    end
  end
  self.stats:batchAdd(avg, labels)

  for i=1,#pathNames do
    local c = labels[i]
    self.classCounts[c] = self.classCounts[c] + 1
    if self.leafSum[c] == nil then
      self.leafSum[c] = torch.zeros(#outputs)
    end
    for j=1,#outputs do
      self.leafSum[c][j] = self.leafSum[c][j] + outputs[j][2][i][1]
    end
  end
end

function M:getStats()
  local s = CifarProcessor.getStats(self)

  local r = 'Leaf probabilities per class:'
  for i=1,#self.leafSum do
    r = r .. '\n' .. tostring(i) .. ':'
    for j=1,self.leafSum[i]:size(1) do
      r = r .. ' ' .. tostring(math.floor((self.leafSum[i][j]/self.classCounts[i])*10000+0.5)/100)
    end
  end

  if opts.phase == 'train' then
    -- end of epoch, toggle frozen layers
    if opts.epoch % 2 == 1 then
      for k,v in pairs(self.model.modules[1].convs) do
        v.accGradInputs = self.model.modules[1].convGrads
      end
      for k,v in pairs(self.model.modules[1].splits) do
        v.accGradInputs = function() end
      end
    else
      for k,v in pairs(self.model.modules[1].convs) do
        v.accGradInputs = function() end
      end
      for k,v in pairs(self.model.modules[1].splits) do
        v.accGradInputs = self.model.modules[1].splitGrads
      end
    end
  end

  return s .. '\n' .. r
end

return M
