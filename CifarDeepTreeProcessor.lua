-- This model is for the version where examples are sent down to a single children depending on if split is <= 0.5.

require 'dpnn'
require 'GreedyInformationGainCriterion'
require 'InformationGainCriterion'
local CifarProcessor = require 'CifarProcessor'
local M = torch.class('CifarDeepTreeProcessor', 'CifarProcessor')

function M:__init(model, processorOpts)
  self.cmd:option('-greedy', false, 'use greedy instance-wise information gain for training splits')
  self.cmd:option('-forwardWeighted', false, 'use weighted combination of leaf predictions. Not available in training mode')
  CifarProcessor.__init(self, model, processorOpts)

  if self.greedy then
    self.splitCriterion = GreedyInformationGainCriterion():cuda()
  else
    self.splitCriterion = InformationGainCriterion():cuda()
  end
end

function M:train(pathNames)
  local inputs = self:loadAndPreprocessInputs(pathNames)
  local outputs = self:forward(pathNames, inputs)
  local labels = self:getLabels(pathNames, outputs)
  local loss = 0
  local gradOutputs
  if not self.model.pretraining then
    loss, gradOutputs = self:getLoss(outputs, labels)
    self:backward(inputs, gradOutputs)
  else
    gradOutputs = outputs.new(outputs:size()):zero()
    for _, leaf in pairs(self.model.module.leaves) do
      if torch.sum(leaf.mask) > 0 then
        local l = self.splitCriterion:forward(outputs[leaf.mask], labels[leaf.mask])
        gradOutputs[leaf.mask] = self.splitCriterion:backward(outputs[leaf.mask], labels[leaf.mask])
        loss = loss + l
      end
    end
    self:backward(inputs, gradOutputs)
  end

  -- update old splits
  for _, node in pairs(self.model.module.nodes) do
    if torch.sum(node.mask) > 0 then
      gradOutputs = node.split_output.new(node.split_output:size()):zero()
      local l = self.splitCriterion:forward(node.split_output[node.mask], labels[node.mask])
      gradOutputs[node.mask] = self.splitCriterion:backward(node.split_output[node.mask], labels[node.mask])
      loss = loss + l
      node.split:backward(node.conv_output, gradOutputs)
    end
  end

  self:updateStats(pathNames, outputs, labels)
  return loss, #pathNames
end

function M:test(pathNames)
  if opts.phase == 'test' and self.forwardWeighted then
    for _, node in pairs(self.model.module.nodes) do
      node.forwardWeighted = true
    end
  end
  return CifarProcessor.test(self, pathNames)
end

function M:resetStats()
  if not self.model.pretraining then
    CifarProcessor.resetStats(self)
  end

  self.classCounts = torch.zeros(2*#self.model.module.leaves, self.model.module.modelOpts.classes)
end

function M:updateStats(pathNames, outputs, labels)
  if not self.model.pretraining then
    CifarProcessor.updateStats(self, pathNames, outputs, labels)
  end

  local id = 1
  for _, leaf in pairs(self.model.module.leaves) do
    if torch.sum(leaf.left) > 0 then
      local left = labels:maskedSelect(leaf.left)
      for j=1,left:size(1) do
        self.classCounts[id][left[j]] = self.classCounts[id][left[j]] + 1
      end
    end
    id = id + 1
    if torch.sum(leaf.right) > 0 then
      local right = labels:maskedSelect(leaf.right)
      for j=1,right:size(1) do
        self.classCounts[id][right[j]] = self.classCounts[id][right[j]] + 1
      end
    end
    id = id + 1
  end
end

function M:getStats()
  local res = ""
  if not self.model.pretraining then
    res = CifarProcessor.getStats(self)
  end
  res = res .. '\n' .. self.classCounts:__tostring__()
  return res
end

return M
