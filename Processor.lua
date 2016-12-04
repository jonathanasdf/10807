require 'image'

require 'Model'

local M = torch.class('Processor')

M.cmd = torch.CmdLine()
function M:__init(model, processorOpts)
  self.model = model
  for k,v in pairs(self.cmd:parse(processorOpts:split(' ='))) do
    self[k] = v
  end
end

-- Takes path and returns a cuda tensor as input to the model
function M:preprocess(path)
  return image.load(path, 3):cuda()
end

function M:loadAndPreprocessInputs(pathNames)
  local first = self:preprocess(pathNames[1])
  local size = torch.LongStorage(first:dim() + 1)
  size[1] = #pathNames
  for i=1,first:dim() do
    size[i+1] = first:size(i)
  end
  local inputs = first.new(size)
  inputs[1] = first
  for i=2,#pathNames do
    inputs[i] = self:preprocess(pathNames[i])
  end
  return inputs
end

-- return the labels associated with the list of paths as cuda tensors
function M:getLabels(pathNames, outputs)
  error('getLabels is not defined.')
end

function M:forward(pathNames, inputs, deterministic)
  return self.model:forward(inputs, deterministic)
end

function M:backward(inputs, gradOutputs)
  return self.model:backward(inputs, gradOutputs)
end

function M:getLoss(outputs, labels)
  if not(self.criterion) then
    error('processor criterion is not defined. Either define a criterion or a custom getLoss function.')
  end

  local loss = self.criterion:forward(outputs, labels)
  local gradOutputs = self.criterion:backward(outputs, labels)
  if type(gradOutputs) == 'table' then
    for i=1,#gradOutputs do
      gradOutputs[i] = gradOutputs[i]
    end
  else
    gradOutputs = gradOutputs
  end

  return loss, gradOutputs
end

-- Called before each validation/test run
function M:resetStats() end
-- Calculate and update stats
function M.updateStats(pathNames, outputs, labels) end
-- Called after each epoch. Returns something printable
function M:getStats() end

-- Performs a single forward and backward pass through the model
function M:train(pathNames)
  local inputs = self:loadAndPreprocessInputs(pathNames)
  local outputs = self:forward(pathNames, inputs)
  local labels = self:getLabels(pathNames, outputs)
  local loss, gradOutputs = self:getLoss(outputs, labels)
  self:backward(inputs, gradOutputs)

  self:updateStats(pathNames, outputs, labels)
  return loss, #pathNames
end

-- return {aggregatedLoss, #instancesTested}
function M:test(pathNames)
  local inputs = self:loadAndPreprocessInputs(pathNames)
  local outputs = self:forward(pathNames, inputs, true)
  local labels = self:getLabels(pathNames, outputs)
  local loss, _ = self:getLoss(outputs, labels)

  self:updateStats(pathNames, outputs, labels)
  return loss, #pathNames
end

return M
