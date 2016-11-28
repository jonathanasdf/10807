-- This model is for the version where examples are sent down to a single children depending on if split is <= 0.5.
-- Requires layer by layer pretraining to learn split functions.

require 'dpnn'
local CifarProcessor = require 'CifarProcessor'
local M = torch.class('CifarDeepTreeProcessor', 'CifarProcessor')

function M:__init(model, processorOpts)
  CifarProcessor.__init(self, model, processorOpts)

  self.entropy = nn.CategoricalEntropy()
  self.criterion = nn.CrossEntropyCriterion():cuda()
end

function M:getLoss(outputs, labels)
  if self.model.pretraining then
    
  else
    return CifarProcessor.getLoss(self, outputs, labels)
  end
end

return M
