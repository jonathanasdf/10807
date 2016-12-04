local DeepTreeNode, Parent = torch.class('DeepTreeNode', 'nn.Module')
function DeepTreeNode:__init(nConvLayers, nInputPlane, nOutputPlane, inputWidth, inputHeight)
  Parent.__init(self)

  self.nConvLayers = nConvLayers
  self.nInputPlane = nInputPlane
  self.nOutputPlane = nOutputPlane
  self.inputWidth = inputWidth
  self.inputHeight = inputHeight

  self.conv = nn.Sequential()
  local inputChannels = nInputPlane
  for i=1,nConvLayers do
    self.conv:add(nn.SpatialConvolution(inputChannels, nOutputPlane, 3, 3, 1, 1, 1, 1):noBias())
    self.conv:add(nn.SpatialBatchNormalization(nOutputPlane))
    self.conv:add(nn.ReLU(true))
    inputChannels = nOutputPlane
  end

  self.split = nn.Sequential()
  self.split:add(nn.View(-1, nOutputPlane*inputWidth*inputHeight))
  self.split:add(nn.Linear(nOutputPlane*inputWidth*inputHeight, 1))
  self.split:add(nn.Sigmoid())
end

function DeepTreeNode:addChildren(children)
  assert(#children == 2)
  self.children = children
end

function DeepTreeNode:updateOutput(input)
  self.conv_output = self.conv:updateOutput(input)
  self.split_output = self.split:updateOutput(self.conv_output)
  if self.children then
    self.left = torch.nonzero(torch.le(self.split_output, 0.5))
    self.right = torch.nonzero(torch.gt(self.split_output, 0.5))
    local outputSize
    if self.left:nElement() > 0 then
      self.left_convOutput = torch.index(self.conv_output, self.left)
      self.left_output = self.children[1]:updateOutput(self.left_convOutput)
      outputSize = self.left_output:size()
    end
    if self.right:nElement() > 0 then
      self.right_convOutput = torch.index(self.conv_output, self.right)
      self.right_output = self.children[2]:updateOutput(self.right_convOutput)
      outputSize = self.right_output:size()
    end
    outputSize[1] = input:size(1)
    if not self.output:size():equal(outputSize) then
      self.output = input.new(outputSize)
    end
    local leftId = 1
    local rightId = 1
    for i=1,input:size(1) do
      if self.split_output[i] <= 0.5 then
        self.output[i] = self.left_output[leftId]
        leftId = leftId + 1
      else
        self.output[i] = self.right_output[rightId]
        rightId = rightId + 1
      end
    end
  else
    self.output = self.split_output
  end
  return self.output
end

function DeepTreeNode:updateGradInput(input, gradOutput)
  if not self.conv_gradOutput or not self.conv_gradOutput:size():equal(self.conv_output:size()) then
    self.conv_gradOutput = self.conv_output.new(self.conv_output:size())
  end
  if self.children then
    if self.left:nElement() > 0 then
      self.left_gradInput = self.children[1]:updateGradInput(
          self.left_convOutput, torch.index(gradOutput, self.left))
    end
    if self.right:nElement() > 0 then
      self.right_gradInput = self.children[2]:updateGradInput(
          self.right_convOutput, torch.index(gradOutput, self.right))
    end
    local leftId = 1
    local rightId = 1
    for i=1,input:size(1) do
      if self.split_output[i] <= 0.5 then
        self.conv_gradOutput[i] = self.left_gradInput[leftId]
        leftId = leftId + 1
      else
        self.conv_gradOutput[i] = self.right_gradInput[rightId]
        rightId = rightId + 1
      end
    end
  else
    self.conv_gradOutput = self.split:updateGradInput(self.conv_output, gradOutput)
  end
  self.gradInput = self.conv:updateGradInput(input, self.conv_gradOutput)
  return self.gradInput
end

function DeepTreeNode:accGradParameters(input, gradOutput, scale) 
  if self.children then
    if self.left:nElement() > 0 then
      self.children[1]:accGradParameters(self.left_convOutput, torch.index(gradOutput, self.left), scale)
    end
    if self.right:nElement() > 0 then
      self.children[2]:accGradParameters(self.right_convOutput, torch.index(gradOutput, self.right), scale)
    end
  else
    self.split:accGradParameters(self.conv_output, gradOutput, scale)
  end
  self.conv:accGradParameters(input, self.conv_gradOutput, scale)
end

function DeepTreeNode:clearState()
  self.conv_output = nil
  self.split_output = nil
  self.left = nil
  self.right = nil
  self.left_convOutput = nil
  self.right_convOutput = nil
  self.left_output = nil
  self.right_output = nil
  self.left_gradInput = nil
  self.right_gradInput = nil
  self.conv_gradOutput = nil
  return Parent.clearState(self) 
end

function DeepTreeNode:__tostring__()
  if self.children then
    return string.format('%s\n  %s\n  %s', torch.type(self), self.children[1], self.children[2]);
  else
    return string.format('%s with no children', torch.type(self));
  end
end

