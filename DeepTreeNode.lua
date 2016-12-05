require 'resnet'
local DeepTreeNode, Parent = torch.class('DeepTreeNode', 'nn.Module')
function DeepTreeNode:__init(nConvLayers, nInputPlane, nOutputPlane, inputWidth, inputHeight, pooling, resnet)
  Parent.__init(self)

  self.nConvLayers = nConvLayers
  self.nInputPlane = nInputPlane
  self.nOutputPlane = nOutputPlane
  self.inputWidth = inputWidth
  self.inputHeight = inputHeight
  self.pooling = pooling or false
  self.resnet = resnet or false
  self.id = 0

  self.conv = nn.Sequential()
  if pooling then
    self.conv:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  end
  local inputChannels = nInputPlane
  for i=1,nConvLayers do
    if self.resnet then
      self.conv:add(nn.BasicResidualModule(inputChannels, nOutputPlane))
    else
      self.conv:add(nn.SpatialConvolution(inputChannels, nOutputPlane, 3, 3, 1, 1, 1, 1):noBias())
      self.conv:add(nn.SpatialBatchNormalization(nOutputPlane))
      self.conv:add(nn.ReLU(true))
    end
    inputChannels = nOutputPlane
  end

  self.split = nn.Sequential()
  if self.resnet then
    self.split:add(nn.SpatialBatchNormalization(nOutputPlane))
    self.split:add(nn.ReLU(true))
  end
  self.split:add(nn.View(-1, nOutputPlane*inputWidth*inputHeight))
  self.split:add(nn.Linear(nOutputPlane*inputWidth*inputHeight, 1))
  self.split:add(nn.Sigmoid())

  cudnn.convert(self.conv, cudnn)
  cudnn.convert(self.split, cudnn)
end

function DeepTreeNode:addChildren(children)
  assert(#children == 2)
  self.children = children
  for i=1,#self.children do
    self.children[i].id = self.id*2 + i
  end
end

function DeepTreeNode:updateOutput(input, mask)
  self.mask = mask or input.new(input:size(1)):gt(0):fill(1)
  self.conv_output = self.conv:updateOutput(input)
  self.split_output = self.split:updateOutput(self.conv_output)
  self.left = torch.cmin(self.mask, torch.le(self.split_output, 0.5))
  self.right = torch.cmin(self.mask, torch.gt(self.split_output, 0.5))
  if self.children then
    local left_output = self.children[1]:updateOutput(self.conv_output, self.left)
    local right_output = self.children[2]:updateOutput(self.conv_output, self.right)

    if self.output:dim() ~= left_output:dim() or
       self.output:size(1) ~= left_output:size(1) or
       self.output:size(2) ~= left_output:size(2) then
      self.output = left_output.new(left_output:size())
    end
    if self.forwardWeighted then
      local left_expanded = torch.expandAs((1-self.split_output):view(input:size(1), 1), self.output)
      local right_expanded = torch.expandAs(self.split_output:view(input:size(1), 1), self.output)
      self.output = torch.cmul(left_expanded, left_output) + torch.cmul(right_expanded, right_output)
    else
      self.output:zero()
      local left_expanded = torch.expandAs(self.left:view(input:size(1), 1), self.output)
      self.output[left_expanded] = left_output[left_expanded]
      local right_expanded = torch.expandAs(self.right:view(input:size(1), 1), self.output)
      self.output[right_expanded] = right_output[right_expanded]
    end
  else
    self.output = self.split_output
  end
  return self.output
end

function DeepTreeNode:zeroGradParameters()
  if self.children then
    self.children[1]:zeroGradParameters()
    self.children[2]:zeroGradParameters()
  end
  self.split:zeroGradParameters()
  self.conv:zeroGradParameters()
end

function DeepTreeNode:updateGradInput(input, gradOutput)
  if not self.conv_gradOutput or self.conv_gradOutput:dim() ~= self.conv_output:dim() or
     self.conv_gradOutput:size(1) ~= self.conv_output:size(1) or self.conv_gradOutput:size(2) ~= self.conv_output:size(2) or
     self.conv_gradOutput:size(3) ~= self.conv_output:size(3) or self.conv_gradOutput:size(4) ~= self.conv_output:size(4) then
    self.conv_gradOutput = self.conv_output.new(self.conv_output:size())
  end
  self.conv_gradOutput:zero()

  if self.children then
    if not self.left_gradOutput or
       self.left_gradOutput:size(1) ~= gradOutput:size(1) or self.left_gradOutput:size(2) ~= gradOutput:size(2) then
      self.left_gradOutput = gradOutput.new(gradOutput:size())
      self.right_gradOutput = gradOutput.new(gradOutput:size())
    end
    self.left_gradOutput:zero()
    local left_expanded = torch.expandAs(self.left:view(input:size(1), 1), gradOutput)
    self.left_gradOutput[left_expanded] = gradOutput[left_expanded]
    local left_gradInput = self.children[1]:updateGradInput(self.conv_output, self.left_gradOutput)

    self.right_gradOutput:zero()
    local right_expanded = torch.expandAs(self.right:view(input:size(1), 1), gradOutput)
    self.right_gradOutput[right_expanded] = gradOutput[right_expanded]
    local right_gradInput = self.children[2]:updateGradInput(self.conv_output, self.right_gradOutput)

    left_expanded = torch.expandAs(self.left:view(input:size(1), 1, 1, 1), self.conv_gradOutput)
    self.conv_gradOutput[left_expanded] = left_gradInput[left_expanded]

    right_expanded = torch.expandAs(self.right:view(input:size(1), 1, 1, 1), self.conv_gradOutput)
    self.conv_gradOutput[right_expanded] = right_gradInput[right_expanded]
  else
    self.conv_gradOutput = self.split:updateGradInput(self.conv_output, gradOutput)
  end
  self.gradInput = self.conv:updateGradInput(input, self.conv_gradOutput)
  return self.gradInput
end

function DeepTreeNode:accGradParameters(input, gradOutput, scale)
  if self.children then
    self.children[1]:accGradParameters(self.conv_output, self.left_gradOutput, scale)
    self.children[2]:accGradParameters(self.conv_output, self.right_gradOutput, scale)
  else
    self.split:accGradParameters(self.conv_output, gradOutput, scale)
  end
  self.conv:accGradParameters(input, self.conv_gradOutput, scale)
end

function DeepTreeNode:parameters()
  local function tinsert(to, from)
    if type(from) == 'table' then
      for i=1,#from do
        tinsert(to,from[i])
      end
    else
      table.insert(to,from)
    end
  end
  local w = {}
  local gw = {}
  local function insert(module)
    local mw, mgw = module:parameters()
    if mw then
      tinsert(w, mw)
      tinsert(gw, mgw)
    end
  end
  if self.children then
    insert(self.children[1])
    insert(self.children[2])
  end
  insert(self.split)
  insert(self.conv)
  return w, gw
end

function DeepTreeNode:clearState()
  self.mask = nil
  self.conv_output = nil
  self.split_output = nil
  self.left = nil
  self.right = nil
  self.conv_gradOutput = nil
  self.left_gradOutput = nil
  self.right_gradOutput = nil
  return Parent.clearState(self)
end

function DeepTreeNode:__tostring__()
  if self.children then
    res = string.format('%s @ {\n|   %s\n|-%s\n|-%s\n}', torch.type(self),
              tostring(self.conv):gsub('\n', '\n| '),
              tostring(self.children[1]):gsub('\n', '\n| '),
              tostring(self.children[2]):gsub('\n', '\n| '))
  else
    res = string.format('%s with no children @ {\n  %s\n  %s\n}', torch.type(self),
              tostring(self.conv):gsub('\n', '\n  '),
              tostring(self.split):gsub('\n', '\n  '))
  end
  return res
end

