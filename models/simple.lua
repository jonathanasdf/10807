require 'resnet'
local cmd = torch.CmdLine()
cmd:option('-layers', '8,8,8,8,8,8', 'comma separated list of layer channel sizes')
cmd:option('-classes', 10, '#output classes')
cmd:option('-pool', '3,5', 'before which depths to add pooling layers (1-indexed)')
cmd:option('-resnet', false, 'use resnet blocks')

local function createModel(modelOpts)
  modelOpts = cmd:parse(modelOpts and modelOpts:split(' =') or {})
  modelOpts.layers = modelOpts.layers:split(',')
  for i=1,#modelOpts.layers do
    modelOpts.layers[i] = tonumber(modelOpts.layers[i])
  end
  modelOpts.pool = modelOpts.pool:split(',')
  for i=1,#modelOpts.pool do
    modelOpts.pool[i] = tonumber(modelOpts.pool[i])
  end

  local c_in = 3
  local c

  local m = nn.Sequential()
  local size = 32

  local d = 1
  for i=1,#modelOpts.layers do
    if tableContains(modelOpts.pool, d) then
      m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      size = size / 2
    end
    c = modelOpts.layers[i]
    if modelOpts.resnet then
      if i == 1 then
        m:add(nn.SpatialConvolution(c_in, c, 3, 3, 1, 1, 1, 1):noBias())
      else
        m:add(nn.BasicResidualModule(c_in, c))
      end
    else
      m:add(nn.SpatialConvolution(c_in, c, 3, 3, 1, 1, 1, 1):noBias())
      m:add(nn.SpatialBatchNormalization(c))
      m:add(nn.ReLU(true))
    end
    c_in = c
    d = d + 1
  end

  if modelOpts.resnet then
    m:add(nn.SpatialBatchNormalization(c))
    m:add(nn.ReLU(true))
  end
  m:add(nn.SpatialAveragePooling(size, size))
  m:add(nn.View(-1, c))
  m:add(nn.Linear(c, modelOpts.classes))
  m:add(nn.SoftMax())
  return m
end
return createModel
