-- This model is for the version where examples are sent down to a single children depending on if split is <= 0.5.
require 'DeepTreeNode'
local cmd = torch.CmdLine()
cmd:option('-preconv', 3, '#layers before branching')
cmd:option('-depth', 4, 'depth of tree. total #nodes = 2^depth')
cmd:option('-channels', 8, '#channels in conv layers')
cmd:option('-nodedepth', 1, '#conv layers per node')
cmd:option('-classes', 10, '#output classes')
cmd:option('-pool', '3,5', 'before which depths to add pooling layers (1-indexed)')
cmd:option('-resnet', false, 'use resnet blocks')

local function createModel(modelOpts)
  local m = nn.Sequential()
  m.modelOpts = cmd:parse(modelOpts and modelOpts:split(' =') or {})
  m.modelOpts.pool = m.modelOpts.pool:split(',')
  for i=1,#m.modelOpts.pool do
    m.modelOpts.pool[i] = tonumber(m.modelOpts.pool[i])
  end

  local c_in = 3
  local c = m.modelOpts.channels

  local size = 32
  local d = 1

  local preconv = nn.Sequential()
  for i=1,m.modelOpts.preconv do
    if tableContains(m.modelOpts.pool, d) then
      preconv:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      size = size / 2
    end
    if m.modelOpts.resnet then
      if i == 1 then
        preconv:add(nn.SpatialConvolution(c_in, c, 3, 3, 1, 1, 1, 1):noBias())
      else
        preconv:add(nn.BasicResidualModule(c_in, c))
      end
    else
      preconv:add(nn.SpatialConvolution(c_in, c, 3, 3, 1, 1, 1, 1):noBias())
      preconv:add(nn.SpatialBatchNormalization(c))
      preconv:add(nn.ReLU(true))
    end
    c_in = c
    d = d+1
  end
  m:add(preconv)

  local pool = false
  if tableContains(m.modelOpts.pool, d) then
    pool = true
    size = size / 2
  end
  local root = DeepTreeNode(m.modelOpts.nodedepth, c_in, c, size, size, pool, m.modelOpts.resnet)
  m:add(root)
  m.leaves = {root}
  m.nodes = {root}
  d = d+1

  for i=2,m.modelOpts.depth-1 do
    pool = false
    if tableContains(m.modelOpts.pool, d) then
      pool = true
      size = size / 2
    end

    local newleaves = {}
    for _, leaf in pairs(m.leaves) do
      local l = DeepTreeNode(m.modelOpts.nodedepth, m.modelOpts.channels, m.modelOpts.channels, size, size, pool, m.modelOpts.resnet)
      local r = DeepTreeNode(m.modelOpts.nodedepth, m.modelOpts.channels, m.modelOpts.channels, size, size, pool, m.modelOpts.resnet)
      leaf:addChildren({l, r})
      newleaves[#newleaves+1] = l
      m.nodes[#m.nodes+1] = l
      newleaves[#newleaves+1] = r
      m.nodes[#m.nodes+1] = r
    end
    m.leaves = newleaves
    d = d+1
  end

  pool = false
  if tableContains(m.modelOpts.pool, d) then
    pool = true
    size = size / 2
  end

  for _, leaf in pairs(m.leaves) do
    local l = nn.Sequential()
    local r = nn.Sequential()
    if pool then
      l:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      r:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    end
    for i=1,m.modelOpts.nodedepth do
      if m.modelOpts.resnet then
        l:add(nn.BasicResidualModule(m.modelOpts.channels, m.modelOpts.channels))
        r:add(nn.BasicResidualModule(m.modelOpts.channels, m.modelOpts.channels))
      else
        l:add(nn.SpatialConvolution(m.modelOpts.channels, m.modelOpts.channels, 3, 3, 1, 1, 1, 1):noBias())
        l:add(nn.SpatialBatchNormalization(m.modelOpts.channels))
        l:add(nn.ReLU(true))
        r:add(nn.SpatialConvolution(m.modelOpts.channels, m.modelOpts.channels, 3, 3, 1, 1, 1, 1):noBias())
        r:add(nn.SpatialBatchNormalization(m.modelOpts.channels))
        r:add(nn.ReLU(true))
      end
    end
    if m.modelOpts.resnet then
      l:add(nn.SpatialBatchNormalization(m.modelOpts.channels))
      l:add(nn.ReLU(true))
      r:add(nn.SpatialBatchNormalization(m.modelOpts.channels))
      r:add(nn.ReLU(true))
    end
    l:add(nn.SpatialAveragePooling(size, size))
    l:add(nn.View(-1, m.modelOpts.channels))
    l:add(nn.Linear(m.modelOpts.channels, m.modelOpts.classes))
    l:add(nn.SoftMax())
    r:add(nn.SpatialAveragePooling(size, size))
    r:add(nn.View(-1, m.modelOpts.channels))
    r:add(nn.Linear(m.modelOpts.channels, m.modelOpts.classes))
    r:add(nn.SoftMax())
    cudnn.convert(l, cudnn)
    cudnn.convert(r, cudnn)
    leaf:addChildren({l, r})
  end

  return m
end
return createModel
