-- This model is for the version where examples are sent down to a single children depending on if split is <= 0.5.
require 'DeepTreeNode'
local cmd = torch.CmdLine()
cmd:option('-preconv', 3, '#layers before branching')
cmd:option('-depth', 4, 'depth of tree. total #nodes = 2^depth')
cmd:option('-channels', 8, '#channels in conv layers')
cmd:option('-nodedepth', 1, '#conv layers per node')
cmd:option('-classes', 10, '#output classes')
cmd:option('-pool', '3,5', 'before which depths to add pooling layers (1-indexed)')

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
    preconv:add(nn.SpatialConvolution(c_in, c, 3, 3, 1, 1, 1, 1):noBias())
    preconv:add(nn.SpatialBatchNormalization(c))
    preconv:add(nn.ReLU(true))
    c_in = c
    d = d+1
  end
  m:add(preconv)

  if tableContains(m.modelOpts.pool, d) then
    m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    size = size / 2
  end
  local root = DeepTreeNode(m.modelOpts.nodedepth, c_in, c, size, size)
  m:add(root)
  m.leaves = {root}
  m.nodes = {root}
  d = d+1

  for i=2,m.modelOpts.depth-1 do
    if tableContains(m.modelOpts.pool, d) then
      size = size / 2
    end

    local newleaves = {}
    for _, leaf in pairs(m.leaves) do
      local pool = tableContains(m.modelOpts.pool, d)
      local l = DeepTreeNode(m.modelOpts.nodedepth, m.modelOpts.channels, m.modelOpts.channels, size, size, pool)
      local r = DeepTreeNode(m.modelOpts.nodedepth, m.modelOpts.channels, m.modelOpts.channels, size, size, pool)
      leaf:addChildren({l, r})
      newleaves[#newleaves+1] = l
      m.nodes[#m.nodes+1] = l
      newleaves[#newleaves+1] = r
      m.nodes[#m.nodes+1] = r
    end
    m.leaves = newleaves
    d = d+1
  end

  if tableContains(m.modelOpts.pool, d) then
    size = size / 2
  end

  for _, leaf in pairs(m.leaves) do
    local l = nn.Sequential()
    local r = nn.Sequential()
    if tableContains(m.modelOpts.pool, d) then
      l:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      r:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    end
    for i=1,m.modelOpts.nodedepth do
      l:add(nn.SpatialConvolution(m.modelOpts.channels, m.modelOpts.channels, 3, 3, 1, 1, 1, 1):noBias())
      l:add(nn.SpatialBatchNormalization(m.modelOpts.channels))
      l:add(nn.ReLU(true))
      r:add(nn.SpatialConvolution(m.modelOpts.channels, m.modelOpts.channels, 3, 3, 1, 1, 1, 1):noBias())
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
    leaf:addChildren({l, r})
  end

  return m
end
return createModel
