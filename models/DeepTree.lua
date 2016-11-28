require 'DeepTreeNode'

-- This model is for the version where examples are sent down to a single children depending on if split is <= 0.5.
-- Requires layer by layer pretraining to learn split functions, so only root node is created here.
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
  m.depth = d+1
  m.size = size

  return m
end
return createModel
