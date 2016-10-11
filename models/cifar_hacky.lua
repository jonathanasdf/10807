local cmd = torch.CmdLine()
cmd:option('-preconv', 3, '#layers before branching')
cmd:option('-depth', 4, 'depth of tree. total #nodes = 2^depth')
cmd:option('-channels', 8, '#channels in conv layers')
cmd:option('-nodedepth', 1, '#conv layers per node')
cmd:option('-classes', 10, '#output classes')
cmd:option('-pool', '3,5', 'before which depths to add pooling layers (1-indexed)')

-- Input: {inputs, prob}
-- Output: {{inputs, left_prob}, {inputs, right_prob}}
local function DeepTreeNode(c_in, c, size, n)
  local m = nn.Sequential()

  local conv = nn.Sequential()
  for i=1,n do
    conv:add(nn.SpatialConvolution(c_in, c, 3, 3, 1, 1, 1, 1):noBias())
    conv:add(nn.SpatialBatchNormalization(c))
    conv:add(nn.ReLU(true))
    c_in = c
  end

  m:add(nn.ParallelTable()
       :add(nn.Sequential()
           :add(conv)
           :add(nn.ConcatTable()
                :add(nn.Identity())
                :add(nn.Sequential()
                     :add(nn.View(-1, c*size*size))
                     :add(nn.Linear(c*size*size, 1))
                     :add(nn.Sigmoid()))))
       :add(nn.Identity()))
  m:add(nn.FlattenTable())
  -- Output: {inputs, weights, prob}

  local l = nn.ConcatTable()
      :add(nn.SelectTable(1))
      -- left_prob = weight * prob
      :add(nn.Sequential()
          :add(nn.ConcatTable()
              :add(nn.SelectTable(2))
              :add(nn.SelectTable(3)))
          :add(nn.CMulTable()))

  local r = nn.ConcatTable()
      :add(nn.SelectTable(1))
      -- right_prob = (1-weight) * prob
      :add(nn.Sequential()
          :add(nn.ConcatTable()
              :add(nn.Sequential()
                  :add(nn.SelectTable(2))
                  :add(nn.MulConstant(-1))
                  :add(nn.AddConstant(1)))
              :add(nn.SelectTable(3)))
          :add(nn.CMulTable()))

  m:add(nn.ConcatTable():add(l):add(r))
  return m
end

-- Input: {image, 1}
local function createModel(modelOpts)
  modelOpts = cmd:parse(modelOpts and modelOpts:split(' =') or {})
  modelOpts.pool = modelOpts.pool:split(',')
  for i=1,#modelOpts.pool do
    modelOpts.pool[i] = tonumber(modelOpts.pool[i])
  end

  local c_in = 3
  local c = modelOpts.channels

  local m = nn.Sequential()
  local size = 32
  local d = 1

  local preconv = nn.Sequential()
  for i=1,modelOpts.preconv do
    if tableContains(modelOpts.pool, d) then
      preconv:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      size = size / 2
    end
    preconv:add(nn.SpatialConvolution(c_in, c, 3, 3, 1, 1, 1, 1):noBias())
    preconv:add(nn.SpatialBatchNormalization(c))
    preconv:add(nn.ReLU(true))
    c_in = c
    d = d+1
  end
  m:add(nn.ConcatTable():add(nn.ParallelTable():add(preconv):add(nn.Identity())))

  local n = 1
  for i=1,modelOpts.depth do
    local t = nn.ParallelTable()
    if tableContains(modelOpts.pool, d) then
      size = size / 2
    end
    for j=1,n do
      local s = t
      if tableContains(modelOpts.pool, d) then
        s = nn.Sequential()
        t:add(s)
        s:add(nn.ParallelTable()
             :add(nn.SpatialMaxPooling(2, 2, 2, 2))
             :add(nn.Identity()))
      end
      s:add(DeepTreeNode(c_in, c, size, modelOpts.nodedepth))
    end
    m:add(t)

    -- hacky one-layer flatten
    local f = nn.ConcatTable()
    for j=1,n do
      f:add(nn.Sequential():add(nn.SelectTable(j)):add(nn.SelectTable(1)))
      f:add(nn.Sequential():add(nn.SelectTable(j)):add(nn.SelectTable(2)))
    end
    m:add(f)

    c_in = c
    n = n*2
    d = d+1
  end

  -- Input: {{inputs, prob}, ...}
  local t = nn.ParallelTable()
  for i=1,n do
    t:add(nn.ParallelTable()
         :add(nn.Sequential()
             :add(nn.SpatialAveragePooling(size, size))
             :add(nn.View(-1, c))
             :add(nn.Linear(c, modelOpts.classes))
             :add(nn.SoftMax()))
         :add(nn.Identity()))
    -- Output: {{label_dist, leaf_prob}, ...}
  end
  m:add(t)

  for _, v in pairs(m:findModules('nn.SpatialConvolution')) do
    v.weight:normal(0,math.sqrt(2/v.kW*v.kH*v.nInputPlane))
  end
  for _, v in pairs(m:findModules('nn.Linear')) do
    v.bias:zero()
  end

  return m
end
return createModel
