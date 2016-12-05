package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'Model'

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to train')
cmd:argument('-input', 'input file or folder')
cmd:argument('-output', 'path to save trained model')
cmd:option('-pretrainEpochs', 1, 'number of epochs to pretrain each layer for')
defineBaseOptions(cmd)      -- defined in Utils.lua
defineTrainingOptions(cmd)  -- defined in Utils.lua
processArgs(cmd)

local model = Model(opts.model)
model.pretraining = true
model.modelOpts = model.module.modelOpts
modelOpts = model.modelOpts

local epochs = opts.epochs
local val = opts.val
opts.epochs = opts.pretrainEpochs
opts.val = ""
for i=2,modelOpts.depth do
  print("Pre-training tree layer ", i-1)
  print(model)
  model:Train()

  if i < modelOpts.depth then
    if tableContains(modelOpts.pool, model.module.depth) then
      model.module.size = model.module.size / 2
    end

    local newleaves = {}
    for _, leaf in pairs(model.module.leaves) do
      model.module.nodes[#model.module.nodes+1] = leaf
      local pool = tableContains(modelOpts.pool, model.module.depth)
      local l = DeepTreeNode(modelOpts.nodedepth, modelOpts.channels, modelOpts.channels, model.module.size, model.module.size, pool)
      local r = DeepTreeNode(modelOpts.nodedepth, modelOpts.channels, modelOpts.channels, model.module.size, model.module.size, pool)
      leaf:addChildren({l:cuda(), r:cuda()})
      newleaves[#newleaves+1] = l
      newleaves[#newleaves+1] = r
    end
    model.module.leaves = newleaves
    model.module.depth = model.module.depth + 1
    model:zeroGradParameters()
    model.params, model.gradParams = model:getParameters()
  end
end

if tableContains(modelOpts.pool, model.module.depth) then
  model.module.size = model.module.size / 2
end

for _, leaf in pairs(model.module.leaves) do
  model.module.nodes[#model.module.nodes+1] = leaf
  local l = nn.Sequential()
  local r = nn.Sequential()
  if tableContains(modelOpts.pool, model.module.depth) then
    l:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    r:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  end
  for i=1,modelOpts.nodedepth do
    l:add(nn.SpatialConvolution(modelOpts.channels, modelOpts.channels, 3, 3, 1, 1, 1, 1):noBias())
    l:add(nn.SpatialBatchNormalization(modelOpts.channels))
    l:add(nn.ReLU(true))
    r:add(nn.SpatialConvolution(modelOpts.channels, modelOpts.channels, 3, 3, 1, 1, 1, 1):noBias())
    r:add(nn.SpatialBatchNormalization(modelOpts.channels))
    r:add(nn.ReLU(true))
  end
  l:add(nn.SpatialAveragePooling(model.module.size, model.module.size))
  l:add(nn.View(-1, modelOpts.channels))
  l:add(nn.Linear(modelOpts.channels, modelOpts.classes))
  l:add(nn.SoftMax())
  r:add(nn.SpatialAveragePooling(model.module.size, model.module.size))
  r:add(nn.View(-1, modelOpts.channels))
  r:add(nn.Linear(modelOpts.channels, modelOpts.classes))
  r:add(nn.SoftMax())
  cudnn.convert(l, cudnn)
  cudnn.convert(r, cudnn)
  leaf:addChildren({l:cuda(), r:cuda()})
end
model.module.depth = model.module.depth + 1
model:zeroGradParameters()
model.params, model.gradParams = model:getParameters()

model.pretraining = false
opts.epochs = epochs
opts.val = val

print("Training model")
print(model)
model:Train()
print("Done!\n")
