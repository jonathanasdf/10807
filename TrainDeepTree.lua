package.path = package.path .. ';/home/jshen/scripts/?.lua'

require 'Model'

local cmd = torch.CmdLine()
cmd:argument('-model', 'model to train')
cmd:argument('-input', 'input file or folder')
cmd:argument('-output', 'path to save trained model')
defineBaseOptions(cmd)      -- defined in Utils.lua
defineTrainingOptions(cmd)  -- defined in Utils.lua
processArgs(cmd)

local model = Model(opts.model)

model.pretraining = true
for i=2:model.modelOpts.depth-1 do
  model:train()

  if tableContains(model.modelOpts.pool, model.depth) then
    model.size = model.size / 2
  end

  local newleaves = {}
  for j=1:#model.leaves do
    local l = DeepTreeNode(model.modelOpts.nodedepth, model.modelOpts.channels, model.modelOpts.channels, model.size, model.size)
    local r = DeepTreeNode(model.modelOpts.nodedepth, model.modelOpts.channels, model.modelOpts.channels, model.size, model.size)
    newleaves[#newleaves+1] = l
    newleaves[#newleaves+1] = r
    if tableContains(model.modelOpts.pool, model.depth) then
      l = nn.Sequential():add(nn.SpatialMaxPooling(2, 2, 2, 2)):add(l)
      r = nn.Sequential():add(nn.SpatialMaxPooling(2, 2, 2, 2)):add(r)
    end
    model.leaves[j]:addChildren({l:cuda(), r:cuda()})
  end
  model.leaves = newleaves
  model.depth = model.depth + 1
end

model:train()

if tableContains(model.modelOpts.pool, model.depth) then
  model.size = model.size / 2
end
  
local newleaves = {}
for j=1:#model.leaves do
  local l = nn.Sequential()
  local r = nn.Sequential()
  newleaves[#newleaves+1] = l
  newleaves[#newleaves+1] = r
  if tableContains(model.modelOpts.pool, model.depth) then
    l:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    r:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  end
  for i=1,model.modelOpts.nodedepth do
    l:add(nn.SpatialConvolution(model.modelOpts.channels, model.modelOpts.channels, 3, 3, 1, 1, 1, 1):noBias())
    l:add(nn.SpatialBatchNormalization(model.modelOpts.channels))
    l:add(nn.ReLU(true))
    r:add(nn.SpatialConvolution(model.modelOpts.channels, model.modelOpts.channels, 3, 3, 1, 1, 1, 1):noBias())
    r:add(nn.SpatialBatchNormalization(model.modelOpts.channels))
    r:add(nn.ReLU(true))
  end
  l:add(nn.SpatialAveragePooling(model.size, model.size))
  l:add(nn.View(-1, model.modelOpts.channels))
  l:add(nn.Linear(model.modelOpts.channels, model.modelOpts.classes))
  r:add(nn.SpatialAveragePooling(model.size, model.size))
  r:add(nn.View(-1, model.modelOpts.channels))
  r:add(nn.Linear(model.modelOpts.channels, model.modelOpts.classes))
  model.leaves[j]:addChildren({l:cuda(), r:cuda()})
end
model.leaves = newleaves
model.depth = model.depth + 1

model.pretraining = false
model:train()
print("Done!\n")
