--  Test to see if two dissimilar groups such as "cat dog" and "automobile truck" can be
--  correctly classified using small number of convolutions.

require 'nn'
local utils = paths.dofile'utils.lua'

local function createModel(opt)
   local model = nn.Sequential()

   -- building block
   local function Block(nInputPlane, nOutputPlane)
      model:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1):noBias())
      model:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
      model:add(nn.ReLU(true))
      return model
   end

   local function MP()
      model:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
      return model
   end

   local size = 32
   Block(3, 8)
   for i=1,opt.depth-1 do
--     MP()
--     size = size / 2
     Block(8, 8)
   end

   model:add(nn.SpatialConvolution(8, 2, 1, 1):noBias())
   model:add(nn.SpatialMaxPooling(size, size))

   utils.testModel(model)
   utils.MSRinit(model)

   return model
end

return createModel
