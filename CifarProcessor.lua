require 'TrueNLLCriterion'
local Processor = require 'Processor'
local M = torch.class('CifarProcessor', 'Processor')

function M:__init(model, processorOpts)
  self.cmd:option('-imageSize', 32, 'input image size')
  self.cmd:option('-flip', 0.5, '(training) probability to do horizontal flip')
  self.cmd:option('-minCropPercent', 1, '(training) minimum of original size to crop to for random cropping')
  self.cmd:option('-randomCrop', 0, '(training) number of pixels to pad for random crop')
  Processor.__init(self, model, processorOpts)

  local data = torch.load('/mnt/sdb1/jshen/cifar10/data.t7')
  self.data = data.data:float()
  self.label = data.labels:cuda()

  self.criterion = TrueNLLCriterion():cuda()

  if opts.logdir then
    self.graph = gnuplot.pngfigure(opts.logdir .. 'acc.png')
    gnuplot.xlabel('epoch')
    gnuplot.ylabel('acc')
    gnuplot.grid(true)
    self.trainAcc = torch.Tensor(opts.epochs)
    self.valAcc = torch.Tensor(opts.epochs)
  end
end

-- Random crop from larger image with optional padding
local function randomCrop(input, pad, mode)
  pad = pad or 'zero'
  mode = mode or 'zero'
  local w, h = input:size(3) + 2*pad, input:size(2) + 2*pad
  local a, b = math.random(), math.random()
  local x1, y1 = math.floor(a * (w - size + 1)), math.floor(b * (h - size + 1))

  if pad > 0 then
    local module
    if mode == 'reflection' then
      module = nn.SpatialReflectionPadding(pad, pad, pad, pad):float()
    elseif mode == 'zero' then
      module = nn.SpatialZeroPadding(pad, pad, pad, pad):float()
    else
      error('unknown mode ' .. mode)
    end
    input = module:forward(input)
  end
  return image.crop(input, x1, y1, x1 + size, y1 + size)
end

-- Random crop a percentage of the entire image
local function randomCropPercent(input, minSizePercent)
  local w, h = input:size(3), input:size(2)
  local p = math.random() * (1 - minSizePercent) + minSizePercent
  local a, b = math.random(), math.random()
  local pw, ph = math.ceil(p * w), math.ceil(p * h)
  local x1, y1 = math.floor(a * (w - pw)), math.floor(b * (h - ph))
  local out = image.crop(input, x1, y1, x1 + pw, y1 + ph)
  assert(out:size(2) == ph and out:size(3) == pw, 'wrong crop size')
  return out
end

local function flip(input, p)
  if torch.uniform() < p then
    input = image.hflip(input)
  end
  return input
end

function M:preprocess(path)
  local img = self.data[tonumber(path)]

  if opts.phase == 'train' then
    if self.randomCrop > 0 then
      img = randomCrop(img, self.randomCrop, 'reflection')
    end
    if self.minCropPercent < 1 then
      img = randomCropPercent(img, self.minCropPercent)
    end
    if self.flip > 0 then
      img = flip(img, self.flip)
    end
  end

  img = image.scale(img, self.imageSize, self.imageSize)
  return img:cuda()
end

function M:getLabels(pathNames, outputs)
  local labels = torch.Tensor(#pathNames):cuda()
  for i=1,#pathNames do
    labels[i] = self.label[tonumber(pathNames[i])]
  end
  return labels
end

function M:resetStats()
  self.stats = optim.ConfusionMatrix(10)
end

function M:updateStats(pathNames, outputs, labels)
  self.stats:batchAdd(outputs, labels)
end

function M:getStats()
  self.stats:updateValids()

  if self.graph then
    if opts.phase == 'train' then
      self.trainAcc[opts.epoch] = self.stats.averageValid
    elseif opts.phase == 'val' then
      self.valAcc[opts.epoch] = self.stats.averageValid
    end

    gnuplot.figure(self.graph)
    local x = torch.range(1, opts.epoch):long()
    if opts.epoch >= opts.valEvery then
      local x2 = torch.range(opts.valEvery, opts.epoch):long()
      gnuplot.plot({'train', x, self.trainAcc:index(1, x), '-'}, {'val', x2, self.valAcc:index(1, x2), '-'})
    else
      gnuplot.plot({'train', x, self.trainAcc:index(1, x), '-'})
    end
    gnuplot.plotflush()
  end
  return tostring(self.stats)
end

return M
