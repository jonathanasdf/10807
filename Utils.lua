function defineBaseOptions(cmd)
  cmd:option('-batchSize', 32, 'batch size')
  cmd:option('-epochSize', -1, 'num batches per epochs. -1 means run all available data once')
end

function defineTrainingOptions(cmd)
  cmd:option('-learningRate', 0.001, 'learning rate')
  cmd:option('-learningRateDecay', 0, 'learning rate decay')
  cmd:option('-LRDropEvery', -1, 'reduce learning rate every n epochs')
  cmd:option('-LRDropFactor', 10, 'factor to reduce learning rate')
  cmd:option('-momentum', 0.9, 'momentum')
  cmd:option('-nesterov', 1, 'use nesterov momentum')
  cmd:option('-weightDecay', 0.0005, 'weight decay')
  cmd:option('-inputWeights', '', 'semicolon separated weights to balance input classes for each batch')
  cmd:option('-epochs', 50, 'num epochs')
  cmd:option('-cacheEvery', 20, 'save model every n epochs. Set to -1 or a value >epochs to disable')
  cmd:option('-keepCaches', false, 'keep all of the caches, instead of just the most recent one')
  cmd:option('-val', '', 'validation data')
  cmd:option('-valBatchSize', -1, 'batch size for validation')
  cmd:option('-valSize', -1, 'num batches to validate. -1 means run all available data once')
  cmd:option('-valEvery', 1, 'run validation every n epochs')
  cmd:option('-valLossMultiplier', 1, 'multiply val loss by this')
  cmd:option('-optimState', '', 'optimState to resume from')
end

function processArgs(cmd)
  torch.setdefaulttensortype('torch.FloatTensor')
  torch.setnumthreads(1)
  opts = cmd:parse(arg or {})
  opts.pid = require("posix").getpid("pid")
  print("Process", opts.pid, "started!")

  if opts.val and opts.val ~= '' then
    if opts.valBatchSize == -1 then
      opts.valBatchSize = opts.batchSize
    end
  end

  if opts.optimState and opts.optimState ~= '' then
    local optimState = torch.load(opts.optimState)
    opts.learningRate = opts.optimState.learningRate
    opts.learningRateDecay  = opts.optimState.learningRateDecay
    opts.momentum = opts.optimState.momentum
    opts.weightDecay = opts.optimState.weightDecay
    opts.nesterov = opts.optimState.nesterov
  end
  if opts.nesterov == 1 then
    opts.dampening = 0.0
  end

  if opts.input then
    opts.input = opts.input:split(';')
  end
  if opts.val and opts.val ~= '' then
    opts.val = opts.val:split(';')
  end
  if opts.inputWeights then
    opts.inputWeights = opts.inputWeights:split(';')
    for i=1,#opts.inputWeights do
      opts.inputWeights[i] = tonumber(opts.inputWeights[i])
    end
  end

  opts.timestamp = os.date("%Y%m%d_%H%M%S")
  if opts.output and opts.output ~= '' then
    opts.dirname = paths.dirname(opts.output) .. '/'
    opts.basename = paths.basename(opts.output, paths.extname(opts.output))
    opts.backupdir = opts.dirname .. 'backups/'
    paths.mkdir(opts.backupdir)
    if opts.keepCaches then
      opts.cachedir = opts.backupdir .. opts.basename .. '_cache/' .. opts.timestamp .. '/'
      paths.mkdir(opts.cachedir)
    end
    opts.logdir = opts.dirname .. 'logs/' .. opts.basename .. '_' .. opts.timestamp .. '/'
    paths.mkdir(opts.logdir)
    cmd:log(opts.logdir .. 'log.txt', opts)
    cmd:addTime()

    opts.lossGraph = gnuplot.pngfigure(opts.logdir .. 'loss.png')
    gnuplot.xlabel('epoch')
    gnuplot.ylabel('loss')
    gnuplot.grid(true)
  else
    for k,v in pairs(opts) do
      print(k, v)
    end
  end
end

function setPhase(phase)
  assert(phase == 'train' or phase == 'test' or phase == 'val')
  opts.phase = phase
end

function requirePath(path)
  local oldPackagePath = package.path
  package.path = paths.dirname(path) .. '/?.lua;' .. package.path
  local M = require(paths.basename(path, 'lua'))
  package.path = oldPackagePath
  return M
end

function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

function tableContains(T, entry)
  for i=1,#T do
    if T[i] == entry then return true end
  end
  return false
end

function string:split(sep)
  sep = sep or ','
  local fields = {}
  local pattern = string.format('([^%s]+)', sep)
  self:gsub(pattern, function(c) fields[#fields+1] = c end)
  return fields
end

function readAll(file)
    local f = io.open(file, "rb")
    local content = f:read("*all")
    f:close()
    return content
end

function mkdir(file)
  local dirname = file
  if paths.extname(file) then
    dirname = paths.dirname(file)
  end
  os.execute('mkdir -p ' .. dirname)
end

function dirtree(dir)
  -- Code by David Kastrup
  require "lfs"
  assert(dir and dir ~= "", "directory parameter is missing or empty")
  if string.sub(dir, -1) == "/" then
    dir=string.sub(dir, 1, -2)
  end

  local function yieldtree(dir)
    for entry in lfs.dir(dir) do
      if entry ~= "." and entry ~= ".." then
        entry=dir.."/"..entry
        local attr=lfs.attributes(entry)
        coroutine.yield(entry,attr)
        if attr.mode == "directory" then
          yieldtree(entry)
        end
      end
    end
  end

  return coroutine.wrap(function() yieldtree(dir) end)
end

function os.capture(cmd, raw)
  local f = assert(io.popen(cmd, 'r'))
  local s = assert(f:read('*a'))
  f:close()
  if raw then return s end
  s = string.gsub(s, '^%s+', '')
  s = string.gsub(s, '%s+$', '')
  return s
end

function runMatlab(cmd)
  return os.capture('matlab -nodisplay -nosplash -nodesktop -r "try, ' .. cmd .. ', catch ME, disp(getReport(ME)); exit, end, exit" | tail -n +10')
end

function findModuleByName(model, name)
  if model.modules then
    for i=1,#model.modules do
      local recur = findModuleByName(model.modules[i], name)
      if recur then return recur end
      if model.modules[i].name == name then
        return model.modules[i]
      end
    end
  end
  return nil
end

function setDropout(model, p)
  if p == nil or p == -1 then return end
  if torch.isTypeOf(model, 'nn.Dropout') then
    model:setp(p)
  elseif model.modules then
    for i=1,#model.modules do
      setDropout(model.modules[i], p)
    end
  end
end

function hasDropout(model)
  if torch.isTypeOf(model, 'nn.Dropout') then
    return true
  end
  if not(model.modules) then
    return false
  end
  for i=1,#model.modules do
    if hasDropout(model.modules[i]) then
      return true
    end
  end
  return false
end

function printOutputSizes(model)
  if model.output then
    print(model, #model.output)
  end
  if not(model.modules) then return end
  for i=1,#model.modules do
    printOutputSizes(model.modules[i])
  end
end

function bind(f, ...)
  local va = require 'vararg'
  local outer = va(...)
  local function closure(...)
    return f(va.concat(outer, va(...)))
  end
  return closure
end

function bindPost(f, arg)
  local function closure(...)
    local args = {...}
    args[#args+1] = arg
    return f(unpack(args))
  end
  return closure
end
