import * as os from 'node:os';
import { execFile } from 'node:child_process';
import type { ExecutionResources } from '@tensorviz/contracts/protocol';

// Driver enumeration only. No project imports, Torch imports, context creation,
// tensor allocation or kernels. UUIDs join CUDA ordinals to NVML observations.
const CUDA_DEVICES = `import ctypes as c, json, sys, uuid
out=[]
try:
 d=c.CDLL('nvcuda.dll' if sys.platform=='win32' else 'libcuda.so.1')
 if d.cuInit(0)!=0: raise RuntimeError()
 count=c.c_int()
 if d.cuDeviceGetCount(c.byref(count))!=0: raise RuntimeError()
 for i in range(min(count.value,128)):
  dev=c.c_int(); name=c.create_string_buffer(512); uid=c.create_string_buffer(16)
  if d.cuDeviceGet(c.byref(dev),i)!=0 or d.cuDeviceGetName(name,512,dev)!=0: continue
  get_uuid=getattr(d,'cuDeviceGetUuid_v2',None)
  identity=''
  if get_uuid is not None and get_uuid(c.byref(uid),dev)==0: identity='GPU-'+str(uuid.UUID(bytes=uid.raw))
  out.append({'id':'cuda:'+str(i),'name':name.value.decode('utf-8','replace'),'uuid':identity})
except (OSError, AttributeError, RuntimeError): pass
print(json.dumps(out))`;
function command(file: string, args: string[], signal: AbortSignal): Promise<string> {
  return new Promise(resolve => {
    // Not execFile's signal option: aborting before a missing tool reports ENOENT
    // kills pid 0, the whole process group. Only a started child is stopped.
    const stop = () => { if (child.pid) child.kill(); };
    const child = execFile(file, args, { timeout: 3000, maxBuffer: 128 * 1024, windowsHide: true, encoding: 'utf8' },
      (error, stdout) => { signal.removeEventListener('abort', stop); resolve(error ? '' : stdout); });
    if (signal.aborted) stop(); else signal.addEventListener('abort', stop, { once: true });
  });
}
const metric = (value: string | undefined, scale = 1): number | null => {
  if (!value || !/^\d+(\.\d+)?$/.test(value.trim())) return null;
  const n = Number(value) * scale;
  return Number.isFinite(n) && n >= 0 ? n : null;
};
export function resourceDevices(cuda: string, nvml: string): ExecutionResources['devices'] {
  let devices: Array<{ id: string; name: string; uuid: string }>;
  try { devices = JSON.parse(cuda); } catch { return []; }
  if (!Array.isArray(devices)) return [];
  const observations = new Map(nvml.trim().split('\n').map(row => {
    const [uuid, utilization, free, total] = row.split(',').map(v => v.trim());
    return [uuid?.toLowerCase(), { utilization: metric(utilization), memoryFree: metric(free, 1024 * 1024), memoryTotal: metric(total, 1024 * 1024) }] as const;
  }));
  return devices.filter(d => d && /^cuda:[0-9]{1,3}$/.test(d.id) && typeof d.name === 'string' && typeof d.uuid === 'string').slice(0, 128).map(d => ({
    id: d.id, name: d.name.slice(0, 512), ...(observations.get(d.uuid.toLowerCase()) ?? { utilization: null, memoryFree: null, memoryTotal: null }),
  }));
}
const cpuTicks = () => os.cpus().reduce((sum, cpu) => ({ idle: sum.idle + cpu.times.idle,
  total: sum.total + Object.values(cpu.times).reduce((n, t) => n + t, 0) }), { idle: 0, total: 0 });
export async function executionResources(interpreter: string, remote: boolean, signal: AbortSignal): Promise<ExecutionResources> {
  const first = cpuTicks();
  const [cuda, nvml] = await Promise.all([
    interpreter ? command(interpreter, ['-I', '-c', CUDA_DEVICES], signal) : Promise.resolve(''),
    command('nvidia-smi', ['--query-gpu=uuid,utilization.gpu,memory.free,memory.total', '--format=csv,noheader,nounits'], signal),
    new Promise(resolve => setTimeout(resolve, 150)),
  ]);
  signal.throwIfAborted();
  const last = cpuTicks(), total = last.total - first.total;
  return { sampledAt: Date.now(), host: os.hostname(), remote, cpu: os.cpus()[0]?.model ?? 'CPU', cores: os.cpus().length,
    cpuPercent: total > 0 ? Math.max(0, Math.min(100, 100 * (1 - (last.idle - first.idle) / total))) : null,
    memoryFree: os.freemem(), memoryTotal: os.totalmem(), devices: [...resourceDevices(cuda, nvml), ...(process.platform === 'darwin' && process.arch === 'arm64' ? [{ id: 'mps', name: 'Apple GPU (runtime support checked on Run)', utilization: null, memoryFree: null, memoryTotal: null }] : [])],
    detail: 'Host observations, not reserved capacity or a prediction of model fit. Missing GPU readings are unknown. Runtime and model device support are checked when you run.' };
}
