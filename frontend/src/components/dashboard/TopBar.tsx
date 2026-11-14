import { useEffect, useState } from 'react';
import { ChevronDown, ChevronLeft, ChevronRight, CircleUserRound, Key, Radio } from 'lucide-react';
import { CESIUM_ION_TOKEN } from '@/config/appConfig';
import { checkHealth, type HealthResponse } from '@/services/api';
import { cn } from '@/lib/utils';

interface TopBarProps {
  simulationId: string | null;
  locationName: string;
  collapsed: boolean;
  isDemoMode: boolean;
  activeView: 'forecast' | 'performance';
  onViewChange: (view: 'forecast' | 'performance') => void;
  onToggleCollapse: () => void;
}

export function applyRuntimeIonToken(token: string): void {
  window.sessionStorage.setItem('VARUNA_ION_TOKEN', token);
}

export function getRuntimeIonToken(): string {
  return window.sessionStorage.getItem('VARUNA_ION_TOKEN') || CESIUM_ION_TOKEN;
}

const NAV_ITEMS: { label: string; sublabel: string; view: 'forecast' | 'performance' }[] = [
  { label: 'Forecast', sublabel: 'Dashboard', view: 'forecast' },
  { label: 'Model performance', sublabel: 'Evaluation', view: 'performance' },
];

export default function TopBar({
  simulationId,
  locationName,
  collapsed,
  isDemoMode,
  activeView,
  onViewChange,
  onToggleCollapse,
}: TopBarProps) {
  const [tokenDialogOpen, setTokenDialogOpen] = useState(false);
  const [tokenInput, setTokenInput] = useState('');
  const [hasToken, setHasToken] = useState(() => Boolean(getRuntimeIonToken()));
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const now = new Date().toLocaleString('en-GB', {
    hour: '2-digit',
    minute: '2-digit',
    day: '2-digit',
    month: 'short',
  });

  useEffect(() => {
    let active = true;
    void checkHealth()
      .then((response) => {
        if (active) setHealth(response);
      })
      .catch(() => {
        if (active) setHealth({ status: 'down', version: 'unknown', solver: 'unavailable' });
      });
    return () => {
      active = false;
    };
  }, []);

  const connectionLabel = isDemoMode
    ? 'DEMO DATA'
    : health?.status === 'ok'
      ? 'API ONLINE'
      : health?.status === 'degraded'
        ? 'API DEGRADED'
        : 'API OFFLINE';

  const saveToken = () => {
    const trimmed = tokenInput.trim();
    if (trimmed) {
      applyRuntimeIonToken(trimmed);
      setHasToken(true);
      setTokenDialogOpen(false);
      setTokenInput('');
      window.location.reload();
    }
  };

  return (
    <header className="relative z-20 flex h-11 shrink-0 items-stretch border-b border-[var(--varuna-border)] bg-black/90 backdrop-blur-xl">
      <div className="flex w-[86px] shrink-0 items-center border-r border-[var(--varuna-border)] px-5">
        <div className="flex items-center gap-2.5">
          <div>
            <p className="text-[15px] font-semibold tracking-[0.18em] text-white">VARUNA</p>
            <p className="text-[8px] text-[#7e8b96]">Urban water intelligence</p>
          </div>
        </div>
      </div>

      <nav className="flex min-w-0 flex-1 items-stretch">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.label}
            onClick={() => onViewChange(item.view)}
            className={cn(
              'my-0 flex min-w-[96px] flex-col justify-center border-b-2 border-transparent px-5 text-left transition-colors hover:bg-white/[0.04]',
              activeView === item.view && 'border-white bg-white/[0.04]'
            )}
          >
            <span className={cn('text-[12px] font-medium', activeView === item.view ? 'text-white' : 'text-[#9a9a9a]')}>
              {item.label}
            </span>
            <span className="text-[10px] text-[#666]">{item.sublabel}</span>
          </button>
        ))}
      </nav>

      <div className="flex shrink-0 items-center gap-3 px-3">
        <div className="hidden text-right xl:block">
          <p className="font-tabular text-[10px] text-[#9a9a9a]">{now} IST</p>
          {simulationId && <p className="font-mono-num text-[9px] text-[#666]">{simulationId}</p>}
        </div>
        <button
          onClick={() => setTokenDialogOpen(true)}
          className={cn(
            'flex items-center gap-1.5 border border-transparent px-1 text-[10px] transition-colors',
            hasToken
              ? 'text-[#b8ff24]'
              : 'text-[#9a9a9a]'
          )}
          title="Configure the Cesium Ion token for real terrain and buildings"
        >
          <Radio size={11} />
          {hasToken ? 'ION CONNECTED' : 'ION FALLBACK'}
        </button>
        <span className={cn(
          'hidden border-l border-[var(--varuna-border)] pl-3 text-[10px] xl:inline',
          isDemoMode || health?.status === 'ok' ? 'text-[#9a9a9a]' : 'text-white'
        )}>
          {isDemoMode ? 'Demo data, synthetic' : connectionLabel}
        </span>
        <div className="hidden items-center gap-2 border-l border-[var(--varuna-border)] pl-3 text-[#9a9a9a] xl:flex">
          <span className="font-tabular text-[10px]">{locationName} · {now} IST</span>
        </div>
        <div className="flex items-center gap-1.5 border-l border-[var(--varuna-border)] pl-3 text-[#9a9a9a]">
          <CircleUserRound size={17} />
          <span className="hidden text-[11px] sm:inline">Operator</span>
          <ChevronDown size={13} className="text-[#858892]" />
        </div>
        <button
          onClick={onToggleCollapse}
          className="flex h-7 w-7 items-center justify-center rounded-sm border border-[var(--varuna-border-strong)] text-[#9a9a9a] transition hover:border-white hover:text-white"
          aria-label={collapsed ? 'Expand left sidebar' : 'Collapse left sidebar'}
        >
          {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
        </button>
      </div>

      {tokenDialogOpen && (
        <div
          className="absolute inset-0 z-30 flex items-start justify-center bg-black/80 pt-24"
          onClick={() => setTokenDialogOpen(false)}
        >
          <div
            className="w-[420px] rounded-md border border-[var(--varuna-border-strong)] bg-[#050505] p-5"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-2 text-sm font-semibold text-[#f0f1f3]">
              <Key size={15} className="text-white" /> Cesium Ion Access Token
            </div>
            <p className="mt-2 text-xs leading-relaxed text-[#b5b8c0]">
              Real 3D terrain and OpenStreetMap buildings stream from Cesium Ion and need a free access token from{' '}
              <a href="https://ion.cesium.com/tokens" target="_blank" rel="noreferrer" className="text-white underline">
                ion.cesium.com/tokens
              </a>
              . Without one, VARUNA uses fallback imagery and synthetic terrain.
            </p>
            <input
              type="password"
              value={tokenInput}
              onChange={(e) => setTokenInput(e.target.value)}
              placeholder="Paste your Ion token"
              className="mt-3 w-full rounded-md border border-[var(--varuna-border-strong)] bg-black px-3 py-2 font-mono-num text-xs text-white placeholder:text-[#666] focus:border-white focus:outline-none"
            />
            <div className="mt-4 flex justify-end gap-2">
              <button onClick={() => setTokenDialogOpen(false)} className="rounded-sm border border-[#626570] px-3 py-1.5 text-xs text-[#d5d7dc] hover:bg-[#454751]">
                Cancel
              </button>
              <button onClick={saveToken} className="rounded-md bg-white px-3 py-1.5 text-xs font-semibold text-black hover:bg-[#ddd]">
                Save and reload
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
