import { useEffect, useState } from 'react';
import { ChevronDown, ChevronLeft, ChevronRight, CircleUserRound, Key, Radio, Waves } from 'lucide-react';
import { CESIUM_ION_TOKEN } from '@/config/appConfig';
import { checkHealth, type HealthResponse } from '@/services/api';
import { cn } from '@/lib/utils';

interface TopBarProps {
  simulationId: string | null;
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
    <header className="relative z-20 flex h-[76px] shrink-0 items-stretch border-b border-[#18231e] bg-[radial-gradient(circle_at_56%_-100%,rgba(184,255,36,0.18),transparent_30%),#050707]/95 shadow-[0_8px_30px_rgba(0,0,0,0.38)] backdrop-blur-xl">
      <div className="flex w-[248px] shrink-0 items-center border-r border-[#18231e] px-6">
        <div className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl border border-[#b8ff24]/70 bg-[#b8ff24]/10 text-[#b8ff24] shadow-[0_0_20px_rgba(184,255,36,0.2)]">
            <Waves size={16} strokeWidth={2.5} />
          </span>
          <div>
            <p className="text-[16px] font-bold tracking-[0.24em] text-[#f2f3f4]">VARUNA</p>
            <p className="text-[9px] uppercase tracking-[0.2em] text-[#9eb0a4]">Urban water intelligence</p>
          </div>
        </div>
      </div>

      <nav className="flex min-w-0 flex-1 items-stretch">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.label}
            onClick={() => onViewChange(item.view)}
            className={cn(
              'my-3 flex min-w-[138px] flex-col justify-center rounded-xl px-5 text-left transition-colors hover:bg-[#111914]',
              activeView === item.view && 'border border-[#8fcf24]/60 bg-[#1a2a12] shadow-[0_0_20px_rgba(184,255,36,0.14),inset_0_0_18px_rgba(184,255,36,0.08)]'
            )}
          >
            <span className={cn('text-[12px] font-medium', activeView === item.view ? 'text-[#f0f1f3]' : 'text-[#c2c4c9]')}>
              {item.label}
            </span>
            <span className="text-[10px] text-[#858892]">{item.sublabel}</span>
          </button>
        ))}
      </nav>

      <div className="flex shrink-0 items-center gap-3 px-4">
        <div className="hidden text-right xl:block">
          <p className="font-mono-num text-[10px] text-[#c7c9ce]">{now} IST</p>
          {simulationId && <p className="font-mono-num text-[9px] text-[#858892]">{simulationId}</p>}
        </div>
        <button
          onClick={() => setTokenDialogOpen(true)}
          className={cn(
            'flex items-center gap-1.5 rounded-sm border px-2 py-1.5 text-[9px] font-semibold uppercase tracking-wider transition-colors',
            hasToken
              ? 'border-[#9ad52d]/50 bg-[#9ad52d]/10 text-[#b7eb62]'
              : 'border-[#f4ca35]/50 bg-[#f4ca35]/10 text-[#f4ca35]'
          )}
          title="Configure the Cesium Ion token for real terrain and buildings"
        >
          <Radio size={11} />
          {hasToken ? 'ION CONNECTED' : 'ION FALLBACK'}
        </button>
        <span className={cn(
          'hidden border-l border-[#202d2d] pl-3 font-mono-num text-[9px] uppercase tracking-[0.12em] xl:inline',
          isDemoMode || health?.status === 'ok' ? 'text-[#b8ff24]' : 'text-[#ff7777]'
        )}>
          {connectionLabel}
        </span>
        <div className="hidden items-center gap-2 border-l border-[#202d2d] pl-3 text-[#d4d6db] xl:flex">
          <span className="h-1.5 w-1.5 rounded-full bg-[#d7ff3f] shadow-[0_0_8px_#d7ff3f]" />
          <span className="font-mono-num text-[9px] uppercase tracking-[0.14em] text-[#b8c5bb]">Bengaluru / IN</span>
        </div>
        <div className="flex items-center gap-1.5 border-l border-[#202d2d] pl-3 text-[#d4d6db]">
          <CircleUserRound size={17} />
          <span className="hidden text-[11px] sm:inline">Operator</span>
          <ChevronDown size={13} className="text-[#858892]" />
        </div>
        <button
          onClick={onToggleCollapse}
          className="flex h-7 w-7 items-center justify-center rounded-sm border border-[#626570] text-[#c8cad0] transition hover:border-[#9ad52d] hover:text-[#b7eb62]"
          aria-label={collapsed ? 'Expand left sidebar' : 'Collapse left sidebar'}
        >
          {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
        </button>
      </div>

      {tokenDialogOpen && (
        <div
          className="absolute inset-0 z-30 flex items-start justify-center bg-[#191a20]/75 pt-24"
          onClick={() => setTokenDialogOpen(false)}
        >
          <div
            className="w-[420px] rounded-sm border border-[#626570] bg-[#3a3c46] p-5 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-2 text-sm font-semibold text-[#f0f1f3]">
              <Key size={15} className="text-[#9ad52d]" /> Cesium Ion Access Token
            </div>
            <p className="mt-2 text-xs leading-relaxed text-[#b5b8c0]">
              Real 3D terrain and OpenStreetMap buildings stream from Cesium Ion and need a free access token from{' '}
              <a href="https://ion.cesium.com/tokens" target="_blank" rel="noreferrer" className="text-[#b7eb62] underline">
                ion.cesium.com/tokens
              </a>
              . Without one, VARUNA uses fallback imagery and synthetic terrain.
            </p>
            <input
              type="password"
              value={tokenInput}
              onChange={(e) => setTokenInput(e.target.value)}
              placeholder="Paste your Ion token"
              className="mt-3 w-full rounded-sm border border-[#626570] bg-[#292b33] px-3 py-2 font-mono-num text-xs text-[#f0f1f3] placeholder:text-[#858892] focus:border-[#9ad52d] focus:outline-none"
            />
            <div className="mt-4 flex justify-end gap-2">
              <button onClick={() => setTokenDialogOpen(false)} className="rounded-sm border border-[#626570] px-3 py-1.5 text-xs text-[#d5d7dc] hover:bg-[#454751]">
                Cancel
              </button>
              <button onClick={saveToken} className="rounded-sm bg-[#9ad52d] px-3 py-1.5 text-xs font-semibold text-[#292b32] hover:bg-[#b7eb62]">
                Save and reload
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
