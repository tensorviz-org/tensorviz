import React, { useState } from 'react';
import { ChevronDown, LogOut } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { signOutUser } from '../../services/authService';
import UserAvatar from './UserAvatar';

type UserMenuProps = {
  /**
   * 'bar' (default) is the icon-only avatar for the top bar. 'rail' is the
   * sidebar foot's full identity row — avatar, name, chevron — which also has to
   * open UPWARD: the rail's foot sits at the bottom of an overflow-hidden root,
   * so a downward popover would be clipped and the sign-out unreachable.
   */
  placement?: 'bar' | 'rail';
};

export default function UserMenu({ placement = 'bar' }: UserMenuProps) {
  const { user, configured } = useAuth();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);

  if (!configured || !user) return null;

  const rail = placement === 'rail';

  const handleSignOut = async () => {
    setBusy(true);
    try {
      await signOutUser();
      navigate('/login', { replace: true });
    } finally {
      setBusy(false);
      setOpen(false);
    }
  };

  return (
    <div className={rail ? 'relative w-full' : 'relative'}>
      <button
        type="button"
        className={
          rail
            ? 'ui2-nav-item w-full flex items-center gap-2.5 px-2 py-2 text-left cursor-pointer'
            : 'p-0 border-0 bg-transparent cursor-pointer'
        }
        onClick={() => setOpen((v) => !v)}
        aria-label="Account menu"
        aria-haspopup="menu"
        aria-expanded={open}
        title={user.email ?? 'Account'}
      >
        <UserAvatar user={user} />
        {rail && (
          <>
            <span className="text-sm truncate min-w-0">
              {user.displayName || user.email?.split('@')[0] || 'Signed in'}
            </span>
            <ChevronDown size={13} className="ml-auto ui2-muted flex-shrink-0" />
          </>
        )}
      </button>

      {open && (
        <>
          <button
            type="button"
            className="fixed inset-0 z-40 cursor-default"
            aria-label="Close menu"
            onClick={() => setOpen(false)}
          />
          <div
            role="menu"
            className={
              rail
                ? 'absolute left-0 bottom-full mb-2 z-50 ui2-panel p-2 w-full min-w-[200px]'
                : 'absolute right-0 top-full mt-2 z-50 ui2-panel p-2 min-w-[200px]'
            }
            style={{ boxShadow: '0 12px 32px rgba(0,0,0,0.4)' }}
          >
            <div className="px-2 py-2 border-b mb-1 flex items-center gap-2" style={{ borderColor: 'var(--ui2-border)' }}>
              <UserAvatar user={user} className="ui2-avatar flex-shrink-0" />
              <div className="min-w-0">
                <div className="text-sm font-medium truncate">{user.displayName || 'Signed in'}</div>
                <div className="text-xs ui2-muted truncate">{user.email}</div>
              </div>
            </div>
            <button
              type="button"
              className="w-full ui2-panel p-2.5 flex items-center gap-2 text-sm text-left"
              onClick={handleSignOut}
              disabled={busy}
            >
              <LogOut size={14} />
              {busy ? 'Signing out…' : 'Sign out'}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
