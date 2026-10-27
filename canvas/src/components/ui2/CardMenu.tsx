import React, { useEffect, useRef, useState } from 'react';
import { MoreHorizontal } from 'lucide-react';

export type CardMenuItem = {
  key: string;
  label: string;
  icon: React.ComponentType<{ size?: number }>;
  run: () => void;
  danger?: boolean;
};

const CARD_MENU_WIDTH = 212;

/**
 * The ⋯ overflow menu used by a Continue card and by a projects-table row.
 *
 * The panel is positioned fixed, not absolute as the toolbar and nav menus are.
 * .ui2-card sets overflow:hidden so the graph thumbnail is clipped to the card's
 * radius, and a menu absolutely positioned inside that card would be sliced off
 * at its edge. The row case does not need that, but sharing one panel keeps the
 * two surfaces from drifting apart. Fixed coordinates are measured from the trigger on open, so the
 * menu closes on scroll and resize rather than drifting away from its button.
 *
 * role="menu" is load-bearing beyond accessibility: .ui2-panel[role="menu"] is
 * the selector that grants the float shadow and the strong border separating
 * the menu from the card underneath it.
 *
 * Escape closes it. The three menus already in the app do not handle Escape —
 * this deviates from them deliberately, because a menu that opens over a card
 * has no other keyboard-reachable way out.
 */
export default function CardMenu({
  label,
  items,
  triggerStyle,
  busy = false,
}: {
  label: string;
  items: CardMenuItem[];
  /** Where the trigger sits: pinned to a card's corner, or inline in a row's
   *  action cluster. The panel positions itself off the trigger either way. */
  triggerStyle?: React.CSSProperties;
  /** An action started from this menu is still running. The trigger is the only
   *  in-flight signal a row has, so it must not stay armed for a second click
   *  that would start the same export again. */
  busy?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [anchor, setAnchor] = useState<{ top: number; left: number } | null>(null);
  const trigger = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (busy) setOpen(false);
  }, [busy]);

  useEffect(() => {
    if (!open) return;
    const close = () => setOpen(false);
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
    };
    // Capture phase: the page scrolls in an inner container, not just on window.
    window.addEventListener('scroll', close, true);
    window.addEventListener('resize', close);
    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.removeEventListener('scroll', close, true);
      window.removeEventListener('resize', close);
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);

  const toggle = () => {
    const rect = trigger.current?.getBoundingClientRect();
    if (rect) {
      // Right-aligned to the trigger, clamped so a card at the left edge of a
      // narrow viewport cannot push the menu off-screen.
      setAnchor({ top: rect.bottom + 6, left: Math.max(8, rect.right - CARD_MENU_WIDTH) });
    }
    setOpen((value) => !value);
  };

  return (
    <>
      <button
        ref={trigger}
        className="ui2-icon-button"
        onClick={toggle}
        disabled={busy}
        aria-label={label}
        aria-busy={busy || undefined}
        aria-haspopup="menu"
        aria-expanded={open}
        title={label}
        style={{ width: 28, height: 28, ...triggerStyle }}
      >
        <MoreHorizontal size={13} />
      </button>
      {open && anchor && (
        <>
          <div className="fixed inset-0" style={{ zIndex: 40 }} onClick={() => setOpen(false)} />
          <div
            className="ui2-panel"
            role="menu"
            style={{
              position: 'fixed',
              top: anchor.top,
              left: anchor.left,
              minWidth: CARD_MENU_WIDTH,
              padding: 6,
              zIndex: 50,
            }}
          >
            {items.map((item) => (
              <button
                key={item.key}
                role="menuitem"
                className="ui2-menu-item"
                style={item.danger ? { color: 'var(--ui2-red)' } : undefined}
                onClick={() => {
                  setOpen(false);
                  item.run();
                }}
              >
                <item.icon size={14} /> {item.label}
              </button>
            ))}
          </div>
        </>
      )}
    </>
  );
}

