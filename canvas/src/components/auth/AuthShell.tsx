import React from 'react';
import { Cpu } from 'lucide-react';
import { Link } from 'react-router-dom';
import '../../styles/auth.css';
import '../../styles/ui2.tokens.css';

type AuthShellProps = {
  title: string;
  subtitle: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
};

export default function AuthShell({ title, subtitle, children, footer }: AuthShellProps) {
  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-brand">
          <span className="auth-brand-mark">
            <Cpu size={18} strokeWidth={2} aria-hidden />
          </span>
          <span className="auth-wordmark">canvas</span>
        </div>

        <h1 className="auth-title">{title}</h1>
        <p className="auth-subtitle">{subtitle}</p>

        {children}

        {footer && <div className="auth-footer">{footer}</div>}
      </div>
    </div>
  );
}

export function AuthConfigBanner() {
  return (
    <div className="auth-config-banner" role="status">
      Firebase is not configured. Copy <code>.env.example</code> to <code>.env.local</code> and add your
      web app keys from the{' '}
      <a
        className="auth-link"
        href="https://console.firebase.google.com/project/tensorviz-98d48/settings/general"
        target="_blank"
        rel="noopener noreferrer"
      >
        Firebase console
      </a>
      .
    </div>
  );
}

/* All three screens render their failure the same way and in the same place —
   immediately above their submit button — so the message is never a toast that
   can be missed. Returns null rather than an empty node so the 12px gap under
   it only exists when there is something to say. */
export function AuthError({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div className="auth-error" role="alert">
      {message}
    </div>
  );
}

export function GoogleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden>
      <path
        fill="#4285F4"
        d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.71v2.26h2.92a8.78 8.78 0 0 0 2.68-6.61z"
      />
      <path
        fill="#34A853"
        d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.92-2.26c-.8.54-1.84.86-3.04.86-2.34 0-4.32-1.58-5.03-3.71H.96v2.33A9 9 0 0 0 9 18z"
      />
      <path
        fill="#FBBC05"
        d="M3.97 10.71A5.41 5.41 0 0 1 3.68 9c0-.59.1-1.16.29-1.71V4.96H.96a9 9 0 0 0 0 8.08l3.01-2.33z"
      />
      <path
        fill="#EA4335"
        d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58C13.47.89 11.43 0 9 0A9 9 0 0 0 .96 4.96l3.01 2.33C4.68 5.16 6.66 3.58 9 3.58z"
      />
    </svg>
  );
}

export function AuthField({
  id,
  label,
  type = 'text',
  value,
  onChange,
  placeholder,
  autoComplete,
  required,
}: {
  id: string;
  label: string;
  type?: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  autoComplete?: string;
  required?: boolean;
}) {
  return (
    <div className="auth-field">
      <label className="auth-label ui2-eyebrow" htmlFor={id}>
        {label}
      </label>
      <input
        id={id}
        className="auth-input"
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        autoComplete={autoComplete}
        required={required}
      />
    </div>
  );
}

export { Link };
