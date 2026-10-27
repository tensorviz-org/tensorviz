import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import '../../styles/auth.css';

export default function RequireAuth({ children }: { children: React.ReactNode }) {
  const { user, loading, configured } = useAuth();
  const location = useLocation();

  if (!configured) {
    return <>{children}</>;
  }

  if (loading) {
    return <div className="auth-loading">Loading…</div>;
  }

  if (!user) {
    const redirect = encodeURIComponent(location.pathname + location.search);
    return <Navigate to={`/login?redirect=${redirect}`} replace />;
  }

  return <>{children}</>;
}

export function RedirectIfAuthed({ children }: { children: React.ReactNode }) {
  const { user, loading, configured } = useAuth();

  if (!configured) {
    return <>{children}</>;
  }

  if (loading) {
    return <div className="auth-loading">Loading…</div>;
  }

  if (user) {
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
}
