import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import AuthShell, { AuthConfigBanner, AuthError, AuthField } from './AuthShell';
import { useAuth } from '../../contexts/AuthContext';
import { sendResetPassword } from '../../services/authService';
import { authErrorMessage } from '../../utils/authErrorMessage';

export default function ForgotPasswordPage() {
  const { configured } = useAuth();
  const [email, setEmail] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);
  const [busy, setBusy] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await sendResetPassword(email);
      setSent(true);
    } catch (err) {
      setError(authErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthShell
      title="Forgot password"
      subtitle={
        sent
          ? 'If an account exists for that email, we sent a reset link. Check your inbox.'
          : 'Enter your email and we will send you a link to reset your password.'
      }
      footer={
        <>
          <Link to="/login">Back to sign in</Link>
        </>
      }
    >
      {!configured && <AuthConfigBanner />}

      {!sent && (
        <form onSubmit={handleSubmit}>
          <AuthField
            id="forgot-email"
            label="Email"
            type="email"
            value={email}
            onChange={setEmail}
            placeholder="you@example.com"
            autoComplete="email"
            required
          />
          <AuthError message={error} />
          <button type="submit" className="auth-primary-btn" disabled={busy || !configured}>
            {busy ? 'Sending…' : 'Send reset link'}
          </button>
        </form>
      )}
    </AuthShell>
  );
}
