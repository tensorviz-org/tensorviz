import React, { useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import AuthShell, { AuthConfigBanner, AuthError, AuthField, GoogleIcon } from './AuthShell';
import { useAuth } from '../../contexts/AuthContext';
import { signInWithEmail, signInWithGoogle } from '../../services/authService';
import { authErrorMessage } from '../../utils/authErrorMessage';

export default function LoginPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { configured } = useAuth();
  const redirectTo = searchParams.get('redirect') || '/';

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const finish = () => navigate(redirectTo, { replace: true });

  const handleEmailLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await signInWithEmail(email, password);
      finish();
    } catch (err) {
      setError(authErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const handleGoogle = async () => {
    setError(null);
    setBusy(true);
    try {
      await signInWithGoogle();
      finish();
    } catch (err) {
      setError(authErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthShell
      title="Sign in"
      subtitle="Projects sync to your account. The backend stays on your machine."
      footer={
        <>
          No account? <Link to="/signup">Create one</Link>
        </>
      }
    >
      {!configured && <AuthConfigBanner />}

      <form onSubmit={handleEmailLogin}>
        <AuthField
          id="login-email"
          label="Email"
          type="email"
          value={email}
          onChange={setEmail}
          placeholder="you@example.com"
          autoComplete="email"
          required
        />
        <AuthField
          id="login-password"
          label="Password"
          type="password"
          value={password}
          onChange={setPassword}
          placeholder="••••••••"
          autoComplete="current-password"
          required
        />
        <div className="auth-forgot-row">
          <Link className="auth-forgot" to="/forgot-password">
            Forgot password?
          </Link>
        </div>
        <AuthError message={error} />
        <button type="submit" className="auth-primary-btn" disabled={busy || !configured}>
          {busy ? 'Signing in…' : 'Sign in'}
        </button>
      </form>

      <div className="auth-divider">or</div>

      <button type="button" className="auth-google-btn" onClick={handleGoogle} disabled={busy || !configured}>
        <GoogleIcon />
        Continue with Google
      </button>
    </AuthShell>
  );
}
