import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import AuthShell, { AuthConfigBanner, AuthError, AuthField, GoogleIcon } from './AuthShell';
import { useAuth } from '../../contexts/AuthContext';
import { signInWithGoogle, signUpWithEmail } from '../../services/authService';
import { authErrorMessage } from '../../utils/authErrorMessage';

export default function SignUpPage() {
  const navigate = useNavigate();
  const { configured } = useAuth();

  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const finish = () => navigate('/', { replace: true });

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters.');
      return;
    }

    setBusy(true);
    try {
      await signUpWithEmail(email, password, name);
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
      title="Create account"
      subtitle="Get started with canvas. Save projects, templates, and agent sessions."
      footer={
        <>
          Already have an account? <Link to="/login">Sign in</Link>
        </>
      }
    >
      {!configured && <AuthConfigBanner />}

      <form onSubmit={handleSignUp}>
        <AuthField
          id="signup-name"
          label="Name"
          value={name}
          onChange={setName}
          placeholder="Your name"
          autoComplete="name"
        />
        <AuthField
          id="signup-email"
          label="Email"
          type="email"
          value={email}
          onChange={setEmail}
          placeholder="you@example.com"
          autoComplete="email"
          required
        />
        <AuthField
          id="signup-password"
          label="Password"
          type="password"
          value={password}
          onChange={setPassword}
          placeholder="At least 6 characters"
          autoComplete="new-password"
          required
        />
        <AuthField
          id="signup-confirm"
          label="Confirm password"
          type="password"
          value={confirmPassword}
          onChange={setConfirmPassword}
          placeholder="Repeat password"
          autoComplete="new-password"
          required
        />
        <AuthError message={error} />
        <button type="submit" className="auth-primary-btn" disabled={busy || !configured}>
          {busy ? 'Creating account…' : 'Create account'}
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
